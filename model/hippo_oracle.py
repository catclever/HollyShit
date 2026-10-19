import torch
import torch.nn as nn
import numpy as np
import scipy.linalg as linalg

def construct_hippo_legs(N):
    """
    Constructs the exact HiPPO-LegS matrix A and B for Legendre polynomial projection.
    """
    A = np.zeros((N, N))
    B = np.zeros((N, 1))
    for i in range(N):
        for j in range(N):
            if i > j:
                A[i, j] = (2 * i + 1) ** 0.5 * (2 * j + 1) ** 0.5
            elif i == j:
                A[i, j] = i + 1
        B[i, 0] = (2 * i + 1) ** 0.5
    return -A, B

class HiPPOOracleCUDA(nn.Module):
    """
    A frozen, mathematically exact Full-Rank Causal HiPPO-LegS State-Space Operator.
    Rolls out the N-order Legendre differential equation c_t = A c_{t-1} + B x_t across all
    timesteps t in [0, L-1], preserving every intermediate state h_t via an exact L x L
    lower-triangular Causal Toeplitz matrix K_HiPPO (Rank = L, 0% information loss, 0 trainable params).
    """
    def __init__(self, hidden_dim: int, N: int = 64, dt: float = 1.0, memory_horizon: float = 64.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.N = N
        
        A_cont, B_cont = construct_hippo_legs(N)
        
        # Bilinear Discretization (Tustin's method) scaled by characteristic memory horizon
        dt_scaled = dt / float(max(memory_horizon, 1.0))
        I = np.eye(N)
        BL = linalg.inv(I - (dt_scaled / 2.0) * A_cont)
        A_disc = BL @ (I + (dt_scaled / 2.0) * A_cont)
        B_disc = (BL @ B_cont * dt_scaled).squeeze(-1)
        
        # Register as fixed buffers (0 learnable parameters, strictly frozen Oracle)
        self.register_buffer("A", torch.tensor(A_disc, dtype=torch.float32)) # (N, N)
        self.register_buffer("B", torch.tensor(B_disc, dtype=torch.float32)) # (N,)
        
        # Analytical Legendre boundary readout vector aligned with B_disc (guarantees k_0 > 0)
        b_tensor = torch.tensor(B_disc, dtype=torch.float32)
        readout = b_tensor / b_tensor.norm().clamp(min=1e-6)
        self.register_buffer("readout", readout.unsqueeze(-1)) # (N, 1)
        
        # Legacy deterministic projection for 2D pooled summary mode
        torch.manual_seed(42)
        self.register_buffer("proj_out", torch.randn(N, 1) / np.sqrt(N))
        
        # Cache for L x L Full-Rank Causal Toeplitz HiPPO Kernel
        self._kernel_cache = {}

    def _get_causal_hippo_kernel(self, L: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        Precomputes the exact L x L lower-triangular Causal Toeplitz matrix K_HiPPO
        from the LTI recurrence c_t = A c_{t-1} + B x_t, h_t = <readout, c_t>.
        - Strictly lower-triangular with positive diagonal k_0 > 0 -> Full Rank = L (100% lossless).
        - Row-normalized to unit L2 norm so input with Std=1.0 maps to output with Std=1.0 at every step t.
        """
        key = (L, device, dtype)
        if key in self._kernel_cache:
            return self._kernel_cache[key]
            
        A_f = self.A.to(device=device, dtype=torch.float32) # (N, N)
        B_f = self.B.to(device=device, dtype=torch.float32) # (N,)
        w_f = self.readout.squeeze(-1).to(device=device, dtype=torch.float32) # (N,)
        
        # Compute impulse response k_tau = w^T (A^tau B) for tau = 0 .. L-1
        impulse = torch.zeros(L, device=device, dtype=torch.float32)
        state = B_f.clone()
        for tau in range(L):
            impulse[tau] = torch.dot(w_f, state)
            state = torch.mv(A_f, state)
            
        # Construct L x L lower-triangular Toeplitz matrix: K[t, s] = impulse[t - s] for t >= s
        idx = torch.arange(L, device=device)
        diff = idx.unsqueeze(1) - idx.unsqueeze(0) # (L, L)
        causal_mask = diff >= 0
        K = torch.where(causal_mask, impulse[diff.clamp(min=0)], torch.zeros_like( diff, dtype=torch.float32))
        
        # Normalize each row t to unit L2 energy so Var(h_t) == Var(x_t) == 1.0 across all positions
        row_norm = K.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-6)
        K = (K / row_norm).to(dtype=dtype)
        
        self._kernel_cache[key] = K
        return K

    @torch.no_grad()
    def forward(self, x_seq: torch.Tensor, return_sequence: bool = True) -> torch.Tensor:
        """
        x_seq: (Batch, Length, D)
        return_sequence:
          - If True (default for Phase 4 1D-Unet): returns (Batch, Length, D) Full-Rank Causal HiPPO
            potential trajectory h_{0..L-1}, preserving 100% token resolution + N=64 Legendre memory.
          - If False (for 2D summary): returns (Batch, D) final pooled state.
        """
        B_size, L, D = x_seq.shape
        K = self._get_causal_hippo_kernel(L, x_seq.device, x_seq.dtype) # (L, L)
        
        # Parallel causal state rollout over all L positions: (L, L) @ (B, L, D) -> (B, L, D)
        h_seq = torch.matmul(K, x_seq)
        
        if return_sequence:
            return h_seq
        else:
            return h_seq[:, -1, :]
