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
    A frozen, mathematically exact HiPPO-LegS Hilbert space projector.
    Projects a sequence of vectors (B, L, D) onto N orthonormal Legendre polynomial
    basis functions over the full interval [-1, 1] without exponential forgetting,
    and reconstructs the macroscopic polynomial potential curve (B, L, D).
    """
    def __init__(self, hidden_dim: int, N: int = 64, dt: float = 1.0, max_seq_len: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.N = N
        
        A_cont, B_cont = construct_hippo_legs(N)
        
        # Scale dt by 1 / max_seq_len so the continuous horizon [0, 1] spans the full sequence
        dt_scaled = dt / float(max(max_seq_len, 1))
        I = np.eye(N)
        BL = linalg.inv(I - (dt_scaled / 2.0) * A_cont)
        A_disc = BL @ (I + (dt_scaled / 2.0) * A_cont)
        B_disc = (BL @ B_cont * dt_scaled).squeeze(-1)
        
        # Register as buffers so they are moved to device but not trained
        self.register_buffer("A", torch.tensor(A_disc, dtype=torch.float32)) # (N, N)
        self.register_buffer("B", torch.tensor(B_disc, dtype=torch.float32)) # (N,)
        
        # Deterministic summary projection for 2D pooled mode (e.g. Phase 5 Mamba summary)
        torch.manual_seed(42)
        self.register_buffer("proj_out", torch.randn(N, 1) / np.sqrt(N))
        
        # Cache for orthonormal Legendre basis matrix P of shape (L, N)
        self._basis_cache = {}

    def _get_legendre_basis(self, L: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        Constructs the exact orthonormal discrete Legendre polynomial basis P in R^{L x N}
        evaluated on [-1, 1] such that P^T P = I_N.
        """
        key = (L, device, dtype)
        if key in self._basis_cache:
            return self._basis_cache[key]
            
        n_orders = min(self.N, L)
        tau = torch.linspace(-1.0, 1.0, steps=L, device=device, dtype=torch.float32)
        V = torch.zeros((L, n_orders), device=device, dtype=torch.float32)
        V[:, 0] = 1.0
        if n_orders > 1:
            V[:, 1] = tau
        for n in range(1, n_orders - 1):
            V[:, n + 1] = ((2 * n + 1) * tau * V[:, n] - n * V[:, n - 1]) / float(n + 1)
            
        # QR orthonormalization with positive diagonal sign convention (Gram-Schmidt on Legendre basis)
        Q, R = torch.linalg.qr(V, mode="reduced")
        diag_sign = torch.sign(torch.diagonal(R))
        diag_sign = torch.where(diag_sign == 0, torch.ones_like(diag_sign), diag_sign)
        Q = Q * diag_sign.unsqueeze(0)
        
        if n_orders < self.N:
            pad = torch.zeros((L, self.N - n_orders), device=device, dtype=torch.float32)
            Q = torch.cat([Q, pad], dim=-1)
            
        basis = Q.to(dtype=dtype)
        self._basis_cache[key] = basis
        return basis

    @torch.no_grad()
    def encode_coeffs(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        Projects sequence x_seq: (B, L, D) -> Legendre coefficients c: (B, D, N)
        """
        B_size, L, D = x_seq.shape
        basis = self._get_legendre_basis(L, x_seq.device, x_seq.dtype) # (L, N)
        # (B, D, L) @ (L, N) -> (B, D, N)
        c = torch.matmul(x_seq.transpose(1, 2), basis)
        return c

    @torch.no_grad()
    def reconstruct_curve(self, c: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Synthesizes macroscopic trajectory (B, L, D) from Legendre coefficients c: (B, D, N)
        """
        basis = self._get_legendre_basis(seq_len, c.device, c.dtype) # (L, N)
        # (B, D, N) @ (N, L) -> (B, D, L) -> (B, L, D)
        return torch.matmul(c, basis.transpose(0, 1)).transpose(1, 2)

    @torch.no_grad()
    def forward(self, x_seq: torch.Tensor, return_sequence: bool = True) -> torch.Tensor:
        """
        x_seq: (Batch, Length, D)
        return_sequence:
          - If True (default for Phase 4 1D-Unet): returns (Batch, Length, D) Legendre polynomial curve
            providing localized macroscopic potential energy at every position l in [0, L-1].
          - If False (for 2D summary): returns (Batch, D) pooled potential vector.
        """
        B_size, L, D = x_seq.shape
        c = self.encode_coeffs(x_seq) # (B, D, N)
        
        if return_sequence:
            return self.reconstruct_curve(c, L) # (B, L, D)
        else:
            proj = self.proj_out.to(device=c.device, dtype=c.dtype)
            return torch.matmul(c, proj).squeeze(-1) # (B, D)
