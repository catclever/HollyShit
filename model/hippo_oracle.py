import math
from typing import Optional
import torch
import torch.nn as nn
import numpy as np
import scipy.linalg as linalg


def construct_hippo_legs(N: int):
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


def build_deterministic_ortho_mix(hidden_dim: int, seed: int = 42) -> torch.Tensor:
    """
    Builds a fixed deterministic orthogonal channel-mixing matrix of shape (hidden_dim, hidden_dim)
    using QR decomposition with positive diagonal sign convention (0 learnable parameters).
    """
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    rand_mat = torch.randn((hidden_dim, hidden_dim), generator=gen, dtype=torch.float32)
    q, r = torch.linalg.qr(rand_mat)
    diag_sign = torch.sign(torch.diagonal(r))
    diag_sign = torch.where(diag_sign == 0, torch.ones_like(diag_sign), diag_sign)
    ortho_mix = q * diag_sign.unsqueeze(0)
    return ortho_mix


class HiPPOOracleCUDA(nn.Module):
    """
    Frozen HiPPO-LegS Sentence Vector Bottleneck & State-Space Operator (0 trainable params).
    - Default mode (`return_sequence=False`): Projects a 3D token trajectory `x_seq` of shape `(B, L, D)`
      into a strictly 2D Sentence Vector `z_sent` of shape `(B, D)` by mixing all `D=1024` channels
      across `K=16` orthogonal Legendre temporal sub-bands (`d_sub = 64` dims per polynomial degree `k=0..15`).
    - Legacy sequence mode (`return_sequence=True`): Returns the 3D causal Toeplitz trajectory `(B, L, D)`.
    """

    def __init__(
        self,
        hidden_dim: int = 1024,
        N: int = 64,
        dt: float = 1.0,
        memory_horizon: float = 64.0,
        num_subbands: int = 16,
    ):
        super().__init__()
        if hidden_dim % num_subbands != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_subbands ({num_subbands})")

        self.hidden_dim = hidden_dim
        self.N = N
        self.num_subbands = num_subbands
        self.d_sub = hidden_dim // num_subbands

        A_cont, B_cont = construct_hippo_legs(N)

        # Bilinear Discretization (Tustin's method) scaled by characteristic memory horizon
        dt_scaled = dt / float(max(memory_horizon, 1.0))
        I = np.eye(N)
        BL = linalg.inv(I - (dt_scaled / 2.0) * A_cont)
        A_disc = BL @ (I + (dt_scaled / 2.0) * A_cont)
        B_disc = (BL @ B_cont * dt_scaled).squeeze(-1)

        # Register as fixed buffers (0 learnable parameters, strictly frozen Oracle)
        self.register_buffer("A", torch.tensor(A_disc, dtype=torch.float32))  # (N, N)
        self.register_buffer("B", torch.tensor(B_disc, dtype=torch.float32))  # (N,)

        # Analytical Legendre boundary readout vector aligned with B_disc (guarantees k_0 > 0)
        b_tensor = torch.tensor(B_disc, dtype=torch.float32)
        readout = b_tensor / b_tensor.norm().clamp(min=1e-6)
        self.register_buffer("readout", readout.unsqueeze(-1))  # (N, 1)

        # Legacy deterministic projection for 2D pooled summary mode
        torch.manual_seed(42)
        self.register_buffer("proj_out", torch.randn(N, 1) / np.sqrt(N))

        # Fixed deterministic orthogonal channel-mixing matrix (D, D) with seed=42
        ortho_mix = build_deterministic_ortho_mix(hidden_dim, seed=42)
        self.register_buffer("ortho_mix", ortho_mix)

        # Caches for basis and causal Toeplitz kernel
        self._kernel_cache = {}
        self._basis_cache = {}

    def _get_legendre_basis(self, L: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        Computes the first K = self.num_subbands (16) orthonormal shifted Legendre polynomials P_k(tau)
        evaluated at midpoint coordinates tau_l = (l + 0.5) / L in (0, 1) for l = 0..L-1,
        normalized so that sum_{l=0}^{L-1} basis[k, l]^2 == 1.0 (equivalently 1/L sum_l P_k(tau_l)^2 == 1.0).
        Returns `basis` of shape (K, L).
        """
        key = (L, device, dtype)
        if key in self._basis_cache:
            return self._basis_cache[key]

        K = self.num_subbands
        tau = (torch.arange(L, device=device, dtype=torch.float32) + 0.5) / float(max(L, 1))
        u = 2.0 * tau - 1.0  # Mapped to (-1, 1)

        polys = []
        for k in range(K):
            if k == 0:
                p_k = torch.ones_like(u)
            elif k == 1:
                p_k = u
            else:
                p_k = ((2.0 * k - 1.0) * u * polys[-1] - (k - 1.0) * polys[-2]) / float(k)
            polys.append(p_k)

        raw_basis = torch.stack(polys, dim=0)  # (K, L)

        # Discrete Gram-Schmidt orthogonalization + unit L2 row normalization (sum_l basis[k, l]^2 == 1.0)
        ortho_rows = []
        for k in range(K):
            v = raw_basis[k]
            if k < L:
                for prev in ortho_rows:
                    v = v - torch.dot(v, prev) * prev
            norm = v.norm(p=2).clamp(min=1e-6)
            ortho_rows.append(v / norm)

        basis = torch.stack(ortho_rows, dim=0).to(dtype=dtype)  # (K, L)
        self._basis_cache[key] = basis
        return basis

    @torch.no_grad()
    def encode_sentence_vector(
        self,
        x_seq: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Projects 3D sequence `x_seq` of shape `(B, L, D)` into a strictly 2D Sentence Vector
        `z_sent` of shape `(B, D)` (`(B, 1024)`):
          1. Mixes all D channels across K=16 sub-bands via `self.ortho_mix`;
          2. Projects along sequence dimension `L` onto the `K` orthonormal Legendre polynomials `basis` `(K, L)`;
          3. Reshapes `(B, K, d_sub)` -> `(B, D)` with unit variance (`Std ≈ 1.0`).
        """
        B_size, L, D = x_seq.shape
        K = self.num_subbands
        d_sub = self.d_sub

        ortho = self.ortho_mix.to(device=x_seq.device, dtype=x_seq.dtype)
        x_mixed = torch.matmul(x_seq, ortho).reshape(B_size, L, K, d_sub)

        basis = self._get_legendre_basis(L, x_seq.device, x_seq.dtype)  # (K, L)
        z_sub = torch.einsum("kl,blkd->bkd", basis, x_mixed)  # (B, K, d_sub)
        z_sent = z_sub.reshape(B_size, D)  # (B, D)
        return z_sent

    @torch.no_grad()
    def reconstruct_macro_wave(self, z_sent: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Reconstructs the rank-16 smooth Legendre guide wave of shape `(B, seq_len, D)`
        from a 2D Sentence Vector `z_sent` of shape `(B, D)` via the exact transpose operation.
        """
        B_size, D = z_sent.shape
        K = self.num_subbands
        d_sub = self.d_sub

        z_sub = z_sent.reshape(B_size, K, d_sub)
        basis = self._get_legendre_basis(seq_len, z_sent.device, z_sent.dtype)  # (K, seq_len)

        x_mixed_rec = torch.einsum("bkd,kl->blkd", z_sub, basis) * math.sqrt(K)
        x_mixed_rec = x_mixed_rec.reshape(B_size, seq_len, D)

        ortho_T = self.ortho_mix.to(device=z_sent.device, dtype=z_sent.dtype).T
        return torch.matmul(x_mixed_rec, ortho_T)

    def _get_causal_hippo_kernel(self, L: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        Precomputes the exact L x L lower-triangular Causal Toeplitz matrix K_HiPPO
        from the LTI recurrence c_t = A c_{t-1} + B x_t, h_t = <readout, c_t>.
        """
        key = (L, device, dtype)
        if key in self._kernel_cache:
            return self._kernel_cache[key]

        A_f = self.A.to(device=device, dtype=torch.float32)  # (N, N)
        B_f = self.B.to(device=device, dtype=torch.float32)  # (N,)
        w_f = self.readout.squeeze(-1).to(device=device, dtype=torch.float32)  # (N,)

        impulse = torch.zeros(L, device=device, dtype=torch.float32)
        state = B_f.clone()
        for tau in range(L):
            impulse[tau] = torch.dot(w_f, state)
            state = torch.mv(A_f, state)

        idx = torch.arange(L, device=device)
        diff = idx.unsqueeze(1) - idx.unsqueeze(0)  # (L, L)
        causal_mask = diff >= 0
        K = torch.where(causal_mask, impulse[diff.clamp(min=0)], torch.zeros_like(diff, dtype=torch.float32))

        row_norm = K.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-6)
        K = (K / row_norm).to(dtype=dtype)

        self._kernel_cache[key] = K
        return K

    @torch.no_grad()
    def forward(
        self,
        x_seq: torch.Tensor,
        return_sequence: bool = False,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x_seq: (Batch, Length, D)
        return_sequence:
          - If False (default for Phase 4 2D Sentence Vector Bottleneck): returns 2D Sentence Vector
            `z_sent` of shape `(Batch, D)` (`(B, 1024)`).
          - If True (legacy 3D mode): returns `(Batch, Length, D)` causal Toeplitz HiPPO trajectory.
        """
        if not return_sequence:
            return self.encode_sentence_vector(x_seq, valid_mask=valid_mask)

        B_size, L, D = x_seq.shape
        K = self._get_causal_hippo_kernel(L, x_seq.device, x_seq.dtype)  # (L, L)
        h_seq = torch.matmul(K, x_seq)
        return h_seq
