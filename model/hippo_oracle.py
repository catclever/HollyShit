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
    A frozen, mathematically exact HiPPO projector.
    It integrates a sequence of vectors and projects them onto the Legendre polynomial basis.
    This tracks all D dimensions independently (state shape: B, D, N) to prevent information loss.
    """
    def __init__(self, hidden_dim: int, N: int = 64, dt: float = 1.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.N = N
        
        A_cont, B_cont = construct_hippo_legs(N)
        
        # Bilinear Discretization (Tustin's method) for mathematically optimal and stable discrete integration
        I = np.eye(N)
        BL = linalg.inv(I - (dt / 2.0) * A_cont)
        A_disc = BL @ (I + (dt / 2.0) * A_cont)
        B_disc = (BL @ B_cont * dt).squeeze(-1)
        
        # Register as buffers so they are moved to device but not trained
        self.register_buffer("A", torch.tensor(A_disc, dtype=torch.float32)) # (N, N)
        self.register_buffer("B", torch.tensor(B_disc, dtype=torch.float32)) # (N,)
        
        # We project the (D, N) state back to (D) so it fits into the Unet AdaLN condition.
        # This is a fixed deterministic projection since this is an Oracle.
        torch.manual_seed(42)
        self.register_buffer("proj_out", torch.randn(N, 1) / np.sqrt(N))

    @torch.no_grad()
    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        x_seq: (Batch, Length, D)
        Returns: (Batch, D) - The polynomial potential energy vector for the full manifold
        """
        B_size, L, D = x_seq.shape
        
        # State: (Batch, D, N)
        c = torch.zeros(B_size, D, self.N, device=x_seq.device, dtype=x_seq.dtype)
        
        # O(L) Recurrent integration over time. Tracks all D dimensions simultaneously!
        for t in range(L):
            x_t = x_seq[:, t, :] # (Batch, D)
            # c_{t} = c_{t-1} @ A.T + x_t ⊗ B
            c = torch.matmul(c, self.A.T) + x_t.unsqueeze(-1) * self.B
            
        # Project Legendre state back to D dimensions to match Unet conditions
        # c is (Batch, D, N), proj_out is (N, 1) -> result is (Batch, D, 1) -> (Batch, D)
        h_potential = torch.matmul(c, self.proj_out).squeeze(-1)
        
        return h_potential
