import torch
import torch.nn as nn
import numpy as np
import scipy.linalg as linalg

class HiPPOContextCompressor(nn.Module):
    """
    Fixed parameter-free HiPPO continuous temporal projector.
    Projects (B, L, D) continuous manifold into (B, D) macroscopic potential energy state.
    
    This replaces Mamba as a deterministic, mathematically optimal LTI operator 
    for context compression without needing target sequences during inference.
    """
    def __init__(self, d_model=1024, order=64, dt=1.0):
        super().__init__()
        self.d_model = d_model
        self.order = order
        
        # 1. Analytical HiPPO-LegS matrix
        A_cont = np.zeros((order, order))
        B_cont = np.zeros((order, 1))
        for i in range(order):
            for j in range(order):
                if i > j:
                    A_cont[i, j] = (2 * i + 1) ** 0.5 * (2 * j + 1) ** 0.5
                elif i == j:
                    A_cont[i, j] = i + 1
            B_cont[i, 0] = (2 * i + 1) ** 0.5
        A_cont = -A_cont
        
        # 2. Bilinear Discretization (Tustin's method)
        I = np.eye(order)
        BL = linalg.inv(I - (dt / 2.0) * A_cont)
        A_disc = BL @ (I + (dt / 2.0) * A_cont)
        B_disc = (BL @ B_cont * dt).squeeze(-1)
        
        # Register as fixed buffers (no gradients, zero learnable parameters in the state transition)
        self.register_buffer("A", torch.tensor(A_disc, dtype=torch.float32)) # (N, N)
        self.register_buffer("B", torch.tensor(B_disc, dtype=torch.float32)) # (N,)
        
        # 3. Projection head: Compress N polynomial coefficients back to 1 condition scale
        # This is the only learnable part: mapping the polynomial state back to a wind vector
        self.proj = nn.Linear(order, 1, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, return_all_states=False) -> torch.Tensor:
        """
        x: (B, L, D) Continuous sequence
        return_all_states: if True, returns sequence of states (B, L, D) instead of just the final state
        
        Returns: h_cond: (B, D) or (B, L, D) Macroscopic state condition
        """
        B, L, D = x.shape
        # State coefficients c: (B, D, N)
        c = torch.zeros(B, D, self.order, device=x.device, dtype=x.dtype)
        
        if return_all_states:
            all_h = []
            for t in range(L):
                xt = x[:, t, :] # (B, D)
                # c_{t} = c_{t-1} @ A.T + x_t ⊗ B
                c = torch.matmul(c, self.A.T) + xt.unsqueeze(-1) * self.B
                ht = self.proj(c).squeeze(-1)
                all_h.append(self.norm(ht))
            return torch.stack(all_h, dim=1)
        else:
            # Sequential rollout (O(L) time, very fast in PyTorch for small L)
            for t in range(L):
                xt = x[:, t, :] # (B, D)
                c = torch.matmul(c, self.A.T) + xt.unsqueeze(-1) * self.B
                
            # Compress polynomial dimension: (B, D, N) -> (B, D)
            h_cond = self.proj(c).squeeze(-1)
            return self.norm(h_cond)
