import torch
import torch.nn as nn
import math

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_dim, frequency_embedding_size=256):
        super().__init__()
        self.freq_dim = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.freq_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return self.mlp(emb)

class AdaLN(nn.Module):
    def __init__(self, hidden_dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.linear = nn.Linear(cond_dim, 2 * hidden_dim)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, h_potential: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D), h_potential: (B, cond_dim)
        x_norm = self.norm(x)
        shift, scale = self.linear(h_potential).chunk(2, dim=-1)
        return x_norm * (1.0 + scale[:, None, :]) + shift[:, None, :]

class OrthogonalAttractor(nn.Module):
    """
    Implements Knowledge Extraction via Dimensional Activation.
    Computes inner product (activation) between current trajectory and knowledge vectors.
    Applies orthogonal force along the activated dimensions.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.norm_x = nn.LayerNorm(hidden_dim)
        self.norm_k = nn.LayerNorm(hidden_dim)
        self.scale = 1.0 / math.sqrt(hidden_dim)

    def forward(self, x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L_x, D) - Unet's current generative trajectory
        k: (B, L_k, D) - Knowledge attractors (e.g. Prompt features)
        """
        # Normalize to stabilize the dot product activation
        x_norm = self.norm_x(x)
        k_norm = self.norm_k(k)
        
        # Activation Matrix: (B, L_x, L_k)
        # Measures how much the current curve 'x' activates the knowledge dimensions 'k'
        activation = torch.bmm(x_norm, k_norm.transpose(1, 2)) * self.scale
        
        # We apply ReLU to ensure we only get pulled by positive dimensional matches
        # This acts as the "Snap-to-Grid" threshold
        activation = torch.relu(activation)
        
        # Apply Force: (B, L_x, D)
        # The force vector aligns perfectly with the knowledge dimension 'k'
        force = torch.bmm(activation, k)
        return force

class Gated1DResBlock(nn.Module):
    def __init__(self, hidden_dim: int, cond_dim: int, kernel_size: int = 5, dilation: int = 1):
        super().__init__()
        
        self.time_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.adaLN1 = AdaLN(hidden_dim, cond_dim)
        self.attractor = OrthogonalAttractor(hidden_dim)
        
        padding = ((kernel_size - 1) * dilation) // 2
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding, dilation=dilation)
        self.act1 = nn.SiLU()
        
        self.adaLN2 = AdaLN(hidden_dim, cond_dim)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding, dilation=dilation)
        
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, h_potential: torch.Tensor, k_attractors: torch.Tensor) -> torch.Tensor:
        residual = x
        
        # 1. Inject Time
        t_spatial = self.time_proj(t_emb).unsqueeze(1)
        x = x + t_spatial
        
        # 2. Inject Momentum (Macroscopic Wind via AdaLN)
        x = self.adaLN1(x, h_potential)
        
        # 3. Inject Knowledge Attractor (Orthogonal Gravity)
        if k_attractors is not None:
            force = self.attractor(x, k_attractors)
            x = x + force
        
        # 4. Sculpt Geometry (1D Conv)
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = x.transpose(1, 2)
        x = self.act1(x)
        
        x = self.adaLN2(x, h_potential)
        x = x.transpose(1, 2)
        x = self.conv2(x)
        x = x.transpose(1, 2)
        
        return residual + x

class FlowMatching1DUnet(nn.Module):
    """
    Vector Field Predictor for OT-CFM.
    Dual-Brain Architecture: Driven by h_delta (Mamba) and orthogonal knowledge attractors.
    """
    def __init__(self, hidden_dim: int = 1024, cond_dim: int = 1024, num_layers: int = 6):
        super().__init__()
        self.t_embedder = TimestepEmbedder(hidden_dim)
        
        self.blocks = nn.ModuleList([
            Gated1DResBlock(
                hidden_dim=hidden_dim, 
                cond_dim=cond_dim, 
                kernel_size=5, 
                dilation=2**(i % 4)
            ) for i in range(num_layers)
        ])
        
        self.final_norm = AdaLN(hidden_dim, cond_dim)
        self.final_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, h_potential: torch.Tensor, k_attractors: torch.Tensor = None) -> torch.Tensor:
        """
        x_t: (B, L, D) - Current state of the trajectory
        t: (B,) - Timesteps
        h_potential: (B, D) - Legendre polynomial coefficients (from HiPPO or Mamba)
        k_attractors: (B, L_k, D) - Knowledge stream coordinates (e.g. Prompt Phase 1 output)
        """
        t_emb = self.t_embedder(t)
        
        x = x_t
        for block in self.blocks:
            x = block(x, t_emb, h_potential, k_attractors)
            
        x = self.final_norm(x, h_potential)
        v_pred = self.final_proj(x)
        
        return v_pred
