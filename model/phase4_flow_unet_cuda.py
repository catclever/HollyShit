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
        # t: (B,)
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

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D), c: (B, cond_dim)
        x_norm = self.norm(x)
        shift, scale = self.linear(c).chunk(2, dim=-1)
        return x_norm * (1.0 + scale[:, None, :]) + shift[:, None, :]

class Gated1DResBlock(nn.Module):
    def __init__(self, hidden_dim: int, cond_dim: int, kernel_size: int = 5, dilation: int = 1):
        super().__init__()
        
        # We explicitly inject time into the spatial sequence
        self.time_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.adaLN1 = AdaLN(hidden_dim, cond_dim)
        padding = ((kernel_size - 1) * dilation) // 2
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding, dilation=dilation)
        self.act1 = nn.SiLU()
        
        self.adaLN2 = AdaLN(hidden_dim, cond_dim)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding, dilation=dilation)
        
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, z_macro: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        residual = x
        
        # Inject time directly into x by adding it broadcasted over L
        t_spatial = self.time_proj(t_emb).unsqueeze(1) # (B, 1, D)
        x = x + t_spatial
        
        # AdaLN uses ONLY the macroscopic state (z_macro)
        x = self.adaLN1(x, z_macro)
        
        # Conv1d expects (B, D, L)
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = x.transpose(1, 2)
        
        x = self.act1(x)
        
        x = self.adaLN2(x, z_macro)
        
        x = x.transpose(1, 2)
        x = self.conv2(x)
        x = x.transpose(1, 2)
        
        return residual + x

class FlowMatching1DUnet(nn.Module):
    """
    Vector Field Predictor for OT-CFM.
    Replaces the heavy $O(L^2)$ DiT with a lightweight 1D-ResNet structure.
    """
    def __init__(self, hidden_dim: int = 1024, cond_dim: int = 1024, num_layers: int = 6):
        super().__init__()
        self.t_embedder = TimestepEmbedder(hidden_dim)
        
        # We use a stack of blocks with exponentially growing dilations to capture long-range dependencies
        self.blocks = nn.ModuleList([
            Gated1DResBlock(
                hidden_dim=hidden_dim, 
                cond_dim=cond_dim, 
                kernel_size=5, 
                dilation=2**(i % 4) # Cycle dilations to get large receptive fields
            ) for i in range(num_layers)
        ])
        
        self.final_norm = AdaLN(hidden_dim, cond_dim)
        self.final_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, z_macro: torch.Tensor) -> torch.Tensor:
        """
        x_t: (B, L, D) - Interpolated state
        t: (B,) - Timesteps
        z_macro: (B, D) - Mamba macroscopic semantic coordinate
        """
        t_emb = self.t_embedder(t)
        
        x = x_t
        for block in self.blocks:
            x = block(x, t_emb, z_macro)
            
        x = self.final_norm(x, z_macro)
        v_pred = self.final_proj(x)
        
        return v_pred

def ot_cfm_loss(unet, mamba, x_in, x_tgt):
    """
    Computes Optimal Transport Conditional Flow Matching (OT-CFM) Loss in PyTorch.
    x_in: the input feature for Mamba to derive z_macro
    x_tgt: the target $L \times D$ concept tensor
    """
    B, L, D = x_tgt.shape
    device = x_tgt.device
    
    # Extract z_macro from the unperturbed input
    z_macro = mamba(x_in) # (B, D)
    
    # Sample t ~ U(0, 1)
    t = torch.rand((B,), device=device)
    t_expand = t[:, None, None] # (B, 1, 1)
    
    # Sample base noise x_0 ~ N(0, I)
    x_0 = torch.randn_like(x_tgt)
    
    # OT Interpolation (Straight line)
    x_t = t_expand * x_tgt + (1 - t_expand) * x_0
    
    # Target Velocity Field
    v_target = x_tgt - x_0
    
    # Predict Velocity Field
    v_pred = unet(x_t, t, z_macro)
    
    # MSE Loss
    loss = torch.mean((v_pred - v_target)**2)
    return loss
