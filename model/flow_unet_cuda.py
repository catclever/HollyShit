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
        # Scale t in [0, 1] by 1000.0 so sinusoidal frequencies span full periods
        t_scaled = t.float() * 1000.0
        emb = t_scaled[:, None] * emb[None, :]
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
        # x: (B, L, D), h_potential: (B, L, cond_dim) or (B, cond_dim)
        x_norm = self.norm(x)
        shift, scale = self.linear(h_potential).chunk(2, dim=-1)
        if shift.dim() == 2:
            shift = shift[:, None, :]
            scale = scale[:, None, :]
        return x_norm * (1.0 + scale) + shift

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
    Dual-Brain Architecture: Driven by Legendre macroscopic potential curve (HiPPO/Mamba),
    1D sinusoidal positional coordinates, and orthogonal knowledge attractors.
    Includes built-in Adaptive Manifold Auto-Calibration (manifold_mean & manifold_std).
    """
    def __init__(self, hidden_dim: int = 1024, cond_dim: int = 1024, num_layers: int = 6):
        super().__init__()
        self.hidden_dim = hidden_dim
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

        # In-Model Adaptive Manifold Auto-Calibration Buffers (persisted in unet.pt)
        self.register_buffer("manifold_mean", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("manifold_std", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("calib_steps", torch.tensor(0, dtype=torch.long))

        self._pos_cache = {}

    def _get_pos_emb(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        Generates 1D Sinusoidal Positional Embeddings of shape (1, seq_len, hidden_dim).
        Breaks translation symmetry so every token position l in [0, L-1] knows its exact order.
        """
        key = (seq_len, device, dtype)
        if key in self._pos_cache:
            return self._pos_cache[key]

        half_dim = self.hidden_dim // 2
        pos = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1) # (L, 1)
        div_term = torch.exp(
            torch.arange(half_dim, dtype=torch.float32, device=device) * (-math.log(10000.0) / max(half_dim - 1, 1))
        ).unsqueeze(0) # (1, D/2)
        angles = pos * div_term
        pos_emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1).unsqueeze(0).to(dtype=dtype) # (1, L, D)
        self._pos_cache[key] = pos_emb
        return pos_emb

    @torch.no_grad()
    def calibrate_manifold(self, x_sample: torch.Tensor, max_calib_steps: int = 50, momentum: float = 0.05) -> bool:
        """
        Online EMA Auto-Calibration for target manifold scale:
        - Step 0: Directly initializes manifold_mean and manifold_std from the first batch.
        - Steps 1 .. max_calib_steps-1: Smooths estimate via EMA.
        - Step >= max_calib_steps: Locks coordinate scale permanently for stable flow training.
        """
        step = int(self.calib_steps.item())
        if step >= max_calib_steps:
            return False

        batch_mean = x_sample.detach().float().mean()
        batch_std = x_sample.detach().float().std().clamp(min=1e-5)

        if step == 0:
            self.manifold_mean.copy_(batch_mean)
            self.manifold_std.copy_(batch_std)
        else:
            self.manifold_mean.lerp_(batch_mean, momentum)
            self.manifold_std.lerp_(batch_std, momentum)

        self.calib_steps.add_(1)
        return True

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Maps raw encoder trajectory (Mean=mu, Std=sigma) -> Normalized Flow Manifold (Mean=0, Std=1)."""
        mean = self.manifold_mean.to(device=x.device, dtype=x.dtype)
        std = self.manifold_std.to(device=x.device, dtype=x.dtype).clamp(min=1e-5)
        return (x - mean) / std

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Restores generated normalized trajectory (Mean=0, Std=1) -> Original Encoder Manifold (Mean=mu, Std=sigma)."""
        mean = self.manifold_mean.to(device=x.device, dtype=x.dtype)
        std = self.manifold_std.to(device=x.device, dtype=x.dtype).clamp(min=1e-5)
        return x * std + mean

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, h_potential: torch.Tensor, k_attractors: torch.Tensor = None) -> torch.Tensor:
        """
        x_t: (B, L, D) - Current state of the normalized trajectory
        t: (B,) - Timesteps in [0, 1]
        h_potential: (B, L, D) or (B, D) - Legendre polynomial potential curve (from HiPPO or Mamba)
        k_attractors: (B, L_k, D) - Knowledge stream coordinates (e.g. Prompt Phase 1 output)
        """
        B, L, D = x_t.shape
        t_emb = self.t_embedder(t)
        
        pos_x = self._get_pos_emb(L, x_t.device, x_t.dtype)
        x = x_t + pos_x

        k_pos = None
        if k_attractors is not None:
            pos_k = self._get_pos_emb(k_attractors.shape[1], k_attractors.device, k_attractors.dtype)
            k_pos = k_attractors + pos_k
        
        for block in self.blocks:
            x = block(x, t_emb, h_potential, k_pos)
            
        x = self.final_norm(x, h_potential)
        v_pred = self.final_proj(x)
        
        return v_pred
