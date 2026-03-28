import mlx.core as mx
import mlx.nn as nn
from .mamba_mlx.mamba_mlx import Mamba, MambaConfig

class MambaPlanner(nn.Module):
    """
    Core Macro planner engine. Takes the trajectory of semantic coordinates,
    runs them through Mamba to understand context and momentum, and outputs 
    the parameters for the predicted future coordinate region (μ, σ) 
    and the length (number of tokens) for the next segment.
    """
    def __init__(self, config: MambaConfig, z_dim: int, residual_mode: bool = False):
        super().__init__()
        self.mamba = Mamba(config)
        self.d_model = config.d_model
        
        # Dual Probability Heads
        self.mu_head = nn.Linear(config.d_model, z_dim)
        self.logvar_head = nn.Linear(config.d_model, z_dim)
        
        self.residual_mode = residual_mode
        
    def __call__(self, x: mx.array):
        """
        Forward pass for time-shifted training/trajectory planning.
        x: (B, L, d_model) - History trajectory up to t-1
        
        Returns:
            mu_net: (B, L, z_dim) - Predicted spatial coordinate centers
            logvar_net: (B, L, z_dim) - Predicted uncertainty variances
            h_t: (B, L, d_model) - Deterministic dynamic hidden state (optional for inspection)
        """
        # 1. State Space Sequence processing
        h_t = self.mamba(x)
        
        # 2. Probability Mapping
        mu = self.mu_head(h_t)
        logvar = self.logvar_head(h_t)
        
        # 3. Residual Vector Field Integration (if enabled)
        if self.residual_mode:
            # The network output is treated as Delta Mu (velocity)
            # We integrate over time using cumsum to get the absolute coordinates
            mu = mx.cumsum(mu, axis=1)
            
        return mu, logvar, h_t
