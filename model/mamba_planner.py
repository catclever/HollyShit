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
    def __init__(self, config: MambaConfig, z_dim: int):
        super().__init__()
        self.mamba = Mamba(config)
        self.d_model = config.d_model
        
    def __call__(self, x: mx.array):
        """
        Forward pass for time-shifted training/trajectory planning.
        x: (B, L, d_model) - History trajectory up to t-1
        
        Returns:
            h_t: (B, L, d_model) - Deterministic dynamic hidden state
        """
        mamba_out = self.mamba(x)
        return mamba_out
