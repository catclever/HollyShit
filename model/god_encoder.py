import mlx.core as mx
import mlx.nn as nn

class GodEncoder(nn.Module):
    """
    The True Deterministic Encoder (Phase 1 V2).
    It acts as the 'God' generating the absolute ground-truth target points 
    (z_target) in the endogenous space, without probability or noise.
    
    This target determines the structural layout for the decoder.
    """
    def __init__(self, input_dim: int, hidden_dim: int, z_dim: int):
        super().__init__()
        self.z_dim = z_dim
        # A simple powerful multi-layer mapping to forge the structural space
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, z_dim)
        )
        
    def __call__(self, x_t: mx.array):
        """
        x_t: (B, L, Input_Dim) - The external textual features (i.e. 'Senses')
        
        Returns:
            z_target: (B, L, Z_Dim) - The deterministic spatial coordinates
        """
        z_target = self.encoder(x_t)
        return z_target
