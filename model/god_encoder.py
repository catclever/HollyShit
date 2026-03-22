import mlx.core as mx
import mlx.nn as nn

class GodEncoder(nn.Module):
    """
    The Single Unique God Encoder.
    Maps the unified fused sensory stream (f_t in d_model) into the 
    absolute truth anchor space (z_target in z_dim).
    """
    def __init__(self, d_model: int, z_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, z_dim)
        )
        
    def __call__(self, f_t: mx.array) -> mx.array:
        return self.net(f_t)
