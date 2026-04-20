import torch
import torch.nn as nn


class _TwoLayerMlpCUDA(nn.Module):
    """Keeps MLX-like key layout: net.layers.{0,2}.*"""

    def __init__(self, d_model: int, z_dim: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Linear(d_model, d_model),
                nn.SiLU(),
                nn.Linear(d_model, z_dim),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers[0](x)
        x = self.layers[1](x)
        x = self.layers[2](x)
        return x


class GodEncoderCUDA(nn.Module):
    def __init__(self, d_model: int, z_dim: int):
        super().__init__()
        self.net = _TwoLayerMlpCUDA(d_model, z_dim)

    def forward(self, f_t: torch.Tensor) -> torch.Tensor:
        return self.net(f_t)
