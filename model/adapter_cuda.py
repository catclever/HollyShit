from typing import List, Optional, Sequence

import torch
import torch.nn as nn


class _TwoLayerMlpCUDA(nn.Module):
    """Keeps MLX-like key layout: net.layers.{0,2}.*"""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Linear(input_dim, output_dim),
                nn.SiLU(),
                nn.Linear(output_dim, output_dim),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers[0](x)
        x = self.layers[1](x)
        x = self.layers[2](x)
        return x


class SenseAdapterCUDA(nn.Module):
    def __init__(self, input_dim: int, d_model: int):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.net = _TwoLayerMlpCUDA(input_dim, d_model)
        self.scale = 1.0 / (input_dim ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x) * self.scale
        return self.net(x)


class SensoryFuserCUDA(nn.Module):
    def __init__(self, emb_dims: Sequence[int], d_model: int):
        super().__init__()
        self.adapters = nn.ModuleList([SenseAdapterCUDA(dim, d_model) for dim in emb_dims])

    def forward(self, embs: List[torch.Tensor], weights: Optional[Sequence[float]] = None) -> torch.Tensor:
        f_list = [adapter(emb) for adapter, emb in zip(self.adapters, embs)]
        f_stack = torch.stack(f_list, dim=-2)
        n_views = len(embs)

        if weights is not None:
            if len(weights) != n_views:
                raise ValueError(f"Expected {n_views} weights, got {len(weights)}")
            w = torch.as_tensor(weights, dtype=f_stack.dtype, device=f_stack.device)
            shape = [1] * f_stack.ndim
            shape[-2] = n_views
            w = w.view(*shape)
            return torch.sum(f_stack * w, dim=-2)
        return torch.mean(f_stack, dim=-2)
