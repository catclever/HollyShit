import mlx.core as mx
import mlx.nn as nn
from typing import List

class SenseAdapter(nn.Module):
    """
    A shallow non-linear layer that maps a specific heterogeneous external embedding
    into the common internal sensory space (d_model).
    """
    def __init__(self, input_dim: int, d_model: int):
        super().__init__()
        # NEW: Adding LayerNorm for absolute dimension suppression to prevent dominating features
        self.norm = nn.LayerNorm(input_dim)
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )
        
        self.scale = 1.0 / (input_dim ** 0.5)
        
    def __call__(self, x: mx.array) -> mx.array:
        # Normalize and scale
        x = self.norm(x) * self.scale
        return self.net(x)

class SensoryFuser(nn.Module):
    """
    Multi-View Fusion Adapter.
    Takes N different embeddings simultaneously (e.g. BGE, GTE, RoBERTa, Text2Vec),
    passes them through their specific SenseAdapters to match d_model,
    and fuses them (Stochastic Routing / Centroid Mean) into a single unified f_t.
    """
    def __init__(self, emb_dims: List[int], d_model: int):
        super().__init__()
        # Instantiate a separate SenseAdapter for each embedding type
        self.adapters = [SenseAdapter(dim, d_model) for dim in emb_dims]
        
    def __call__(self, embs: List[mx.array], weights: List[float] = None) -> mx.array:
        """
        Args:
            embs: A list of mx.arrays, each shaping (batch_size, [seq_len], dim_i)
            weights: Optional list of N floats. If provided, does weighted sum. If None, computes mean.
        Returns:
            f_fused: Universal representation (batch_size, [seq_len], d_model)
        """
        f_list = [adp(e) for adp, e in zip(self.adapters, embs)]
        f_stack = mx.stack(f_list, axis=-2)
        N = len(embs)
        
        if weights is not None:
            if len(weights) != N:
                raise ValueError(f"Expected {N} weights, got {len(weights)}")
            w_array = mx.array(weights).reshape(N, 1)
            f_fused = mx.sum(f_stack * w_array, axis=-2)
        else:
            # Centroid Mean Fusion
            f_fused = mx.mean(f_stack, axis=-2)
            
        return f_fused
