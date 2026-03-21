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
        self.net = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )
        
    def __call__(self, x: mx.array) -> mx.array:
        return self.net(x)

class MultiSenseAdapter(nn.Module):
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
        
    def __call__(self, embs: List[mx.array], training: bool = True) -> mx.array:
        """
        Args:
            embs: A list of mx.arrays, each shaping (batch_size, [seq_len], dim_i)
            training: If True, randomly route 1 embedding per batch to prevent co-adaptation.
                      If False, compute the Centroid Mean.
        Returns:
            f_fused: Universal representation (batch_size, [seq_len], d_model)
        """
        # 1. Gather all individual f_t's from each SenseAdapter
        f_list = [adp(e) for adp, e in zip(self.adapters, embs)]
        
        # Stack to shape (B, N, d_model) or (B, L, N, d_model) depending on input
        f_stack = mx.stack(f_list, axis=-2)
        N = len(embs)
        
        if training:
            # Stochastic Drop-Emb (Random Routing)
            idx_array = mx.random.randint(0, N, shape=(1,))
            idx = idx_array.item()
            f_fused = f_list[idx]
        else:
            # Centroid Mean Fusion
            f_fused = mx.mean(f_stack, axis=-2)
            
        return f_fused

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
