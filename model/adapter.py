from typing import List, Optional, Sequence, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np


class SenseAdapter(nn.Module):
    """
    A shallow non-linear layer that maps a specific heterogeneous external embedding
    into the common internal sensory space (d_model).
    """

    def __init__(self, input_dim: int, d_model: int):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.scale = 1.0 / (input_dim ** 0.5)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.norm(x) * self.scale
        return self.net(x)


class SensoryFuser(nn.Module):
    """
    Phase 0 v3 Multi-View Fusion Adapter (MLX).
    Preserves 100% parameter key compatibility with SensoryFuserCUDA:
      - adapters.{i}.norm.{weight,bias}
      - adapters.{i}.net.layers.{0,2}.{weight,bias}
    Supports both (N_views,) global weights and (B, N_views) per-sample Dirichlet simplex weights.
    """

    def __init__(self, emb_dims: List[int], d_model: int):
        super().__init__()
        self.emb_dims = [int(d) for d in emb_dims]
        self.d_model = int(d_model)
        self.adapters = [SenseAdapter(dim, d_model) for dim in self.emb_dims]

    @property
    def num_views(self) -> int:
        return len(self.adapters)

    def encode_views(self, embs: List[mx.array]) -> mx.array:
        if len(embs) != len(self.adapters):
            raise ValueError(f"Expected {len(self.adapters)} teacher embeddings, got {len(embs)}")
        f_list = [adp(e) for adp, e in zip(self.adapters, embs)]
        return mx.stack(f_list, axis=-2)

    def fuse_encoded_views(
        self,
        f_stack: mx.array,
        weights: Optional[Union[Sequence[float], mx.array]] = None,
    ) -> mx.array:
        n_views = f_stack.shape[-2]
        if weights is None:
            return mx.mean(f_stack, axis=-2)

        w_array = weights if isinstance(weights, mx.array) else mx.array(weights, dtype=f_stack.dtype)
        if w_array.ndim == 1:
            if w_array.shape[0] != n_views:
                raise ValueError(f"Expected {n_views} weights, got {w_array.shape[0]}")
            shape = [1] * f_stack.ndim
            shape[-2] = n_views
            w_array = w_array.reshape(*shape)
        elif w_array.ndim == 2:
            if w_array.shape[0] != f_stack.shape[0] or w_array.shape[1] != n_views:
                raise ValueError(
                    f"Expected per-sample weights of shape {(f_stack.shape[0], n_views)}, got {tuple(w_array.shape)}"
                )
            if f_stack.ndim == 3:
                w_array = w_array.reshape(f_stack.shape[0], n_views, 1)
            elif f_stack.ndim == 4:
                w_array = w_array.reshape(f_stack.shape[0], 1, n_views, 1)
            else:
                raise ValueError(f"Unsupported f_stack ndim={f_stack.ndim} for 2D weights")
        elif w_array.ndim == 3:
            w_array = mx.expand_dims(w_array, axis=-1)
        else:
            raise ValueError(f"Unsupported weights ndim={w_array.ndim}")

        return mx.sum(f_stack * w_array, axis=-2)

    def sample_dirichlet_weights(
        self,
        batch_size: int,
        alpha: Union[float, Sequence[float]] = 1.0,
        drop_prob: float = 0.15,
        min_active: int = 2,
        rng: Optional[np.random.Generator] = None,
    ) -> mx.array:
        n_views = len(self.adapters)
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        gen = rng if rng is not None else np.random.default_rng()

        if isinstance(alpha, (int, float)):
            if float(alpha) <= 0.0:
                raise ValueError(f"Dirichlet concentration alpha must be > 0, got {alpha}")
            alpha_vec = np.full(n_views, float(alpha), dtype=np.float64)
        else:
            alpha_vec = np.asarray(alpha, dtype=np.float64)
            if alpha_vec.shape != (n_views,):
                raise ValueError(f"Expected alpha of shape ({n_views},), got {alpha_vec.shape}")
            if np.any(alpha_vec <= 0.0):
                raise ValueError("All Dirichlet concentration values must be > 0")

        w = gen.dirichlet(alpha_vec, size=batch_size).astype(np.float32)
        eff_min_active = min(max(1, int(min_active)), n_views)
        if drop_prob > 0.0 and n_views > eff_min_active:
            if not (0.0 <= drop_prob < 1.0):
                raise ValueError(f"drop_prob must be in [0, 1), got {drop_prob}")
            keep_mask = gen.random((batch_size, n_views)) >= float(drop_prob)
            for i in range(batch_size):
                if keep_mask[i].sum() < eff_min_active:
                    chosen = gen.choice(n_views, size=eff_min_active, replace=False)
                    keep_mask[i, chosen] = True
            w = w * keep_mask.astype(np.float32)

        w = w / np.clip(w.sum(axis=-1, keepdims=True), 1e-8, None)
        return mx.array(w)

    def forward_cloud(
        self,
        embs: List[mx.array],
        alpha: Union[float, Sequence[float]] = 1.0,
        drop_prob: float = 0.15,
        min_active: int = 2,
        return_weights: bool = False,
        rng: Optional[np.random.Generator] = None,
    ) -> Union[mx.array, Tuple[mx.array, mx.array]]:
        f_stack = self.encode_views(embs)
        w = self.sample_dirichlet_weights(
            batch_size=f_stack.shape[0],
            alpha=alpha,
            drop_prob=drop_prob,
            min_active=min_active,
            rng=rng,
        )
        f_cloud = self.fuse_encoded_views(f_stack, weights=w)
        if return_weights:
            return f_cloud, w
        return f_cloud

    def __call__(
        self,
        embs: List[mx.array],
        weights: Optional[Union[Sequence[float], mx.array]] = None,
    ) -> mx.array:
        f_stack = self.encode_views(embs)
        return self.fuse_encoded_views(f_stack, weights=weights)
