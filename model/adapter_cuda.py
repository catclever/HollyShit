from typing import List, Optional, Sequence, Tuple, Union

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
        if not torch.is_autocast_enabled(x.device.type) and x.dtype != self.norm.weight.dtype:
            x = x.to(dtype=self.norm.weight.dtype)
        # Workaround for PyTorch 2.10 Apple MPS `native_layer_norm_backward` bug:
        # when `input.requires_grad == False`, MPS leaves `LayerNorm.{weight,bias}.grad`
        # populated with uninitialized Metal memory (`inf` / garbage).
        if (
            x.device.type == "mps"
            and torch.is_grad_enabled()
            and not x.requires_grad
            and self.norm.weight.requires_grad
        ):
            x = x.detach().requires_grad_(True)
        x = self.norm(x) * self.scale
        return self.net(x)


class SensoryFuserCUDA(nn.Module):
    """
    Phase 0 v3 Multi-View Sensory Fuser (PyTorch CUDA/MPS/CPU).
    Preserves 100% state_dict key compatibility with legacy MLX/PyTorch checkpoints:
      - adapters.{i}.norm.{weight,bias}
      - adapters.{i}.net.layers.{0,2}.{weight,bias}
    Adds Dirichlet simplex cloud sampling with stochastic modality dropout and per-sample
    (B, N_views) weight fusion.
    """

    def __init__(self, emb_dims: Sequence[int], d_model: int):
        super().__init__()
        self.emb_dims = [int(d) for d in emb_dims]
        self.d_model = int(d_model)
        self.adapters = nn.ModuleList([SenseAdapterCUDA(dim, d_model) for dim in self.emb_dims])

    @property
    def num_views(self) -> int:
        return len(self.adapters)

    def encode_views(self, embs: Sequence[torch.Tensor]) -> torch.Tensor:
        """
        Projects each teacher embedding into the shared d_model space without fusing.
        Returns:
            f_stack: Tensor of shape (B, [L], N_views, d_model)
        """
        if len(embs) != len(self.adapters):
            raise ValueError(f"Expected {len(self.adapters)} teacher embeddings, got {len(embs)}")
        f_list = [adapter(emb) for adapter, emb in zip(self.adapters, embs)]
        return torch.stack(f_list, dim=-2)

    def fuse_encoded_views(
        self,
        f_stack: torch.Tensor,
        weights: Optional[Union[Sequence[float], torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Fuses pre-encoded teacher views f_stack of shape (B, [L], N_views, d_model)
        using either:
          - None (uniform centroid mean across N_views),
          - 1D weights of shape (N_views,),
          - 2D per-sample weights of shape (B, N_views),
          - 3D per-token weights of shape (B, L, N_views).
        """
        n_views = f_stack.shape[-2]
        if weights is None:
            return torch.mean(f_stack, dim=-2)

        w = torch.as_tensor(weights, dtype=f_stack.dtype, device=f_stack.device)
        if w.ndim == 1:
            if w.shape[0] != n_views:
                raise ValueError(f"Expected {n_views} weights, got {w.shape[0]}")
            shape = [1] * f_stack.ndim
            shape[-2] = n_views
            w = w.view(*shape)
        elif w.ndim == 2:
            if w.shape[0] != f_stack.shape[0] or w.shape[1] != n_views:
                raise ValueError(
                    f"Expected per-sample weights of shape {(f_stack.shape[0], n_views)}, got {tuple(w.shape)}"
                )
            if f_stack.ndim == 3:
                w = w.unsqueeze(-1)  # (B, N_views, 1)
            elif f_stack.ndim == 4:
                w = w.unsqueeze(1).unsqueeze(-1)  # (B, 1, N_views, 1)
            else:
                raise ValueError(f"Unsupported f_stack ndim={f_stack.ndim} for 2D weights")
        elif w.ndim == 3:
            if f_stack.ndim != 4 or w.shape != f_stack.shape[:3]:
                raise ValueError(
                    f"Expected 3D weights of shape {tuple(f_stack.shape[:3])}, got {tuple(w.shape)}"
                )
            w = w.unsqueeze(-1)
        else:
            raise ValueError(f"Unsupported weights ndim={w.ndim}")

        return torch.sum(f_stack * w, dim=-2)

    def sample_dirichlet_weights(
        self,
        batch_size: int,
        alpha: Union[float, Sequence[float], torch.Tensor] = 1.0,
        drop_prob: float = 0.15,
        min_active: int = 2,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Samples per-sample Dirichlet simplex weights of shape (B, N_views) with optional
        stochastic view dropout keeping at least `min_active` teachers active per sample.
        Computes simplex normalization in float32 before casting to `dtype` so `1e-8` never
        underflows to 0/0 NaN under FP16.
        """
        n_views = len(self.adapters)
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if device is None:
            device = next(self.parameters()).device

        # Sample Gamma on CPU for universal compatibility (including Apple MPS where
        # _standard_gamma may not be implemented for all dtypes), then transfer to device.
        if isinstance(alpha, (int, float)):
            if float(alpha) <= 0.0:
                raise ValueError(f"Dirichlet concentration alpha must be > 0, got {alpha}")
            conc = torch.full((batch_size, n_views), float(alpha), dtype=torch.float32, device="cpu")
        else:
            alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device="cpu")
            if alpha_t.ndim == 1:
                if alpha_t.shape[0] != n_views:
                    raise ValueError(f"Expected {n_views} alpha values, got {alpha_t.shape[0]}")
                conc = alpha_t.unsqueeze(0).expand(batch_size, n_views).contiguous()
            elif alpha_t.ndim == 2 and alpha_t.shape == (batch_size, n_views):
                conc = alpha_t.contiguous()
            else:
                raise ValueError(f"Invalid alpha shape: {tuple(alpha_t.shape)}")
            if torch.any(conc <= 0):
                raise ValueError("All Dirichlet concentration values must be > 0")

        gamma_samples = torch._standard_gamma(conc).clamp(min=1e-8)
        w = gamma_samples.to(device=device, dtype=torch.float32)

        eff_min_active = min(max(1, int(min_active)), n_views)
        if drop_prob > 0.0 and n_views > eff_min_active:
            if not (0.0 <= drop_prob < 1.0):
                raise ValueError(f"drop_prob must be in [0, 1), got {drop_prob}")
            keep_mask = torch.rand((batch_size, n_views), device=device) >= float(drop_prob)
            # Guarantee at least `eff_min_active` active views per row
            needs_rescue = keep_mask.sum(dim=-1, keepdim=True) < eff_min_active
            if torch.any(needs_rescue):
                rescue_scores = torch.rand((batch_size, n_views), device=device)
                top_idx = rescue_scores.topk(eff_min_active, dim=-1).indices
                force_keep = torch.zeros_like(keep_mask).scatter_(-1, top_idx, True)
                keep_mask = torch.where(needs_rescue, keep_mask | force_keep, keep_mask)
            w = w * keep_mask.to(dtype=torch.float32)

        w = w / w.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return w.to(dtype=dtype)

    def forward_cloud(
        self,
        embs: Sequence[torch.Tensor],
        alpha: Union[float, Sequence[float], torch.Tensor] = 1.0,
        drop_prob: float = 0.15,
        min_active: int = 2,
        return_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Encodes the 5 teacher embeddings and fuses them using per-sample stochastic
        Dirichlet simplex weights + view dropout.
        """
        f_stack = self.encode_views(embs)
        w = self.sample_dirichlet_weights(
            batch_size=f_stack.shape[0],
            alpha=alpha,
            drop_prob=drop_prob,
            min_active=min_active,
            device=f_stack.device,
            dtype=f_stack.dtype,
        )
        f_cloud = self.fuse_encoded_views(f_stack, weights=w)
        if return_weights:
            return f_cloud, w
        return f_cloud

    def forward(
        self,
        embs: Sequence[torch.Tensor],
        weights: Optional[Union[Sequence[float], torch.Tensor]] = None,
    ) -> torch.Tensor:
        f_stack = self.encode_views(embs)
        return self.fuse_encoded_views(f_stack, weights=weights)
