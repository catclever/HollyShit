import math
from typing import List, Optional, Sequence

import mlx.core as mx
import mlx.nn as nn


def project_to_sphere(
    z: mx.array,
    scale_sqrt_d: bool = True,
    eps: float = 1e-8,
) -> mx.array:
    """
    Projects latent vectors onto the hypersphere in MLX.
    When `scale_sqrt_d=True`, projects onto the sphere of radius sqrt(z_dim).
    """
    norm = mx.sqrt(mx.sum(mx.square(z), axis=-1, keepdims=True) + eps)
    z_unit = z / norm
    if scale_sqrt_d:
        return z_unit * math.sqrt(float(z.shape[-1]))
    return z_unit


class GodEncoder(nn.Module):
    """
    Phase 0 v3 Single Unique God Encoder (MLX).
    Maps the unified fused sensory stream (f_t in d_model) into the
    isotropic truth anchor space (z_target in z_dim).
    Preserves 100% parameter key compatibility with GodEncoderCUDA:
      - net.layers.{0,2}.{weight,bias}
    """

    def __init__(self, d_model: int, z_dim: int, sphere_scale: bool = False):
        super().__init__()
        self.d_model = int(d_model)
        self.z_dim = int(z_dim)
        self.sphere_scale = bool(sphere_scale)
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, z_dim),
        )

    @staticmethod
    def project_to_sphere(
        z: mx.array,
        scale_sqrt_d: bool = True,
        eps: float = 1e-8,
    ) -> mx.array:
        return project_to_sphere(z, scale_sqrt_d=scale_sqrt_d, eps=eps)

    def __call__(
        self,
        f_t: mx.array,
        normalize_sphere: Optional[bool] = None,
    ) -> mx.array:
        z = self.net(f_t)
        use_sphere = self.sphere_scale if normalize_sphere is None else bool(normalize_sphere)
        if use_sphere:
            z = project_to_sphere(z, scale_sqrt_d=True)
        return z


class _SingleTeacherReconHead(nn.Module):
    def __init__(self, z_dim: int, out_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        h_dim = int(hidden_dim) if hidden_dim is not None else int(z_dim)
        self.norm = nn.LayerNorm(z_dim)
        self.lin = nn.Linear(z_dim, out_dim)
        self.fc1 = nn.Linear(z_dim, h_dim)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(h_dim, out_dim)

    def __call__(self, z: mx.array) -> mx.array:
        h = self.norm(z)
        return self.lin(h) + self.fc2(self.act(self.fc1(h)))


class MultiViewReconstructionHeads(nn.Module):
    """
    MLX equivalent of MultiViewReconstructionHeadsCUDA.
    """

    def __init__(
        self,
        z_dim: int,
        emb_dims: Sequence[int],
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        self.z_dim = int(z_dim)
        self.emb_dims = [int(d) for d in emb_dims]
        self.heads = [
            _SingleTeacherReconHead(self.z_dim, dim, hidden_dim=hidden_dim) for dim in self.emb_dims
        ]

    def __call__(self, z: mx.array) -> List[mx.array]:
        return [head(z) for head in self.heads]


class AsymmetricNextClauseHead(nn.Module):
    """
    MLX equivalent of AsymmetricNextClauseHeadCUDA.
    """

    def __init__(self, z_dim: int, use_norm: bool = True):
        super().__init__()
        self.z_dim = int(z_dim)
        self.norm = nn.LayerNorm(self.z_dim) if use_norm else None
        self.proj = nn.Linear(self.z_dim, self.z_dim, bias=False)

    def __call__(self, z_t: mx.array) -> mx.array:
        h = self.norm(z_t) if self.norm is not None else z_t
        return self.proj(h)
