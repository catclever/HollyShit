import math
from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def project_to_sphere(
    z: torch.Tensor,
    scale_sqrt_d: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Projects latent vectors onto the hypersphere.
    When `scale_sqrt_d=True`, projects onto the sphere of radius sqrt(z_dim) so that
    per-dimension average energy E[z_d^2] = 1.0 (matching FlowDecoder.embed_text).
    """
    z_unit = F.normalize(z, p=2, dim=-1, eps=eps)
    if scale_sqrt_d:
        return z_unit * math.sqrt(float(z.shape[-1]))
    return z_unit


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
    """
    Phase 0 v3 Isotropic God Encoder (PyTorch CUDA/MPS/CPU).
    Preserves 100% state_dict key compatibility with legacy checkpoints:
      - net.layers.{0,2}.{weight,bias}
    Supports optional hypersphere projection of radius sqrt(z_dim).
    """

    def __init__(self, d_model: int, z_dim: int, sphere_scale: bool = False):
        super().__init__()
        self.d_model = int(d_model)
        self.z_dim = int(z_dim)
        self.sphere_scale = bool(sphere_scale)
        self.net = _TwoLayerMlpCUDA(d_model, z_dim)

    @staticmethod
    def project_to_sphere(
        z: torch.Tensor,
        scale_sqrt_d: bool = True,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        return project_to_sphere(z, scale_sqrt_d=scale_sqrt_d, eps=eps)

    def forward(
        self,
        f_t: torch.Tensor,
        normalize_sphere: Optional[bool] = None,
    ) -> torch.Tensor:
        z = self.net(f_t)
        use_sphere = self.sphere_scale if normalize_sphere is None else bool(normalize_sphere)
        if use_sphere:
            z = project_to_sphere(z, scale_sqrt_d=True)
        return z


class _SingleTeacherReconHeadCUDA(nn.Module):
    """
    Lightweight two-layer reverse head D_m: R^{z_dim} -> R^{d_m}
    used during Phase 0 v3 training to ensure z_god is a sufficient statistic
    for all 5 teachers without losing any individual teacher subspace.
    """

    def __init__(self, z_dim: int, out_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        h_dim = int(hidden_dim) if hidden_dim is not None else int(z_dim)
        self.norm = nn.LayerNorm(z_dim)
        self.lin = nn.Linear(z_dim, out_dim)
        self.fc1 = nn.Linear(z_dim, h_dim)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(h_dim, out_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.norm(z)
        return self.lin(h) + self.fc2(self.act(self.fc1(h)))


class MultiViewReconstructionHeadsCUDA(nn.Module):
    """
    5-way reverse reconstruction heads {D_m}_{m=1}^M mapping z_god in R^{z_dim}
    back to each teacher's native embedding space R^{d_m}.
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
        self.heads = nn.ModuleList(
            [_SingleTeacherReconHeadCUDA(self.z_dim, dim, hidden_dim=hidden_dim) for dim in self.emb_dims]
        )

    def forward(self, z: torch.Tensor) -> List[torch.Tensor]:
        return [head(z) for head in self.heads]


class AsymmetricNextClauseHeadCUDA(nn.Module):
    """
    Asymmetric forward transition head W_fwd: R^{z_dim} -> R^{z_dim}
    used during Phase 0 v3 training for contiguous-clause directional InfoNCE
    (predicting z_{t+1} from z_t without forcing z_t and z_{t+1} to collapse).
    """

    def __init__(self, z_dim: int, use_norm: bool = True):
        super().__init__()
        self.z_dim = int(z_dim)
        self.norm = nn.LayerNorm(self.z_dim) if use_norm else nn.Identity()
        self.proj = nn.Linear(self.z_dim, self.z_dim, bias=False)
        self._reset_parameters()

    def _reset_parameters(self):
        with torch.no_grad():
            eye = torch.eye(self.z_dim)
            orth = torch.empty(self.z_dim, self.z_dim)
            nn.init.orthogonal_(orth)
            self.proj.weight.copy_(0.7 * eye + 0.3 * orth)

    def forward(self, z_t: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(z_t))
