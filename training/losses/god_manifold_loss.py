import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.adapter_cuda import SensoryFuserCUDA
from model.god_encoder_cuda import (
    AsymmetricNextClauseHeadCUDA,
    GodEncoderCUDA,
    MultiViewReconstructionHeadsCUDA,
    project_to_sphere,
)


@dataclass
class GodManifoldLossWeights:
    """Hyperparameter weights for Phase 0 v3 Isotropic God Manifold training."""

    w_recon: float = 2.5
    w_rel: float = 2.0
    w_cloud: float = 1.0
    w_iso: float = 1.0
    w_unif: float = 0.5
    w_momentum: float = 0.3
    w_next: float = 0.5
    rel_tau: float = 0.07
    cloud_tau: float = 0.10
    next_tau: float = 0.10
    dirichlet_alpha: float = 1.0
    modality_drop_prob: float = 0.15
    min_active_views: int = 2
    mixup_beta: float = 2.0


class ManifoldFIFOQueue:
    """
    Detached cross-batch FIFO queue of recent z_god vectors (e.g. max_size=2048 >= 2 * z_dim).
    Breaks the single-batch rank ceiling (min(B - 1, D)) during VICReg covariance whitening
    and hyperspherical uniformity regularization when mini-batch size B < z_dim.
    Stored in float32 so cross-batch covariance accumulation never loses precision under AMP.
    """

    def __init__(self, max_size: int = 2048):
        self.max_size = int(max_size)
        self._buffer: Optional[torch.Tensor] = None

    def get(self) -> Optional[torch.Tensor]:
        return self._buffer

    def clear(self):
        self._buffer = None

    @torch.no_grad()
    def enqueue(self, z: torch.Tensor):
        if self.max_size <= 0 or z.numel() == 0:
            return
        z_det = z.detach().float()
        if self._buffer is None:
            self._buffer = z_det[-self.max_size :].clone()
        else:
            buf = self._buffer.to(device=z_det.device, dtype=torch.float32)
            combined = torch.cat([buf, z_det], dim=0)
            if combined.shape[0] > self.max_size:
                combined = combined[-self.max_size :]
            self._buffer = combined.clone()


def _mps_safe_teacher_embs(
    embs: Sequence[torch.Tensor],
    fuser: nn.Module,
) -> Sequence[torch.Tensor]:
    """
    Workaround for PyTorch 2.10 Apple MPS `native_layer_norm_backward` bug:
    when `input.requires_grad == False`, MPS leaves `LayerNorm.weight.grad` and
    `LayerNorm.bias.grad` filled with uninitialized Metal memory (`inf` / garbage),
    corrupting `clip_grad_norm_` and stalling adapter convergence.
    Attaching a local leaf `requires_grad_(True)` view on MPS forces the full backward
    kernel path so adapter LayerNorm parameter gradients match CPU/CUDA to ~1e-6.
    Zero overhead on CUDA and CPU.
    """
    if (
        len(embs) == 0
        or embs[0].device.type != "mps"
        or not torch.is_grad_enabled()
        or not any(p.requires_grad for p in fuser.parameters())
    ):
        return embs
    return [e if e.requires_grad else e.detach().requires_grad_(True) for e in embs]


def compute_teacher_consensus_matrix(
    teacher_embs: Sequence[torch.Tensor],
    teacher_weights: Optional[Sequence[float]] = None,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Centers and L2-normalizes each teacher's batch embeddings to form per-teacher
    whitened cosine similarity matrices S^{(m)} in R^{B x B}, and averages them into
    the consensus similarity matrix S_bar (always in float32 for AMP safety).
    """
    n_teachers = len(teacher_embs)
    if n_teachers == 0:
        raise ValueError("teacher_embs must be non-empty")

    if teacher_weights is None:
        w = [1.0 / n_teachers] * n_teachers
    else:
        if len(teacher_weights) != n_teachers:
            raise ValueError(f"Expected {n_teachers} teacher_weights, got {len(teacher_weights)}")
        w_sum = sum(teacher_weights)
        w = [float(x) / max(w_sum, eps) for x in teacher_weights]

    per_teacher_s: List[torch.Tensor] = []
    s_consensus: Optional[torch.Tensor] = None

    with torch.autocast(device_type=teacher_embs[0].device.type, enabled=False):
        for weight, emb in zip(w, teacher_embs):
            emb_f = emb.float()
            e_centered = emb_f - emb_f.mean(dim=0, keepdim=True)
            e_norm = F.normalize(e_centered, p=2, dim=-1, eps=eps)
            s_m = e_norm @ e_norm.T
            per_teacher_s.append(s_m)
            s_consensus = s_m * weight if s_consensus is None else (s_consensus + s_m * weight)

    assert s_consensus is not None
    return s_consensus, per_teacher_s


def compute_single_view_recon_loss(
    z: torch.Tensor,
    teacher_embs: Sequence[torch.Tensor],
    recon_heads: MultiViewReconstructionHeadsCUDA,
    mse_weight: float = 0.1,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, List[float]]:
    """
    Computes Multi-View Teacher Reconstruction loss:
      mean_{m=1..M} [ (1 - cos(D_m(z), e_m)) + mse_weight * MSE(L2Norm(D_m(z)), L2Norm(e_m)) ]
    Returns:
        loss: scalar tensor (float32)
        per_teacher_cos: list of mean cosine similarities per teacher
    """
    preds = recon_heads(z)
    if len(preds) != len(teacher_embs):
        raise ValueError(f"recon_heads produced {len(preds)} outputs, expected {len(teacher_embs)}")

    total_loss = torch.zeros((), device=z.device, dtype=torch.float32)
    per_teacher_cos: List[float] = []

    with torch.autocast(device_type=z.device.type, enabled=False):
        for pred_m, target_m in zip(preds, teacher_embs):
            pred_norm = F.normalize(pred_m.float(), p=2, dim=-1, eps=eps)
            targ_norm = F.normalize(target_m.float(), p=2, dim=-1, eps=eps)
            cos_sim = (pred_norm * targ_norm).sum(dim=-1)
            cos_loss = (1.0 - cos_sim).mean()
            mse_loss = (pred_norm - targ_norm).pow(2).sum(dim=-1).mean()
            total_loss = total_loss + cos_loss + mse_weight * mse_loss
            per_teacher_cos.append(float(cos_sim.detach().mean().item()))

    total_loss = total_loss / float(len(teacher_embs))
    return total_loss, per_teacher_cos


def compute_centered_cosine_matrix(z: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    with torch.autocast(device_type=z.device.type, enabled=False):
        z_f = z.float()
        z_centered = z_f - z_f.mean(dim=0, keepdim=True)
        z_norm = F.normalize(z_centered, p=2, dim=-1, eps=eps)
        return z_norm @ z_norm.T


def compute_relational_distill_loss(
    z: torch.Tensor,
    s_consensus: torch.Tensor,
    tau: float = 0.07,
    frob_weight: float = 1.0,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Batch Relational Similarity Matrix Distillation:
      Row-wise KL(softmax(S_bar / tau) || softmax(S_god / tau)) (diagonal masked)
      + Frobenius MSE ||S_god - S_bar||_F^2 on off-diagonal entries.
    """
    bsz = z.shape[0]
    s_god = compute_centered_cosine_matrix(z, eps=eps)
    with torch.autocast(device_type=z.device.type, enabled=False):
        s_cons_f = s_consensus.detach().float()
        if bsz <= 2:
            return F.mse_loss(s_god, s_cons_f), s_god

        off_mask = ~torch.eye(bsz, dtype=torch.bool, device=z.device)
        s_god_off = s_god[off_mask].view(bsz, bsz - 1)
        s_cons_off = s_cons_f[off_mask].view(bsz, bsz - 1)

        log_q = F.log_softmax(s_god_off / tau, dim=-1)
        p = F.softmax(s_cons_off / tau, dim=-1)
        loss_kl = F.kl_div(log_q, p, reduction="batchmean") * (tau * 10.0)
        loss_frob = (s_god_off - s_cons_off).pow(2).sum(dim=-1).mean()
        loss_rel = loss_kl + frob_weight * loss_frob

    return loss_rel, s_god


def compute_vicreg_and_uniformity_loss(
    z: torch.Tensor,
    z_queue: Optional[torch.Tensor] = None,
    gamma: float = 1.0,
    eps: float = 1e-4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes (strictly in float32 so 1024x1024 covariance Frobenius norms & logsumexp never overflow FP16):
      1) loss_iso: VICReg variance hinge + orthogonally-invariant full-spectrum covariance
         whitening (1/D) * ||C_mix - I_D||_F^2 where C_mix = 0.5 * C_batch + 0.5 * C_queue
         (with batch and queue centered separately so mean drift never injects a rank-1 spike).
      2) loss_unif: Hyperspherical radial uniformity log E_{i != j}[exp(-2 ||z_hat_i - z_hat_j||_2^2)]
         within the current batch.
    """
    bsz, dim = z.shape
    if bsz <= 1:
        zero = torch.zeros((), device=z.device, dtype=torch.float32)
        return zero, zero

    with torch.autocast(device_type=z.device.type, enabled=False):
        z_f = z.float()
        z_c = z_f - z_f.mean(dim=0, keepdim=True)
        var_batch = z_c.pow(2).sum(dim=0) / float(bsz - 1)
        std_batch = torch.sqrt(var_batch + eps)
        loss_var = F.relu(gamma - std_batch).mean()
        loss_center = z_f.mean(dim=0).pow(2).mean() * 0.1

        cov_batch = (z_c.T @ z_c) / float(bsz - 1)
        if z_queue is not None and z_queue.shape[0] > 2:
            z_q = z_queue.detach().to(device=z.device, dtype=torch.float32)
            z_q_c = z_q - z_q.mean(dim=0, keepdim=True)
            cov_q = (z_q_c.T @ z_q_c) / float(z_q.shape[0] - 1)
            cov_mix = 0.5 * cov_batch + 0.5 * cov_q
        else:
            cov_mix = cov_batch

        # Orthogonally-invariant full spectrum whitening: (1/D) * ||C_mix - I_D||_F^2 == (1/D) * sum_k (lambda_k - 1)^2
        eye_d = torch.eye(dim, device=z.device, dtype=torch.float32)
        loss_cov_full = (cov_mix - eye_d).pow(2).sum() / float(dim)
        loss_iso = loss_var + loss_center + 0.35 * loss_cov_full

        # Hyperspherical radial uniformity within current batch only
        z_unit = F.normalize(z_f, p=2, dim=-1, eps=1e-8)
        sq_pdist = (2.0 - 2.0 * (z_unit @ z_unit.T)).clamp(min=0.0)
        off_mask = ~torch.eye(bsz, dtype=torch.bool, device=z.device)
        off_pdist = sq_pdist[off_mask]
        loss_unif = torch.logsumexp(-2.0 * off_pdist, dim=0) - math.log(
            float(max(off_pdist.numel(), 1))
        )

    return loss_iso, loss_unif


def _sample_beta_weights(
    batch_size: int,
    beta_param: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Device-safe Beta(beta_param, beta_param) sampling on CPU then moved to target device."""
    conc = torch.full((batch_size, 1), float(beta_param), dtype=torch.float32, device="cpu")
    g1 = torch._standard_gamma(conc).clamp(min=1e-8)
    g2 = torch._standard_gamma(conc).clamp(min=1e-8)
    lam = g1 / (g1 + g2)
    return lam.to(device=device, dtype=dtype)


def compute_cloud_and_mixup_loss(
    f_mean: torch.Tensor,
    z_mean: torch.Tensor,
    z_view_a: torch.Tensor,
    z_view_b: torch.Tensor,
    teacher_embs: Sequence[torch.Tensor],
    s_consensus: torch.Tensor,
    god_encoder: GodEncoderCUDA,
    recon_heads: MultiViewReconstructionHeadsCUDA,
    cloud_tau: float = 0.10,
    mixup_beta: float = 2.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Computes:
      1) Simplex cloud consistency between independent Dirichlet views z_view_a and z_view_b
         via InfoNCE + (1 - cos(z_a, z_b))
      2) Neighbor Manifold Mixup between each sample i and its top-1 teacher semantic neighbor pi(i),
         forcing both GodEncoder(f_mix) and latent spherical interpolation to reconstruct
         interpolated teacher embeddings and agree with each other (eliminating cavities).
    """
    bsz = z_mean.shape[0]
    with torch.autocast(device_type=z_mean.device.type, enabled=False):
        z_a_norm = F.normalize(z_view_a.float(), p=2, dim=-1, eps=eps)
        z_b_norm = F.normalize(z_view_b.float(), p=2, dim=-1, eps=eps)
        z_m_norm = F.normalize(z_mean.float(), p=2, dim=-1, eps=eps)

        loss_align = (
            (1.0 - (z_a_norm * z_b_norm).sum(dim=-1)).mean()
            + 0.5 * (1.0 - (z_a_norm * z_m_norm).sum(dim=-1)).mean()
            + 0.5 * (1.0 - (z_b_norm * z_m_norm).sum(dim=-1)).mean()
        )

        if bsz > 1:
            logits_ab = (z_a_norm @ z_b_norm.T) / cloud_tau
            labels = torch.arange(bsz, device=z_mean.device)
            loss_nce = 0.5 * (
                F.cross_entropy(logits_ab, labels) + F.cross_entropy(logits_ab.T, labels)
            )
        else:
            loss_nce = torch.zeros((), device=z_mean.device, dtype=torch.float32)

    if bsz <= 1:
        return loss_align + 0.2 * loss_nce

    # Find top-1 semantic neighbor pi(i) under S_bar (AMP-safe float32 + -inf diagonal)
    s_masked = s_consensus.detach().float().clone()
    s_masked.fill_diagonal_(float("-inf"))
    nn_idx = s_masked.argmax(dim=-1)

    lam_f32 = _sample_beta_weights(bsz, mixup_beta, device=z_mean.device, dtype=torch.float32)
    lam_f = lam_f32.to(dtype=f_mean.dtype)
    f_mix = lam_f * f_mean + (1.0 - lam_f) * f_mean[nn_idx]
    z_mix = god_encoder(f_mix)

    # Spherical interpolation in latent space (float32)
    with torch.autocast(device_type=z_mean.device.type, enabled=False):
        z_interp_dir = F.normalize(
            lam_f32 * z_m_norm + (1.0 - lam_f32) * z_m_norm[nn_idx], p=2, dim=-1, eps=eps
        )
        z_interp = z_interp_dir * math.sqrt(float(z_mean.shape[-1]))
        loss_geodesic = (
            1.0 - (F.normalize(z_mix.float(), p=2, dim=-1, eps=eps) * z_interp_dir).sum(dim=-1)
        ).mean()
        mixed_teacher_embs = [
            lam_f32 * emb.float() + (1.0 - lam_f32) * emb[nn_idx].float() for emb in teacher_embs
        ]

    loss_mix_recon_enc, _ = compute_single_view_recon_loss(
        z_mix, mixed_teacher_embs, recon_heads, eps=eps
    )
    loss_mix_recon_lat, _ = compute_single_view_recon_loss(
        z_interp, mixed_teacher_embs, recon_heads, eps=eps
    )
    loss_mix_recon = 0.5 * (loss_mix_recon_enc + loss_mix_recon_lat)

    return 0.6 * loss_align + 0.2 * loss_nce + 0.3 * loss_geodesic + 0.6 * loss_mix_recon


def compute_momentum_and_next_clause_loss(
    triplet_z: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    triplet_teacher_embs: Optional[
        Tuple[Sequence[torch.Tensor], Sequence[torch.Tensor], Sequence[torch.Tensor]]
    ],
    next_clause_head: Optional[AsymmetricNextClauseHeadCUDA],
    batch_z_negatives: Optional[torch.Tensor] = None,
    next_tau: float = 0.10,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes:
      1) loss_momentum: trajectory momentum continuity 1 - cos(z_{t+1} - z_t, z_t - z_{t-1})
         gated by teacher semantic continuity so unrelated document boundaries are ignored.
      2) loss_next: asymmetric next-clause InfoNCE using W_fwd(z_t) to retrieve z_{t+1}.
    """
    z_prev, z_curr, z_next = triplet_z
    n_trip = z_curr.shape[0]
    if n_trip == 0:
        zero = torch.zeros((), device=z_curr.device, dtype=torch.float32)
        return zero, zero

    with torch.autocast(device_type=z_curr.device.type, enabled=False):
        if triplet_teacher_embs is not None:
            embs_prev, embs_curr, embs_next = triplet_teacher_embs
            cos_01_list = []
            cos_12_list = []
            for e0, e1, e2 in zip(embs_prev, embs_curr, embs_next):
                e0_n = F.normalize(e0.float(), p=2, dim=-1, eps=eps)
                e1_n = F.normalize(e1.float(), p=2, dim=-1, eps=eps)
                e2_n = F.normalize(e2.float(), p=2, dim=-1, eps=eps)
                cos_01_list.append((e0_n * e1_n).sum(dim=-1))
                cos_12_list.append((e1_n * e2_n).sum(dim=-1))
            mean_cos_01 = torch.stack(cos_01_list, dim=0).mean(dim=0)
            mean_cos_12 = torch.stack(cos_12_list, dim=0).mean(dim=0)
            gate_01 = torch.sigmoid((mean_cos_01 - 0.15) / 0.10).detach()
            gate_12 = torch.sigmoid((mean_cos_12 - 0.15) / 0.10).detach()
            gate_trip = (gate_01 * gate_12).clamp(min=0.05)
        else:
            gate_01 = torch.ones(n_trip, device=z_curr.device, dtype=torch.float32)
            gate_12 = torch.ones(n_trip, device=z_curr.device, dtype=torch.float32)
            gate_trip = torch.ones(n_trip, device=z_curr.device, dtype=torch.float32)

        z_prev_f = z_prev.float()
        z_curr_f = z_curr.float()
        z_next_f = z_next.float()
        delta_1 = z_curr_f - z_prev_f
        delta_2 = z_next_f - z_curr_f
        cos_mom = F.cosine_similarity(delta_2, delta_1, dim=-1, eps=eps)
        loss_momentum = ((1.0 - cos_mom) * gate_trip).sum() / gate_trip.sum().clamp(min=eps)

    if next_clause_head is None:
        return loss_momentum, torch.zeros((), device=z_curr.device, dtype=torch.float32)

    raw_pred_curr = next_clause_head(z_prev)
    raw_pred_next = next_clause_head(z_curr)
    with torch.autocast(device_type=z_curr.device.type, enabled=False):
        pred_curr = F.normalize(raw_pred_curr.float(), p=2, dim=-1, eps=eps)
        pred_next = F.normalize(raw_pred_next.float(), p=2, dim=-1, eps=eps)
        targ_curr = F.normalize(z_curr_f, p=2, dim=-1, eps=eps)
        targ_next = F.normalize(z_next_f, p=2, dim=-1, eps=eps)

        if batch_z_negatives is not None and batch_z_negatives.shape[0] > 0:
            neg_pool = F.normalize(batch_z_negatives.detach().float(), p=2, dim=-1, eps=eps)
            pool_curr = torch.cat([targ_curr, neg_pool], dim=0)
            pool_next = torch.cat([targ_next, neg_pool], dim=0)
        else:
            pool_curr = targ_curr
            pool_next = targ_next

        logits_01 = (pred_curr @ pool_curr.T) / next_tau
        logits_12 = (pred_next @ pool_next.T) / next_tau
        targets = torch.arange(n_trip, device=z_curr.device)

        ce_01 = F.cross_entropy(logits_01, targets, reduction="none")
        ce_12 = F.cross_entropy(logits_12, targets, reduction="none")
        loss_nce = 0.5 * (
            (ce_01 * gate_01).sum() / gate_01.sum().clamp(min=eps)
            + (ce_12 * gate_12).sum() / gate_12.sum().clamp(min=eps)
        )
        loss_dir_cos = 0.5 * (
            ((1.0 - (pred_curr * targ_curr).sum(dim=-1)) * gate_01).sum()
            / gate_01.sum().clamp(min=eps)
            + ((1.0 - (pred_next * targ_next).sum(dim=-1)) * gate_12).sum()
            / gate_12.sum().clamp(min=eps)
        )
        loss_next = 0.15 * loss_nce + loss_dir_cos

    return loss_momentum, loss_next


def compute_god_manifold_loss(
    fuser: SensoryFuserCUDA,
    god_encoder: GodEncoderCUDA,
    recon_heads: MultiViewReconstructionHeadsCUDA,
    teacher_embs: Sequence[torch.Tensor],
    next_clause_head: Optional[AsymmetricNextClauseHeadCUDA] = None,
    triplet_teacher_embs: Optional[
        Tuple[Sequence[torch.Tensor], Sequence[torch.Tensor], Sequence[torch.Tensor]]
    ] = None,
    weights: Optional[GodManifoldLossWeights] = None,
    manifold_queue: Optional[ManifoldFIFOQueue] = None,
    z_history_queue: Optional[torch.Tensor] = None,
    enqueue_to_queue: bool = True,
) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, torch.Tensor]]:
    """
    Full Phase 0 v3 Isotropic God Manifold Loss computation.
    """
    cfg = weights if weights is not None else GodManifoldLossWeights()
    bsz = teacher_embs[0].shape[0]

    # 1. Encode all 5 teacher views once and form centroid + 2 stochastic Dirichlet simplex views
    f_stack = fuser.encode_views(_mps_safe_teacher_embs(teacher_embs, fuser))
    f_mean = fuser.fuse_encoded_views(f_stack, weights=None)

    w_a = fuser.sample_dirichlet_weights(
        batch_size=bsz,
        alpha=cfg.dirichlet_alpha,
        drop_prob=cfg.modality_drop_prob,
        min_active=cfg.min_active_views,
        device=f_stack.device,
        dtype=torch.float32,
    ).to(dtype=f_stack.dtype)
    w_b = fuser.sample_dirichlet_weights(
        batch_size=bsz,
        alpha=cfg.dirichlet_alpha,
        drop_prob=cfg.modality_drop_prob,
        min_active=cfg.min_active_views,
        device=f_stack.device,
        dtype=torch.float32,
    ).to(dtype=f_stack.dtype)
    f_view_a = fuser.fuse_encoded_views(f_stack, weights=w_a)
    f_view_b = fuser.fuse_encoded_views(f_stack, weights=w_b)

    z_mean = god_encoder(f_mean)
    z_view_a = god_encoder(f_view_a)
    z_view_b = god_encoder(f_view_b)

    # 2. Multi-View Teacher Reconstruction Loss
    loss_recon_mean, per_teacher_cos = compute_single_view_recon_loss(
        z_mean, teacher_embs, recon_heads
    )
    loss_recon_a, _ = compute_single_view_recon_loss(z_view_a, teacher_embs, recon_heads)
    loss_recon_b, _ = compute_single_view_recon_loss(z_view_b, teacher_embs, recon_heads)
    loss_recon = (loss_recon_mean + loss_recon_a + loss_recon_b) / 3.0

    # 3. Batch Relational Similarity Matrix Distillation
    s_consensus, _ = compute_teacher_consensus_matrix(teacher_embs)
    loss_rel_mean, s_god = compute_relational_distill_loss(z_mean, s_consensus, tau=cfg.rel_tau)
    loss_rel_a, _ = compute_relational_distill_loss(z_view_a, s_consensus, tau=cfg.rel_tau)
    loss_rel_b, _ = compute_relational_distill_loss(z_view_b, s_consensus, tau=cfg.rel_tau)
    loss_rel = 0.5 * loss_rel_mean + 0.25 * loss_rel_a + 0.25 * loss_rel_b

    # 4. Simplex Cloud Consistency + Neighbor Manifold Mixup
    loss_cloud = compute_cloud_and_mixup_loss(
        f_mean=f_mean,
        z_mean=z_mean,
        z_view_a=z_view_a,
        z_view_b=z_view_b,
        teacher_embs=teacher_embs,
        s_consensus=s_consensus,
        god_encoder=god_encoder,
        recon_heads=recon_heads,
        cloud_tau=cfg.cloud_tau,
        mixup_beta=cfg.mixup_beta,
    )

    # 5. VICReg Covariance Whitening + Hyperspherical Uniformity (per-view, NEVER stacking views of same clause)
    q_tensor = manifold_queue.get() if manifold_queue is not None else z_history_queue
    loss_iso_m, loss_unif_m = compute_vicreg_and_uniformity_loss(z_mean, z_queue=q_tensor)
    loss_iso_a, _ = compute_vicreg_and_uniformity_loss(z_view_a, z_queue=None)
    loss_iso_b, _ = compute_vicreg_and_uniformity_loss(z_view_b, z_queue=None)
    loss_iso = 0.6 * loss_iso_m + 0.2 * loss_iso_a + 0.2 * loss_iso_b
    loss_unif = loss_unif_m

    if manifold_queue is not None and enqueue_to_queue:
        manifold_queue.enqueue(z_mean.detach())

    # 6. Contiguous-Clause Trajectory Momentum & Asymmetric Next-Clause InfoNCE
    if triplet_teacher_embs is not None:
        embs_prev, embs_curr, embs_next = triplet_teacher_embs
        z_prev = god_encoder(fuser(_mps_safe_teacher_embs(embs_prev, fuser)))
        z_curr = god_encoder(fuser(_mps_safe_teacher_embs(embs_curr, fuser)))
        z_next = god_encoder(fuser(_mps_safe_teacher_embs(embs_next, fuser)))
        triplet_z = (z_prev, z_curr, z_next)
        loss_momentum, loss_next = compute_momentum_and_next_clause_loss(
            triplet_z=triplet_z,
            triplet_teacher_embs=triplet_teacher_embs,
            next_clause_head=next_clause_head,
            batch_z_negatives=z_mean,
            next_tau=cfg.next_tau,
        )
    elif bsz >= 3:
        embs_prev = [e[:-2] for e in teacher_embs]
        embs_curr = [e[1:-1] for e in teacher_embs]
        embs_next = [e[2:] for e in teacher_embs]
        triplet_z = (z_mean[:-2], z_mean[1:-1], z_mean[2:])
        loss_momentum, loss_next = compute_momentum_and_next_clause_loss(
            triplet_z=triplet_z,
            triplet_teacher_embs=(embs_prev, embs_curr, embs_next),
            next_clause_head=next_clause_head,
            batch_z_negatives=z_mean,
            next_tau=cfg.next_tau,
        )
    else:
        loss_momentum = torch.zeros((), device=z_mean.device, dtype=torch.float32)
        loss_next = torch.zeros((), device=z_mean.device, dtype=torch.float32)

    total_loss = (
        cfg.w_recon * loss_recon
        + cfg.w_rel * loss_rel
        + cfg.w_cloud * loss_cloud
        + cfg.w_iso * loss_iso
        + cfg.w_unif * loss_unif
        + cfg.w_momentum * loss_momentum
        + cfg.w_next * loss_next
    )

    metrics = {
        "total_loss": float(total_loss.detach().item()),
        "loss_recon": float(loss_recon.detach().item()),
        "loss_rel": float(loss_rel.detach().item()),
        "loss_cloud": float(loss_cloud.detach().item()),
        "loss_iso": float(loss_iso.detach().item()),
        "loss_unif": float(loss_unif.detach().item()),
        "loss_momentum": float(loss_momentum.detach().item()),
        "loss_next": float(loss_next.detach().item()),
        "mean_teacher_recon_cos": float(sum(per_teacher_cos) / max(len(per_teacher_cos), 1)),
    }
    aux_tensors = {
        "z_mean": z_mean.detach(),
        "z_view_a": z_view_a.detach(),
        "z_view_b": z_view_b.detach(),
        "s_consensus": s_consensus.detach(),
        "s_god": s_god.detach(),
    }
    return total_loss, metrics, aux_tensors


def _compute_participation_ratio(z: torch.Tensor, eps: float = 1e-8) -> Tuple[float, float]:
    """
    Computes exact, un-doctored Gram-matrix Participation Ratio:
      PR = (sum_i lambda_i)^2 / (sum_i lambda_i^2) = (Tr(G))^2 / ||G||_F^2
    for both raw centered z and variance-whitened z.
    """
    n_samples, _dim = z.shape
    if n_samples <= 1:
        return 1.0, 1.0

    with torch.autocast(device_type=z.device.type, enabled=False):
        z_f = z.float()
        z_c = z_f - z_f.mean(dim=0, keepdim=True)
        gram_raw = (z_c @ z_c.T) / float(n_samples - 1)
        tr_raw = torch.diagonal(gram_raw).sum()
        frob_sq_raw = gram_raw.pow(2).sum().clamp(min=eps)
        raw_pr = float((tr_raw.pow(2) / frob_sq_raw).item())

        std = torch.sqrt(z_c.var(dim=0, unbiased=True) + 1e-6)
        z_w = z_c / std.unsqueeze(0)
        gram_w = (z_w @ z_w.T) / float(n_samples - 1)
        tr_w = torch.diagonal(gram_w).sum()
        frob_sq_w = gram_w.pow(2).sum().clamp(min=eps)
        whitened_pr = float((tr_w.pow(2) / frob_sq_w).item())

    return raw_pr, whitened_pr


def _spearman_corr_torch(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> float:
    """Computes exact Spearman rank correlation between flattened 1D tensors x and y."""
    x_flat = x.detach().float().reshape(-1)
    y_flat = y.detach().float().reshape(-1)
    if x_flat.numel() <= 1:
        return 0.0

    rank_x = torch.argsort(torch.argsort(x_flat)).float()
    rank_y = torch.argsort(torch.argsort(y_flat)).float()

    rx = rank_x - rank_x.mean()
    ry = rank_y - rank_y.mean()
    denom = (rx.norm() * ry.norm()).clamp(min=eps)
    return float((rx @ ry / denom).item())


def _roc_auc_torch(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> float:
    """Computes exact Mann-Whitney U ROC-AUC between positive and negative 1D score tensors."""
    pos = pos_scores.detach().float().reshape(-1)
    neg = neg_scores.detach().float().reshape(-1)
    if pos.numel() == 0 or neg.numel() == 0:
        return 0.5
    diff = pos[:, None] - neg[None, :]
    auc = ((diff > 0).float() + 0.5 * (diff == 0).float()).mean()
    return float(auc.item())


@torch.no_grad()
def evaluate_god_manifold_health(
    fuser: SensoryFuserCUDA,
    god_encoder: GodEncoderCUDA,
    recon_heads: MultiViewReconstructionHeadsCUDA,
    teacher_embs: Sequence[torch.Tensor],
    next_clause_head: Optional[AsymmetricNextClauseHeadCUDA] = None,
    triplet_teacher_embs: Optional[
        Tuple[Sequence[torch.Tensor], Sequence[torch.Tensor], Sequence[torch.Tensor]]
    ] = None,
    z_eval_pool: Optional[torch.Tensor] = None,
    dirichlet_alpha: float = 1.0,
    modality_drop_prob: float = 0.15,
) -> Dict[str, float]:
    """
    Quantitative Manifold Health Probe for Phase 0 v3.
    When `z_eval_pool` (e.g. N >= 512..2048 accumulated or dataset-wide z_god vectors) is
    provided, `participation_ratio` is computed on `z_eval_pool` to avoid the single-batch
    Cauchy-Schwarz rank ceiling (min(B-1, D)).
    """
    bsz = teacher_embs[0].shape[0]
    f_stack = fuser.encode_views(teacher_embs)
    f_mean = fuser.fuse_encoded_views(f_stack, weights=None)
    z_mean = god_encoder(f_mean)

    cloud_views_z = [z_mean]
    for _ in range(2):
        w = fuser.sample_dirichlet_weights(
            batch_size=bsz,
            alpha=dirichlet_alpha,
            drop_prob=modality_drop_prob,
            min_active=2,
            device=f_stack.device,
            dtype=torch.float32,
        ).to(dtype=f_stack.dtype)
        cloud_views_z.append(god_encoder(fuser.fuse_encoded_views(f_stack, weights=w)))

    z_view_a = cloud_views_z[1]
    z_view_b = cloud_views_z[2]

    # 1. Teacher reconstruction cosine
    _, per_teacher_cos = compute_single_view_recon_loss(z_mean, teacher_embs, recon_heads)
    mean_teacher_recon_cos = float(sum(per_teacher_cos) / max(len(per_teacher_cos), 1))

    # 2. Relational Spearman correlation
    s_consensus, _ = compute_teacher_consensus_matrix(teacher_embs)
    s_god = compute_centered_cosine_matrix(z_mean)
    if bsz > 2:
        off_mask = ~torch.eye(bsz, dtype=torch.bool, device=z_mean.device)
        relational_spearman = _spearman_corr_torch(s_god[off_mask], s_consensus[off_mask])
    else:
        relational_spearman = 1.0

    # 3. Cloud view top-1 retrieval accuracy
    with torch.autocast(device_type=z_mean.device.type, enabled=False):
        z_a_norm = F.normalize(z_view_a.float(), p=2, dim=-1, eps=1e-8)
        z_b_norm = F.normalize(z_view_b.float(), p=2, dim=-1, eps=1e-8)
        sim_ab = z_a_norm @ z_b_norm.T
        labels = torch.arange(bsz, device=z_mean.device)
        acc_a2b = (sim_ab.argmax(dim=-1) == labels).float().mean().item()
        acc_b2a = (sim_ab.T.argmax(dim=-1) == labels).float().mean().item()
    cloud_view_top1_acc = float(0.5 * (acc_a2b + acc_b2a))

    # 4. Midpoint cavity reconstruction cosine
    if bsz > 1:
        s_masked = s_consensus.float().clone()
        s_masked.fill_diagonal_(float("-inf"))
        nn_idx = s_masked.argmax(dim=-1)

        f_mid = 0.5 * f_mean + 0.5 * f_mean[nn_idx]
        z_mid_enc = god_encoder(f_mid)
        z_mid_lat = project_to_sphere(
            0.5 * (z_mean.float() + z_mean[nn_idx].float()), scale_sqrt_d=True
        )
        mid_teacher_embs = [0.5 * (emb.float() + emb[nn_idx].float()) for emb in teacher_embs]

        _, mid_cos_enc = compute_single_view_recon_loss(z_mid_enc, mid_teacher_embs, recon_heads)
        _, mid_cos_lat = compute_single_view_recon_loss(z_mid_lat, mid_teacher_embs, recon_heads)
        midpoint_cavity_recon_cos = float(
            0.5
            * (
                sum(mid_cos_enc) / max(len(mid_cos_enc), 1)
                + sum(mid_cos_lat) / max(len(mid_cos_lat), 1)
            )
        )
    else:
        midpoint_cavity_recon_cos = mean_teacher_recon_cos

    # 5. Participation Ratio (Effective Dimensionality)
    if triplet_teacher_embs is not None:
        embs_prev, embs_curr, embs_next = triplet_teacher_embs
        z_prev = god_encoder(fuser(embs_prev))
        z_curr = god_encoder(fuser(embs_curr))
        z_next = god_encoder(fuser(embs_next))
    elif bsz >= 2:
        z_curr = z_mean[:-1]
        z_next = z_mean[1:]
    else:
        z_curr = z_mean
        z_next = z_mean

    target_pr_pool = z_eval_pool if z_eval_pool is not None else z_mean
    raw_pr, whitened_pr = _compute_participation_ratio(target_pr_pool)
    single_raw_pr, single_whitened_pr = _compute_participation_ratio(z_mean)
    participation_ratio = max(raw_pr, whitened_pr)

    # 6. Contiguous next-clause accuracy and adjacent AUC
    n_pairs = z_curr.shape[0]
    if n_pairs > 1:
        raw_q_curr = next_clause_head(z_curr) if next_clause_head is not None else z_curr
        with torch.autocast(device_type=z_mean.device.type, enabled=False):
            q_curr = F.normalize(raw_q_curr.float(), p=2, dim=-1, eps=1e-8)
            k_next = F.normalize(z_next.float(), p=2, dim=-1, eps=1e-8)

            sim_next = q_curr @ k_next.T
            pair_labels = torch.arange(n_pairs, device=z_mean.device)
            next_clause_top1_acc = float(
                (sim_next.argmax(dim=-1) == pair_labels).float().mean().item()
            )

            pos_scores = torch.diagonal(sim_next)
            idx_grid = torch.arange(n_pairs, device=z_mean.device)
            neg_mask = (idx_grid[:, None] - idx_grid[None, :]).abs() > 1
            if torch.any(neg_mask):
                neg_scores = sim_next[neg_mask]
            else:
                off_mask_pairs = ~torch.eye(n_pairs, dtype=torch.bool, device=z_mean.device)
                neg_scores = sim_next[off_mask_pairs]
            adjacent_auc = _roc_auc_torch(pos_scores, neg_scores)
    else:
        next_clause_top1_acc = 1.0
        adjacent_auc = 1.0

    return {
        "participation_ratio": float(participation_ratio),
        "raw_participation_ratio": float(raw_pr),
        "whitened_participation_ratio": float(whitened_pr),
        "single_view_participation_ratio": float(max(single_raw_pr, single_whitened_pr)),
        "mean_teacher_recon_cos": float(mean_teacher_recon_cos),
        "relational_spearman": float(relational_spearman),
        "cloud_view_top1_acc": float(cloud_view_top1_acc),
        "next_clause_top1_acc": float(next_clause_top1_acc),
        "adjacent_auc": float(adjacent_auc),
        "midpoint_cavity_recon_cos": float(midpoint_cavity_recon_cos),
    }
