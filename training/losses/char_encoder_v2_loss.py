from typing import Dict, List, Optional, Sequence, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_word_order_corrupted_tokens(
    token_inputs: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    pad_token_id: int = 0,
    special_token_ids: Tuple[int, ...] = (0, 1, 2, 3),
) -> torch.Tensor:
    """
    Constructs hard-negative word-order-corrupted token sequences by taking valid interior
    tokens (excluding BOS/EOS/PAD) and swapping non-overlapping multi-character spans while
    reversing sub-spans. Preserves the exact character multiset (Bag-of-Chars invariant)
    and identical attention_mask.
    """
    if attention_mask is None:
        attention_mask = (token_inputs != pad_token_id).float()

    corrupted = token_inputs.clone()
    B, L = token_inputs.shape
    special_set = set(int(s) for s in special_token_ids) | {int(pad_token_id)}

    for b in range(B):
        valid_pos = torch.nonzero(attention_mask[b] > 0, as_tuple=False).squeeze(-1)
        n_valid = valid_pos.numel()
        if n_valid <= 2:
            continue

        start_ptr = 0
        end_ptr = n_valid
        if n_valid > 3 and int(token_inputs[b, valid_pos[0]].item()) in special_set:
            start_ptr += 1
        if (end_ptr - start_ptr) > 2 and int(token_inputs[b, valid_pos[-1]].item()) in special_set:
            end_ptr -= 1

        interior_pos = valid_pos[start_ptr:end_ptr]
        m = interior_pos.numel()
        if m < 2:
            continue

        orig_tokens = token_inputs[b, interior_pos]
        if m >= 4:
            s0, s1, s2, s3 = torch.tensor_split(interior_pos, 4)
            perm_pos = torch.cat([s2.flip(0), s0, s3, s1.flip(0)], dim=0)
        else:
            perm_pos = interior_pos.flip(0)

        new_tokens = token_inputs[b, perm_pos]
        if torch.equal(new_tokens, orig_tokens):
            new_tokens = orig_tokens.flip(0)
            if torch.equal(new_tokens, orig_tokens) and m >= 3:
                new_tokens = torch.roll(orig_tokens, shifts=1, dims=0)

        corrupted[b, interior_pos] = new_tokens

    return corrupted


class OrderDirectionHeadCUDA(nn.Module):
    """
    Binary order discriminator classifying intact vs word-order-corrupted clause pairs:
      - Positive (label 1): `[z_pred, z_corrupt]`
      - Negative (label 0): `[z_corrupt, z_pred]`
    """

    def __init__(self, z_dim: int = 1024, hidden_dim: int = 256):
        super().__init__()
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(3 * z_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, z_first: torch.Tensor, z_second: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if z_second is None:
            if z_first.size(-1) != 2 * self.z_dim:
                raise ValueError(
                    f"Expected concatenated input of last dim {2 * self.z_dim}, got {z_first.size(-1)}"
                )
            z_first, z_second = z_first.chunk(2, dim=-1)

        za = F.normalize(z_first.float(), p=2, dim=-1)
        zb = F.normalize(z_second.float(), p=2, dim=-1)
        feat = torch.cat([za, zb, za - zb], dim=-1)
        return self.net(feat).squeeze(-1)

    def compute_loss_and_acc(
        self, z_pred: torch.Tensor, z_corrupt: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits_pos = self.forward(z_pred, z_corrupt)
        logits_neg = self.forward(z_corrupt, z_pred)

        loss_pos = F.binary_cross_entropy_with_logits(logits_pos, torch.ones_like(logits_pos))
        loss_neg = F.binary_cross_entropy_with_logits(logits_neg, torch.zeros_like(logits_neg))
        loss = 0.5 * (loss_pos + loss_neg)

        with torch.no_grad():
            acc = 0.5 * ((logits_pos > 0).float().mean() + (logits_neg < 0).float().mean())
        return loss, acc


class AsymmetricNextClauseHeadCUDA(nn.Module):
    """
    Asymmetric discourse progression predictor `W_fwd z_t` that retrieves `z_{t+1}`
    among all batch candidates and explicitly ranks forward progression `z_{t+1}`
    above reverse progression `z_{t-1}` / `W_fwd z_{t+1} -> z_t`.
    """

    def __init__(self, z_dim: int = 1024, temperature: float = 0.07):
        super().__init__()
        self.z_dim = z_dim
        self.temperature = float(temperature)
        self.w_fwd = nn.Linear(z_dim, z_dim, bias=False)
        nn.init.eye_(self.w_fwd.weight)

    def forward(self, z_t: torch.Tensor) -> torch.Tensor:
        z_norm = F.normalize(z_t.float(), p=2, dim=-1)
        return F.normalize(self.w_fwd(z_norm), p=2, dim=-1)

    def compute_loss(
        self,
        z_all: torch.Tensor,
        contiguous_pairs: Optional[torch.Tensor] = None,
        direction_margin: float = 0.15,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B = z_all.size(0)
        device = z_all.device
        if B < 2:
            zero = z_all.sum() * 0.0
            return zero, {
                "next_top1_acc": torch.tensor(1.0, device=device),
                "next_fwd_vs_rev_acc": torch.tensor(1.0, device=device),
            }

        if contiguous_pairs is None:
            idx_t = torch.arange(0, B - 1, device=device, dtype=torch.long)
            idx_next = idx_t + 1
        else:
            if contiguous_pairs.numel() == 0:
                zero = z_all.sum() * 0.0
                return zero, {
                    "next_top1_acc": torch.tensor(1.0, device=device),
                    "next_fwd_vs_rev_acc": torch.tensor(1.0, device=device),
                }
            idx_t = contiguous_pairs[:, 0].long().to(device)
            idx_next = contiguous_pairs[:, 1].long().to(device)

        q_t = self.forward(z_all[idx_t])
        k_all = F.normalize(z_all.float(), p=2, dim=-1)

        logits = torch.matmul(q_t, k_all.t()) / self.temperature
        # Mask out self-index idx_t so the query cannot trivially retrieve itself
        self_mask = torch.zeros_like(logits, dtype=torch.bool)
        self_mask.scatter_(1, idx_t.unsqueeze(1), True)
        logits = logits.masked_fill(self_mask, -1e4)

        loss_infonce = F.cross_entropy(logits, idx_next)

        # Asymmetric directional margin: forward (t -> t+1) > reverse (t+1 -> t) and (t -> t-1)
        k_next = k_all[idx_next]
        k_curr = k_all[idx_t]
        q_next = self.forward(z_all[idx_next])

        sim_fwd = (q_t * k_next).sum(dim=-1)
        sim_rev = (q_next * k_curr).sum(dim=-1)
        loss_dir = F.relu(direction_margin - (sim_fwd - sim_rev)).mean()

        has_prev = idx_t > 0
        if has_prev.any():
            idx_prev = (idx_t[has_prev] - 1).clamp(min=0)
            sim_prev = (q_t[has_prev] * k_all[idx_prev]).sum(dim=-1)
            loss_dir = loss_dir + F.relu(direction_margin - (sim_fwd[has_prev] - sim_prev)).mean()

        with torch.no_grad():
            top1_acc = (logits.argmax(dim=-1) == idx_next).float().mean()
            dir_acc = (sim_fwd > sim_rev).float().mean()

        return loss_infonce + loss_dir, {
            "next_top1_acc": top1_acc,
            "next_fwd_vs_rev_acc": dir_acc,
        }


class CharSubspaceAnchorHeadCUDA(nn.Module):
    """
    Lightweight cosine + linear prototype probe predicting current character ID `token_inputs`
    and neighbor bigram transitions from the 32-D character subspace `h_sub[b, t]` alone
    on valid (`attention_mask > 0`) positions.
    """

    def __init__(self, d_sub: int = 32, vocab_size: int = 32000, init_scale: float = 20.0):
        super().__init__()
        self.d_sub = d_sub
        self.vocab_size = vocab_size
        self.char_prototypes = nn.Parameter(torch.randn(vocab_size, d_sub) * (d_sub**-0.5))
        self.char_bias = nn.Parameter(torch.zeros(vocab_size))
        self.logit_scale = nn.Parameter(torch.tensor(float(init_scale)))
        self.bigram_proj = nn.Linear(d_sub, d_sub, bias=False)
        nn.init.orthogonal_(self.bigram_proj.weight)

    def forward(self, h_sub: torch.Tensor) -> torch.Tensor:
        h_norm = F.normalize(h_sub.float(), p=2, dim=-1)
        w_norm = F.normalize(self.char_prototypes.float(), p=2, dim=-1)
        scale = self.logit_scale.float().clamp(min=1.0, max=60.0)
        return F.linear(h_norm, w_norm * scale, self.char_bias.float())

    def compute_loss_and_acc(
        self,
        h_sub: torch.Tensor,
        token_inputs: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        valid_mask = attention_mask > 0
        if not valid_mask.any():
            zero = h_sub.sum() * 0.0
            return zero, torch.tensor(0.0, device=h_sub.device)

        h_valid = h_sub[valid_mask]
        tokens_valid = token_inputs[valid_mask].long().clamp(min=0, max=self.vocab_size - 1)

        logits = self.forward(h_valid)
        loss_id = F.cross_entropy(logits, tokens_valid)

        with torch.no_grad():
            top1_acc = (logits.argmax(dim=-1) == tokens_valid).float().mean()

        # Neighbor bigram transition prediction on valid adjacent token pairs
        adj_mask = valid_mask[:, :-1] & valid_mask[:, 1:]
        if adj_mask.any():
            h_curr = h_sub[:, :-1][adj_mask]
            next_tokens = token_inputs[:, 1:][adj_mask].long().clamp(min=0, max=self.vocab_size - 1)
            h_bigram = self.bigram_proj(h_curr.float())
            bigram_logits = self.forward(h_bigram)
            loss_bigram = F.cross_entropy(bigram_logits, next_tokens)
            loss_total = loss_id + 0.25 * loss_bigram
        else:
            loss_total = loss_id

        return loss_total, top1_acc


def compute_centered_cosine_matrix(x: torch.Tensor) -> torch.Tensor:
    """Computes batch-centered L2-normalized cosine similarity matrix S in R^{B x B}."""
    x_f = x.float()
    if x_f.size(0) > 1:
        x_f = x_f - x_f.mean(dim=0, keepdim=True)
    x_norm = F.normalize(x_f, p=2, dim=-1)
    return torch.matmul(x_norm, x_norm.t())


def compute_teacher_consensus_similarity(
    batch_embs: Optional[Sequence[torch.Tensor]],
    z_target_truth: torch.Tensor,
    weights: Optional[Sequence[float]] = None,
) -> torch.Tensor:
    """
    Builds the 5-teacher consensus centered cosine similarity matrix S_bar in R^{B x B}.
    Falls back to `z_target_truth` centered similarity if `batch_embs` is empty/None.
    """
    if not batch_embs:
        return compute_centered_cosine_matrix(z_target_truth.detach())

    sim_list: List[torch.Tensor] = []
    for emb in batch_embs:
        sim_list.append(compute_centered_cosine_matrix(emb.detach()))

    sim_stack = torch.stack(sim_list, dim=0)
    if weights is not None and len(weights) == len(sim_list):
        w = torch.tensor(weights, dtype=sim_stack.dtype, device=sim_stack.device)
        w = w / w.sum().clamp(min=1e-8)
        return (sim_stack * w.view(-1, 1, 1)).sum(dim=0)
    return sim_stack.mean(dim=0)


def compute_relational_distill_loss(
    z_pred: torch.Tensor,
    s_pool: torch.Tensor,
    s_teacher: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    Matches student batch-centered similarity matrices (both `z_pred` and `s_pool`)
    to the teacher consensus similarity matrix `s_teacher` via row-wise KL divergence
    (Soft InfoNCE) + Frobenius MSE.
    """
    B = z_pred.size(0)
    if B < 2:
        return z_pred.sum() * 0.0

    s_z = compute_centered_cosine_matrix(z_pred)
    s_p = compute_centered_cosine_matrix(s_pool)
    s_t = s_teacher.detach().float()

    loss_frob = 0.5 * (F.mse_loss(s_z, s_t) + F.mse_loss(s_p, s_t))

    off_diag = ~torch.eye(B, dtype=torch.bool, device=z_pred.device)
    s_t_off = s_t[off_diag].view(B, B - 1)
    s_z_off = s_z[off_diag].view(B, B - 1)
    s_p_off = s_p[off_diag].view(B, B - 1)

    p_t = F.softmax(s_t_off / temperature, dim=-1)
    kl_z = F.kl_div(F.log_softmax(s_z_off / temperature, dim=-1), p_t, reduction="batchmean") * (
        temperature**2
    )
    kl_p = F.kl_div(F.log_softmax(s_p_off / temperature, dim=-1), p_t, reduction="batchmean") * (
        temperature**2
    )
    return loss_frob + 0.5 * (kl_z + kl_p)


def compute_token_subspace_vicreg_loss(
    h_sub: torch.Tensor,
    attention_mask: torch.Tensor,
    ortho_loss: torch.Tensor,
    target_std: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Token-level VICReg Isotropic Equalization on valid token projections
    `H_valid = h_sub[attention_mask > 0]` in R^{N_valid x 32}:
      - Enforces unit std per channel (`std(H_valid[:, c]) == 1.0`) so no single axis
        can monopolize variance or collapse.
      - Enforces zero off-diagonal covariance across the 32 channels.
      - Adds Stiefel manifold orthogonality penalty `ortho_loss`.
    """
    valid_mask = attention_mask > 0
    h_valid = h_sub[valid_mask].float()
    N_valid, d_sub = h_valid.shape

    if N_valid <= 2:
        zero = h_sub.sum() * 0.0 + ortho_loss
        return zero, {
            "subspace_pc1_var_ratio": torch.tensor(1.0 / d_sub, device=h_sub.device),
            "subspace_participation_ratio": torch.tensor(float(d_sub), device=h_sub.device),
        }

    h_mean = h_valid.mean(dim=0, keepdim=True)
    h_centered = h_valid - h_mean

    # Keep residual mean near zero after char_token_mean subtraction
    mean_drift_loss = (h_mean.pow(2)).mean()

    var = h_centered.var(dim=0, unbiased=False)
    std = torch.sqrt(var + 1e-4)
    var_hinge = F.relu(target_std - std).mean()
    var_equal = ((std - target_std) ** 2).mean() + 0.25 * ((var - (target_std**2)) ** 2).mean()

    cov = torch.matmul(h_centered.t(), h_centered) / float(max(N_valid - 1, 1))
    off_diag_mask = ~torch.eye(d_sub, dtype=torch.bool, device=cov.device)
    cov_off_loss = (cov[off_diag_mask] ** 2).mean() * float(d_sub)

    loss_char_iso = var_hinge + var_equal + cov_off_loss + 0.1 * mean_drift_loss + ortho_loss

    with torch.no_grad():
        eigvals = torch.linalg.eigvalsh(cov).clamp(min=0.0)
        eig_sum = eigvals.sum().clamp(min=1e-12)
        pc1_ratio = eigvals[-1] / eig_sum
        pr = (eig_sum**2) / (eigvals.pow(2).sum().clamp(min=1e-12))

    return loss_char_iso, {
        "subspace_pc1_var_ratio": pc1_ratio,
        "subspace_participation_ratio": pr,
    }


def compute_phase1_v2_distill_loss(
    student: nn.Module,
    token_inputs: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    z_target_truth: torch.Tensor,
    batch_embs: Optional[Sequence[torch.Tensor]] = None,
    order_head: Optional[OrderDirectionHeadCUDA] = None,
    next_clause_head: Optional[AsymmetricNextClauseHeadCUDA] = None,
    char_anchor_head: Optional[CharSubspaceAnchorHeadCUDA] = None,
    contiguous_pairs: Optional[torch.Tensor] = None,
    order_margin: float = 0.35,
    w_distill: float = 2.0,
    w_rel: float = 2.0,
    w_order: float = 2.0,
    w_next: float = 0.5,
    w_char_iso: float = 2.0,
    w_char_id: float = 1.0,
    update_ema: bool = True,
    ema_momentum: float = 0.05,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Self-contained Phase 1 v2 distillation loss suite with zero WeakDecoder char-CE:
      1. `loss_distill`: Sum-over-dim normalized MSE + Cosine alignment to `z_target_truth`.
      2. `loss_rel`: Batch centered cosine similarity matrix distillation (KL + Frobenius MSE)
         against the 5-teacher consensus matrix S_bar.
      3. `loss_order`: Anti-BoW word-order hard-negative contrastive margin + `OrderDirectionHeadCUDA`.
      4. `loss_next`: Asymmetric next-clause contrastive retrieval + directional ranking.
      5. `loss_char_iso`: Valid-token 32-D VICReg variance equalization + covariance whitening +
         Stiefel manifold orthogonality penalty.
      6. `loss_char_id`: 32-D character identity & neighbor bigram anchor probe.
    """
    mask = student._resolve_attention_mask(token_inputs, attention_mask)
    h_tokens = student.encode_tokens(token_inputs, mask)

    if update_ema and hasattr(student, "update_char_token_mean"):
        student.update_char_token_mean(h_tokens, mask, momentum=ema_momentum)

    sum_mask = mask.sum(dim=-1, keepdim=True).clamp(min=1e-8).to(dtype=h_tokens.dtype)
    s_pool = h_tokens.sum(dim=1) / sum_mask
    z_pred = student.out_proj(s_pool)
    h_sub = student._project_char_subspace_from_tokens(h_tokens, mask, center=True)

    # 1. Semantic Target Distillation (loss_distill)
    z_t = z_target_truth.detach().float()
    z_p = z_pred.float()
    target_energy = z_t.pow(2).sum(dim=-1).mean().clamp(min=1.0)
    loss_mse_scaled = F.mse_loss(z_p, z_t, reduction="none").sum(dim=-1).mean() / target_energy
    cos_z_god = F.cosine_similarity(z_p, z_t, dim=-1)
    loss_cos = (1.0 - cos_z_god).mean()
    if s_pool.size(-1) == z_t.size(-1):
        loss_s_cos = (1.0 - F.cosine_similarity(s_pool.float(), z_t, dim=-1)).mean()
        loss_distill = loss_mse_scaled + loss_cos + 0.25 * loss_s_cos
    else:
        loss_distill = loss_mse_scaled + loss_cos

    # 2. Relational Topology Distillation (loss_rel)
    s_teacher = compute_teacher_consensus_similarity(batch_embs, z_t)
    loss_rel = compute_relational_distill_loss(z_p, s_pool.float(), s_teacher)

    # 3. Anti-BoW Word-Order Hard-Negative Contrastive Loss (loss_order)
    corrupted_tokens = make_word_order_corrupted_tokens(
        token_inputs, mask, pad_token_id=getattr(student, "pad_token_id", 0)
    )
    dual_corrupt = student.encode_dual(corrupted_tokens, mask)
    z_corrupt = dual_corrupt["z_god"].float()
    s_corrupt = dual_corrupt["s_pool"].float()

    cos_order = F.cosine_similarity(z_p, z_corrupt, dim=-1)
    cos_order_detach = F.cosine_similarity(z_p.detach(), z_corrupt, dim=-1)
    cos_s_order = F.cosine_similarity(s_pool.float(), s_corrupt, dim=-1)
    cos_target_corrupt = F.cosine_similarity(z_t, z_corrupt, dim=-1)

    margin_viol = (
        F.relu(cos_order - order_margin)
        + F.relu(cos_order_detach - order_margin)
        + 0.5 * F.relu(cos_s_order - order_margin)
        + 0.5 * F.relu(cos_target_corrupt - order_margin)
    )
    loss_order_margin = (margin_viol + margin_viol.pow(2)).mean()

    if order_head is not None:
        loss_order_disc, order_discrim_acc = order_head.compute_loss_and_acc(z_p, z_corrupt)
    else:
        loss_order_disc = z_p.sum() * 0.0
        order_discrim_acc = (cos_z_god > cos_target_corrupt).float().mean()

    loss_order = loss_order_margin + loss_order_disc

    # 4. Asymmetric Discourse Progression Contrastive Loss (loss_next)
    if next_clause_head is not None:
        loss_next, next_metrics = next_clause_head.compute_loss(
            z_p, contiguous_pairs=contiguous_pairs
        )
    else:
        loss_next = z_p.sum() * 0.0
        next_metrics = {
            "next_top1_acc": torch.tensor(0.0, device=z_p.device),
            "next_fwd_vs_rev_acc": torch.tensor(0.0, device=z_p.device),
        }

    # 5. Token-Level Isotropic 32-D Subspace Loss (loss_char_iso)
    ortho_loss = student.orthogonal_subspace_loss()
    loss_char_iso, iso_metrics = compute_token_subspace_vicreg_loss(
        h_sub, mask, ortho_loss=ortho_loss
    )

    # 6. Token Anchor Probe Loss (loss_char_id)
    if char_anchor_head is not None:
        loss_char_id, char_top1_acc = char_anchor_head.compute_loss_and_acc(
            h_sub, token_inputs, mask
        )
    else:
        loss_char_id = h_sub.sum() * 0.0
        char_top1_acc = torch.tensor(0.0, device=h_sub.device)

    total_loss = (
        w_distill * loss_distill
        + w_rel * loss_rel
        + w_order * loss_order
        + w_next * loss_next
        + w_char_iso * loss_char_iso
        + w_char_id * loss_char_id
    )

    metrics = {
        "loss_total": total_loss.detach(),
        "loss_distill": loss_distill.detach(),
        "loss_rel": loss_rel.detach(),
        "loss_order": loss_order.detach(),
        "loss_next": loss_next.detach(),
        "loss_char_iso": loss_char_iso.detach(),
        "loss_char_id": loss_char_id.detach(),
        "loss_ortho": ortho_loss.detach(),
        "z_god_cos": cos_z_god.detach().mean(),
        "shuffled_order_cos": cos_order.detach().mean(),
        "order_discrim_acc": order_discrim_acc.detach(),
        "subspace_pc1_var_ratio": iso_metrics["subspace_pc1_var_ratio"].detach(),
        "subspace_participation_ratio": iso_metrics["subspace_participation_ratio"].detach(),
        "subspace_char_top1_acc": char_top1_acc.detach(),
        "next_top1_acc": next_metrics["next_top1_acc"].detach(),
        "next_fwd_vs_rev_acc": next_metrics["next_fwd_vs_rev_acc"].detach(),
    }
    return total_loss, metrics


def _spearman_rank_corr(x: torch.Tensor, y: torch.Tensor) -> float:
    x_flat = x.detach().float().reshape(-1)
    y_flat = y.detach().float().reshape(-1)
    if x_flat.numel() < 2:
        return 1.0
    rx = torch.argsort(torch.argsort(x_flat)).float()
    ry = torch.argsort(torch.argsort(y_flat)).float()
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = (rx.norm() * ry.norm()).clamp(min=1e-12)
    return float((torch.dot(rx, ry) / denom).item())


@torch.no_grad()
def evaluate_phase1_v2_health(
    student: nn.Module,
    token_inputs: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    z_target_truth: torch.Tensor,
    batch_embs: Optional[Sequence[torch.Tensor]] = None,
    order_head: Optional[OrderDirectionHeadCUDA] = None,
    char_anchor_head: Optional[CharSubspaceAnchorHeadCUDA] = None,
) -> Dict[str, float]:
    """
    Quantitative diagnostic probe for Phase 1 v2 Dual-Head Masked Character Encoder.
    Computes and returns:
      - `z_god_cos`
      - `relational_spearman`
      - `shuffled_order_cos`
      - `order_discrim_acc`
      - `subspace_pc1_var_ratio`
      - `subspace_participation_ratio`
      - `subspace_char_top1_acc`
      - `masked_vs_auto_mask_max_diff`
    """
    was_training = student.training
    student.eval()

    explicit_mask = student._resolve_attention_mask(token_inputs, attention_mask)

    # Verify auto-masking invariance: encode_tokens(x, None) vs encode_tokens(x, explicit_mask)
    h_auto = student.encode_tokens(token_inputs, None)
    h_explicit = student.encode_tokens(token_inputs, explicit_mask)
    dual_auto = student.encode_dual(token_inputs, None)
    dual_explicit = student.encode_dual(token_inputs, explicit_mask)

    masked_vs_auto_mask_max_diff = float(
        max(
            (h_auto - h_explicit).abs().max().item(),
            (dual_auto["z_god"] - dual_explicit["z_god"]).abs().max().item(),
            (dual_auto["h_sub"] - dual_explicit["h_sub"]).abs().max().item(),
        )
    )

    z_pred = dual_explicit["z_god"].float()
    h_sub = dual_explicit["h_sub"].float()
    z_t = z_target_truth.detach().float()

    # 1. Teacher alignment cosine
    z_god_cos = float(F.cosine_similarity(z_pred, z_t, dim=-1).mean().item())

    # 2. Relational Spearman correlation with 5-teacher consensus matrix
    B = z_pred.size(0)
    if B >= 3:
        s_stu = compute_centered_cosine_matrix(z_pred)
        s_tea = compute_teacher_consensus_similarity(batch_embs, z_t)
        off_diag = ~torch.eye(B, dtype=torch.bool, device=z_pred.device)
        relational_spearman = _spearman_rank_corr(s_stu[off_diag], s_tea[off_diag])
    else:
        relational_spearman = 1.0

    # 3. Shuffled word-order self-cosine & order discrimination accuracy
    corrupted_tokens = make_word_order_corrupted_tokens(
        token_inputs, explicit_mask, pad_token_id=getattr(student, "pad_token_id", 0)
    )
    dual_corrupt = student.encode_dual(corrupted_tokens, explicit_mask)
    z_corrupt = dual_corrupt["z_god"].float()

    shuffled_order_cos = float(F.cosine_similarity(z_pred, z_corrupt, dim=-1).mean().item())
    if order_head is not None:
        _, acc_t = order_head.compute_loss_and_acc(z_pred, z_corrupt)
        order_discrim_acc = float(acc_t.item())
    else:
        cos_orig = F.cosine_similarity(z_pred, z_t, dim=-1)
        cos_corr = F.cosine_similarity(z_corrupt, z_t, dim=-1)
        order_discrim_acc = float((cos_orig > cos_corr).float().mean().item())

    # 4. 32-D Subspace PC1 variance ratio & participation ratio on valid tokens
    valid_mask = explicit_mask > 0
    h_valid = h_sub[valid_mask]
    n_valid, d_sub = h_valid.shape
    if n_valid > 2:
        h_centered = h_valid - h_valid.mean(dim=0, keepdim=True)
        cov = torch.matmul(h_centered.t(), h_centered) / float(max(n_valid - 1, 1))
        eigvals = torch.linalg.eigvalsh(cov).clamp(min=0.0)
        eig_sum = eigvals.sum().clamp(min=1e-12)
        subspace_pc1_var_ratio = float((eigvals[-1] / eig_sum).item())
        subspace_participation_ratio = float(
            ((eig_sum**2) / eigvals.pow(2).sum().clamp(min=1e-12)).item()
        )
    else:
        subspace_pc1_var_ratio = 1.0 / float(d_sub)
        subspace_participation_ratio = float(d_sub)

    # 5. Character identity top-1 accuracy from 32-D h_sub alone
    if char_anchor_head is not None and n_valid > 0:
        _, acc_char = char_anchor_head.compute_loss_and_acc(h_sub, token_inputs, explicit_mask)
        subspace_char_top1_acc = float(acc_char.item())
    elif n_valid > 0:
        # Nearest class-centroid cosine probe on valid 32-D tokens
        tokens_valid = token_inputs[valid_mask].long()
        unique_ids, inv_idx = torch.unique(tokens_valid, return_inverse=True)
        h_norm = F.normalize(h_valid, p=2, dim=-1)
        centroids = torch.zeros(unique_ids.numel(), d_sub, device=h_valid.device, dtype=h_valid.dtype)
        centroids.index_add_(0, inv_idx, h_norm)
        centroids = F.normalize(centroids, p=2, dim=-1)
        sims = torch.matmul(h_norm, centroids.t())
        preds = unique_ids[sims.argmax(dim=-1)]
        subspace_char_top1_acc = float((preds == tokens_valid).float().mean().item())
    else:
        subspace_char_top1_acc = 0.0

    if was_training:
        student.train()

    return {
        "z_god_cos": z_god_cos,
        "relational_spearman": relational_spearman,
        "shuffled_order_cos": shuffled_order_cos,
        "order_discrim_acc": order_discrim_acc,
        "subspace_pc1_var_ratio": subspace_pc1_var_ratio,
        "subspace_participation_ratio": subspace_participation_ratio,
        "subspace_char_top1_acc": subspace_char_top1_acc,
        "masked_vs_auto_mask_max_diff": masked_vs_auto_mask_max_diff,
    }
