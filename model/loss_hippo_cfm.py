import torch
import torch.nn.functional as F


def hippo_cfm_loss(
    unet,
    hippo_oracle,
    x_prompt,
    x_target,
    use_knowledge_attractor=True,
    valid_mask=None,
    prompt_mask=None,
    pad_loss_weight: float = 0.05,
    return_metrics: bool = False,
    translator=None,
    w_target=None,
    cavity_prob: float = 0.70,
    ce_weight: float = 0.35,
    corrupt_z_sent: bool = True,
):
    """
    Phase 4 v2 Loss (#021 / #050): Train No-Skip Cavity-Tolerant LeftBrainCrystallizer (`FlowMatching1DUnet`)
    using the 2D Sentence Vector Bottleneck (`z_sent` of shape `(B, D)`) from `HiPPOOracleCUDA` +
    Orthogonal Knowledge Attractors `k_attractors` + 3-Mode Cavity-Immunization Curriculum:
      - 30% Exact Mode: clean `z_sent` (K=32);
      - 35% High-Band Spectral Coarsening: keep low bands k=0..5, attenuate k=6..31 by u in [0.0, 0.35] + noise;
      - 35% Multi-Clause Cavity Mix: alpha * z_sent + (1 - alpha) * z_distractor_mean (alpha in [0.25, 0.70]).
    Supervises the implied endpoint `x_1_hat` against clean `x_target_norm` with `loss_vel`, `loss_cos`,
    32-D subspace `loss_slot32` (scaled by d_sub/D ~ 0.04 + slot transition diff loss), and frozen-Translator
    token cross-entropy `loss_ce`.
    """
    B, L, D = x_target.shape
    device = x_target.device

    if translator is not None and hasattr(unet, "bind_translator_codebook"):
        if getattr(unet, "_char_codebook_norm", None) is None:
            unet.bind_translator_codebook(translator)

    # 0. Adaptive Manifold Calibration & Normalization (maps raw valid Std~0.122 -> Std=1.0)
    with torch.no_grad():
        if unet.training:
            unet.calibrate_manifold(x_target, valid_mask=valid_mask)
        x_target_norm = unet.normalize(x_target)
        x_prompt_norm = unet.normalize(x_prompt)

        # 1. Oracle: Extract strictly 2D Sentence Vector bottleneck z_sent_clean of shape (B, D)
        z_sent_clean = hippo_oracle(
            x_target_norm,
            return_sequence=False,
            valid_mask=valid_mask,
            subspace_proj=unet.subspace_proj,
        )
        assert z_sent_clean.dim() == 2 and z_sent_clean.shape == (B, D), (
            f"Expected 2D sentence vector z_sent of shape {(B, D)}, got {z_sent_clean.shape}"
        )

        if unet.training and hasattr(unet, "calibrate_cavity_stats"):
            unet.calibrate_cavity_stats(z_sent_clean, seq_len=L)

        # 1b. 3-Mode Cavity-Immunization Curriculum on z_sent (#050)
        z_sent = z_sent_clean.clone()
        if unet.training and corrupt_z_sent and float(cavity_prob) > 0.0:
            p_cav = min(max(float(cavity_prob), 0.0), 1.0)
            p_spec = 0.5 * p_cav  # 35% Spectral Coarsening + 35% Multi-Clause Cavity Mix when cavity_prob=0.70
            u_mode = torch.rand((B,), device=device)
            spec_mask = (u_mode >= (1.0 - p_cav)) & (u_mode < (1.0 - p_spec))
            cav_mask = u_mode >= (1.0 - p_spec)

            # Guarantee both corruption modes are represented when batch size B >= 3
            if B >= 3 and p_cav >= 0.50:
                if not spec_mask.any():
                    spec_mask[0] = True
                    cav_mask[0] = False
                if not cav_mask.any():
                    cav_mask[-1] = True
                    spec_mask[-1] = False

            K = getattr(unet, "num_subbands", 32)
            d_sub = D // K

            # Mode 2 (35%): High-Band Spectral Coarsening (keep k=0..5, attenuate k=6..31 by u in [0, 0.35] + noise)
            if spec_mask.any():
                n_s = int(spec_mask.sum().item())
                z_sub_s = z_sent_clean[spec_mask].view(n_s, K, d_sub).clone()
                u_att = torch.empty((n_s, 1, 1), device=device, dtype=z_sub_s.dtype).uniform_(0.0, 0.35)
                # Zero out high bands completely on half of spectral-coarsened samples to cover exact k=0..5 truncation
                u_att = torch.where(
                    torch.rand((n_s, 1, 1), device=device) < 0.35,
                    torch.zeros_like(u_att),
                    u_att,
                )
                sig_n = torch.empty((n_s, 1, 1), device=device, dtype=z_sub_s.dtype).uniform_(0.0, 0.08) * u_att
                z_sub_s[:, 6:, :] = u_att * z_sub_s[:, 6:, :] + sig_n * torch.randn_like(z_sub_s[:, 6:, :])
                z_spec = z_sub_s.reshape(n_s, D)
                z_spec = z_spec / z_spec.float().std(dim=-1, keepdim=True).clamp(min=1e-5).to(z_spec.dtype)
                z_sent[spec_mask] = z_spec

            # Mode 3 (35%): Multi-Clause Cavity Mix (alpha * z_sent + (1 - alpha) * mean of 4 random distractor clauses)
            if cav_mask.any():
                n_c = int(cav_mask.sum().item())
                if B > 1:
                    dist_idx = torch.randint(0, B, (n_c, 4), device=device)
                    z_dist_mean = z_sent_clean[dist_idx].mean(dim=1)
                else:
                    z_dist_mean = torch.randn_like(z_sent_clean[cav_mask])
                alpha = torch.empty((n_c, 1), device=device, dtype=z_sent.dtype).uniform_(0.25, 0.70)
                z_mix = alpha * z_sent_clean[cav_mask] + (1.0 - alpha) * z_dist_mean
                z_mix = z_mix / z_mix.float().std(dim=-1, keepdim=True).clamp(min=1e-5).to(z_mix.dtype)
                z_sent[cav_mask] = z_mix

    # 2. Setup Symmetric Flow Matching Trajectory (Both X_1 and X_0 have Std = 1.0)
    # Sample t ~ Uniform(0, 0.95) so 1.0 - t >= 0.05 holds identically everywhere
    t = torch.rand((B,), device=device, dtype=x_target_norm.dtype) * 0.95
    t_expand = t[:, None, None]  # (B, 1, 1)
    x_0 = torch.randn_like(x_target_norm)
    x_t = t_expand * x_target_norm + (1.0 - t_expand) * x_0
    v_target = x_target_norm - x_0

    # 3. Predict Velocity conditioned on 2D Sentence Vector z_sent + k_attractors with prompt_mask
    k_attractors = x_prompt_norm if use_knowledge_attractor else None
    k_mask = prompt_mask if use_knowledge_attractor else None
    v_pred = unet(x_t, t, h_potential=z_sent, k_attractors=k_attractors, k_mask=k_mask)

    # 4. Mask-Weighted Flow Matching Velocity MSE Loss
    denom = (1.0 - t_expand).clamp(min=0.05)
    sq_err = (denom * (v_pred - v_target)) ** 2
    if valid_mask is not None:
        mask_f = valid_mask.to(dtype=sq_err.dtype).unsqueeze(-1)  # (B, L, 1)
        weights = mask_f + (1.0 - mask_f) * pad_loss_weight
        loss_vel = (sq_err * weights).sum() / (weights.sum() * D).clamp(min=1.0)
        loss_valid_raw = (sq_err * mask_f).sum() / (mask_f.sum() * D).clamp(min=1.0)
    else:
        loss_vel = torch.mean(sq_err)
        loss_valid_raw = loss_vel

    # 5. Implied Endpoint Prediction & Valid-Token Cosine Direction Loss
    x_1_hat = x_t + denom * v_pred
    cos_sim = F.cosine_similarity(x_1_hat.float(), x_target_norm.float(), dim=-1, eps=1e-6)
    cos_err = 1.0 - cos_sim
    if valid_mask is not None and valid_mask.any():
        mask_2d = valid_mask.to(dtype=cos_err.dtype)
        loss_cos = 0.1 * (cos_err * mask_2d).sum() / mask_2d.sum().clamp(min=1.0)
    else:
        loss_cos = 0.1 * cos_err.mean()

    # 6. Dedicated 32-D Principal Subspace Slot Consistency & First-Order Transition Loss (#049 / #050)
    P = unet.subspace_proj.to(device=x_1_hat.device, dtype=x_1_hat.dtype)
    x1_32 = torch.matmul(x_1_hat, P).float()
    xtgt_32 = torch.matmul(x_target_norm, P).float()
    if valid_mask is not None and valid_mask.any():
        vm_bool = valid_mask.bool()
        mse_32 = F.mse_loss(x1_32[vm_bool], xtgt_32[vm_bool])
        cos_32 = (1.0 - F.cosine_similarity(x1_32[vm_bool], xtgt_32[vm_bool], dim=-1, eps=1e-6)).mean()
    else:
        mse_32 = F.mse_loss(x1_32, xtgt_32)
        cos_32 = (1.0 - F.cosine_similarity(x1_32, xtgt_32, dim=-1, eps=1e-6)).mean()

    if L > 1:
        dx1_32 = x1_32[:, 1:, :] - x1_32[:, :-1, :]
        dxtgt_32 = xtgt_32[:, 1:, :] - xtgt_32[:, :-1, :]
        if valid_mask is not None and (valid_mask[:, 1:] & valid_mask[:, :-1]).any():
            pair_m = (valid_mask[:, 1:] & valid_mask[:, :-1]).bool()
            slot_diff_loss = F.mse_loss(dx1_32[pair_m], dxtgt_32[pair_m])
        else:
            slot_diff_loss = F.mse_loss(dx1_32, dxtgt_32)
    else:
        slot_diff_loss = torch.zeros_like(mse_32)

    loss_slot32 = 0.04 * (mse_32 + 0.35 * slot_diff_loss) + 0.05 * cos_32

    # 7. Frozen-Translator Token Cross-Entropy Supervision on Denormalized Endpoint (#050)
    if translator is not None and w_target is not None and float(ce_weight) > 0.0:
        x_1_denorm = unet.denormalize(x_1_hat)
        logits = translator(x_1_denorm.float())
        if valid_mask is not None and valid_mask.any():
            ce_raw = F.cross_entropy(logits[valid_mask.bool()], w_target[valid_mask.bool()].long())
        else:
            ce_raw = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), w_target.reshape(-1).long())
        loss_ce = float(ce_weight) * ce_raw
    else:
        loss_ce = torch.zeros((), device=device, dtype=loss_vel.dtype)

    loss = (
        loss_vel
        + loss_cos.to(dtype=loss_vel.dtype)
        + loss_slot32.to(dtype=loss_vel.dtype)
        + loss_ce.to(dtype=loss_vel.dtype)
    )

    if return_metrics:
        metrics = {
            "loss_vel": loss_vel.detach(),
            "loss_valid_raw": loss_valid_raw.detach(),
            "loss_cos": loss_cos.detach(),
            "loss_slot32": loss_slot32.detach(),
            "loss_ce": loss_ce.detach(),
            "z_sent_std": z_sent.float().std().detach(),
        }
        return loss, metrics

    return loss
