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
):
    """
    Phase 4 Loss: Train Flow Match Unet (Right Brain) using the 2D Sentence Vector Bottleneck
    (`z_sent` of shape `(B, D)`) from HiPPOOracleCUDA + Orthogonal Knowledge Attractors `k_attractors`.

    x_prompt: (B, L_p, D) Raw continuous embeddings of the context + knowledge (used as k_attractors)
    x_target: (B, L_t, D) Raw continuous embeddings of the single-clause target sentence
    use_knowledge_attractor: (bool) Toggle for orthogonal attraction from k_attractors
    valid_mask: (B, L_t) Boolean mask (True for non-PAD tokens, False for PAD tokens)
    prompt_mask: (B, L_p) Boolean mask (True for non-PAD tokens in prompt, False for PAD tokens)
    pad_loss_weight: Relative loss weight on PAD positions (default 0.05) to prevent gradient dilution
    return_metrics: (bool) If True, returns (loss, metrics_dict)
    """
    B, L, D = x_target.shape
    device = x_target.device

    # 0. Adaptive Manifold Calibration & Normalization (maps raw valid Std~0.122 -> Std=1.0)
    with torch.no_grad():
        if unet.training:
            unet.calibrate_manifold(x_target, valid_mask=valid_mask)
        x_target_norm = unet.normalize(x_target)
        x_prompt_norm = unet.normalize(x_prompt)

        # 1. Oracle: Extract strictly 2D Sentence Vector bottleneck z_sent of shape (B, D)
        z_sent = hippo_oracle(x_target_norm, return_sequence=False, valid_mask=valid_mask)
        assert z_sent.dim() == 2 and z_sent.shape == (B, D), (
            f"Expected 2D sentence vector z_sent of shape {(B, D)}, got {z_sent.shape}"
        )

    # 2. Setup Symmetric Flow Matching Trajectory (Both X_1 and X_0 have Std = 1.0)
    # Sample t ~ Uniform(0, 1) spanning the full continuous unit interval
    t = torch.rand((B,), device=device, dtype=x_target_norm.dtype)
    t_expand = t[:, None, None]  # (B, 1, 1)
    x_0 = torch.randn_like(x_target_norm)
    x_t = t_expand * x_target_norm + (1.0 - t_expand) * x_0
    v_target = x_target_norm - x_0

    # 3. Predict Velocity conditioned on 2D Sentence Vector z_sent + k_attractors with prompt_mask
    k_attractors = x_prompt_norm if use_knowledge_attractor else None
    k_mask = prompt_mask if use_knowledge_attractor else None
    v_pred = unet(x_t, t, h_potential=z_sent, k_attractors=k_attractors, k_mask=k_mask)

    # 4. Mask-Weighted Flow Matching Velocity MSE Loss
    sq_err = (v_pred - v_target) ** 2
    if valid_mask is not None:
        mask_f = valid_mask.to(dtype=sq_err.dtype).unsqueeze(-1)  # (B, L, 1)
        weights = mask_f + (1.0 - mask_f) * pad_loss_weight
        loss_vel = (sq_err * weights).sum() / (weights.sum() * D).clamp(min=1.0)
        loss_valid_raw = (sq_err * mask_f).sum() / (mask_f.sum() * D).clamp(min=1.0)
    else:
        loss_vel = torch.mean(sq_err)
        loss_valid_raw = loss_vel

    # 5. Implied Endpoint Prediction & Valid-Token Cosine Direction Loss
    x_1_hat = x_t + (1.0 - t_expand) * v_pred
    cos_sim = F.cosine_similarity(x_1_hat.float(), x_target_norm.float(), dim=-1, eps=1e-6)
    cos_err = 1.0 - cos_sim
    if valid_mask is not None and valid_mask.any():
        mask_2d = valid_mask.to(dtype=cos_err.dtype)
        loss_cos = 0.1 * (cos_err * mask_2d).sum() / mask_2d.sum().clamp(min=1.0)
    else:
        loss_cos = 0.1 * cos_err.mean()

    loss = loss_vel + loss_cos.to(dtype=loss_vel.dtype)

    if return_metrics:
        metrics = {
            "loss_vel": loss_vel.detach(),
            "loss_valid_raw": loss_valid_raw.detach(),
            "loss_cos": loss_cos.detach(),
            "z_sent_std": z_sent.float().std().detach(),
        }
        return loss, metrics

    return loss
