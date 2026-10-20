import torch

def hippo_cfm_loss(
    unet,
    hippo_oracle,
    x_prompt,
    x_target,
    use_knowledge_attractor=True,
    valid_mask=None,
    pad_loss_weight: float = 0.05
):
    """
    Phase 4 Loss: Train Flow Match Unet using the absolute HiPPO Oracle.
    Automatically normalizes the target & prompt trajectories to unit variance (Mean=0, Std=1)
    via unet's built-in Adaptive Manifold Auto-Scaler so X_target and X_0 ~ N(0, 1) have 1:1 energy.
    
    x_prompt: (B, L_p, D) Raw continuous embeddings of the context (used as knowledge anchor)
    x_target: (B, L_t, D) Raw continuous embeddings of the target sentence
    use_knowledge_attractor: (bool) CFG/Curriculum toggle for orthogonal attraction.
    valid_mask: (B, L_t) Boolean mask (True for non-PAD tokens, False for PAD tokens).
    pad_loss_weight: Relative loss weight on PAD positions (default 0.05) to prevent gradient dilution.
    """
    B, L, D = x_target.shape
    device = x_target.device
    
    # 0. Adaptive Manifold Calibration & Normalization (maps raw valid Std~0.122 -> Std=1.0)
    with torch.no_grad():
        if unet.training:
            unet.calibrate_manifold(x_target, valid_mask=valid_mask)
        x_target_norm = unet.normalize(x_target)
        x_prompt_norm = unet.normalize(x_prompt)
        
        # 1. Oracle: Extract pure causal Toeplitz HiPPO potential curve (B, L, D) from normalized target
        h_hilbert = hippo_oracle(x_target_norm)
    
    # 2. Setup Symmetric Flow Matching Trajectory (Both X_1 and X_0 have Std = 1.0)
    t = torch.rand((B,), device=device, dtype=x_target_norm.dtype)
    t_expand = t[:, None, None] # (B, 1, 1)
    x_0 = torch.randn_like(x_target_norm)
    x_t = t_expand * x_target_norm + (1.0 - t_expand) * x_0
    v_target = x_target_norm - x_0
    
    # 3. Predict Velocity
    k_attractors = x_prompt_norm if use_knowledge_attractor else None
        
    v_pred = unet(x_t, t, h_potential=h_hilbert, k_attractors=k_attractors)
    
    # 4. Mask-Weighted Flow Matching Loss (prevents 90% <PAD> tails from diluting gradients)
    sq_err = (v_pred - v_target) ** 2
    if valid_mask is not None:
        mask_f = valid_mask.to(dtype=sq_err.dtype).unsqueeze(-1) # (B, L, 1)
        weights = mask_f + (1.0 - mask_f) * pad_loss_weight
        loss = (sq_err * weights).sum() / (weights.sum() * D).clamp(min=1.0)
    else:
        loss = torch.mean(sq_err)
    return loss
