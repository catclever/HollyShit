import mlx.core as mx
import math

def ot_cfm_loss(model, token_ids: mx.array, z_target: mx.array, mask: mx.array = None,
                x1_weight: float = 0.0, snap_ce_weight: float = 0.0,
                t_power: float = 1.0):
    """
    Optimal Transport Conditional Flow Matching (OT-CFM) Loss with auxiliary objectives.
    
    Args:
        model: FlowDecoder instance
        token_ids: (B, L) ground truth token IDs
        z_target: (B, z_dim) semantic anchor from GodEncoder
        mask: (B, L) padding mask (1=valid, 0=pad)
        x1_weight: Weight for x₁-consistency loss (recommended 0.5~1.0)
        snap_ce_weight: Weight for snap cross-entropy loss (recommended 0.1~0.5)
        t_power: Power for high-t sampling bias. 1.0 = uniform, <1.0 = bias towards t≈1
                 (recommended 0.5 for sqrt bias)
    
    Returns:
        total_loss, loss_dict  (loss_dict for logging)
    """
    B, L = token_ids.shape
    
    # 1. Ground Truth Destination (The physical coordinates of the sentence)
    x_1 = model.embed_text(token_ids)  # (B, L, d_model)
    
    # 2. Primordial Chaos (Standard Normal Distribution)
    x_0 = mx.random.normal(shape=x_1.shape)  # (B, L, d_model)
    
    # 3. Time Sampling with optional high-t bias
    # t_power < 1.0 → t = u^power, biases towards t≈1 (last-mile precision)
    # t_power = 1.0 → uniform (original behavior)
    # t_power = 0.5 → sqrt bias, ~70% of samples have t > 0.5
    u = mx.random.uniform(shape=(B, 1))
    t = mx.power(u, t_power)
    
    # 4. Interpolate the exact intermediate state
    t_expand = mx.expand_dims(t, 2)  # (B, 1, 1)
    x_t = (1.0 - t_expand) * x_0 + t_expand * x_1
    
    # 5. Optimal Transport Straight-Line Velocity (ground truth)
    v_true = x_1 - x_0
    
    # 6. Model Prediction
    v_pred = model(x_t, t, z_target, mask=mask)
    
    # ====== Loss 1: Velocity MSE (core OT-CFM objective) ======
    mse = mx.square(v_pred - v_true).mean(axis=-1)  # (B, L)
    
    if mask is not None:
        num_valid = mx.maximum(mask.sum(), 1.0)
        L_velocity = (mse * mask).sum() / num_valid
    else:
        num_valid = B * L
        L_velocity = mse.mean()
    
    total_loss = L_velocity
    
    # ====== Loss 2: x₁-Consistency (does the predicted velocity point to the right destination?) ======
    L_x1 = mx.array(0.0)
    if x1_weight > 0:
        # From v_pred, reverse-engineer where the model thinks x₁ should be
        # x₁_pred = x_t + (1 - t) * v_pred
        x1_pred = x_t + (1.0 - t_expand) * v_pred
        x1_mse = mx.square(x1_pred - x_1).mean(axis=-1)  # (B, L)
        
        if mask is not None:
            L_x1 = (x1_mse * mask).sum() / num_valid
        else:
            L_x1 = x1_mse.mean()
        
        total_loss = total_loss + x1_weight * L_x1
    
    # ====== Loss 3: Snap Cross-Entropy (can the prediction snap to the right character?) ======
    L_snap = mx.array(0.0)
    if snap_ce_weight > 0:
        # Predict x₁ from current state
        x1_pred = x_t + (1.0 - t_expand) * v_pred  # (B, L, d_model)
        
        # Get the embedding bank (unscaled to align with continuous space optimization)
        emb_bank = model.char_embedding.weight  # (V, d_model)
        
        # Compute logits via dot product similarity, scaled by temperature 1/sqrt(d_model)
        # (B, L, d_model) @ (d_model, V) → (B, L, V)
        logits = mx.matmul(x1_pred, emb_bank.T) / math.sqrt(model.d_model)
        
        # Cross-entropy against ground truth tokens
        # Flatten for cross_entropy: (B*L, V) vs (B*L,)
        logits_flat = logits.reshape(-1, logits.shape[-1])
        targets_flat = token_ids.reshape(-1)
        
        # Manual cross-entropy: -log_softmax(logits)[target]
        log_probs = logits_flat - mx.logsumexp(logits_flat, axis=-1, keepdims=True)  # (B*L, V)
        # Gather the log-prob of the correct token at each position
        ce = -log_probs[mx.arange(targets_flat.shape[0]), targets_flat]  # (B*L,)
        
        if mask is not None:
            mask_flat = mask.reshape(-1)
            L_snap = (ce * mask_flat).sum() / num_valid
        else:
            L_snap = ce.mean()
        
        total_loss = total_loss + snap_ce_weight * L_snap
    
    return total_loss
