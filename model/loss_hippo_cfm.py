import torch

def hippo_cfm_loss(unet, hippo_oracle, x_prompt, x_target, use_knowledge_attractor=True):
    """
    Phase 4 Loss: Train Flow Match Unet using the absolute HiPPO Oracle.
    This trains the Unet to perfectly draw curves from valid Legendre coefficients.
    
    x_prompt: (B, L_p, D) Continuous embeddings of the context (used as knowledge anchor)
    x_target: (B, L_t, D) Continuous embeddings of the target sentence
    use_knowledge_attractor: (bool) CFG/Curriculum toggle for orthogonal attraction.
    """
    B, L, D = x_target.shape
    device = x_target.device
    
    # 1. Oracle: Extract pure Legendre polynomial coefficients from the target curve
    with torch.no_grad():
        h_hilbert = hippo_oracle(x_target) # (B, D)
    
    # 2. Setup Flow Matching Trajectory
    t = torch.rand((B,), device=device)
    t_expand = t[:, None, None] # (B, 1, 1)
    x_0 = torch.randn_like(x_target)
    x_t = t_expand * x_target + (1 - t_expand) * x_0
    v_target = x_target - x_0
    
    # 3. Predict Velocity
    # Curriculum Learning: randomly drop out the attractor to force grammar mastery
    if use_knowledge_attractor and torch.rand(1).item() > 0.5:
        k_attractors = x_prompt
    else:
        k_attractors = None
        
    v_pred = unet(x_t, t, h_potential=h_hilbert, k_attractors=k_attractors)
    
    loss = torch.mean((v_pred - v_target)**2)
    return loss
