import torch

def ot_cfm_loss_legacy(unet, mamba, x_in, x_tgt):
    """
    Legacy joint-training loss (Mamba + Flow Match).
    Restored for backward compatibility with the old script.
    """
    B, L, D = x_tgt.shape
    device = x_tgt.device
    
    # Mamba computes z_macro directly from input
    z_macro = mamba(x_in) # (B, D)
    
    # Sample t ~ U(0, 1)
    t = torch.rand((B,), device=device)
    t_expand = t[:, None, None] # (B, 1, 1)
    
    # Sample base noise x_0 ~ N(0, I)
    x_0 = torch.randn_like(x_tgt)
    
    # OT Interpolation
    x_t = t_expand * x_tgt + (1 - t_expand) * x_0
    
    # Target Velocity Field
    v_target = x_tgt - x_0
    
    # Predict Velocity Field
    # Notice we map z_macro to the new h_potential argument, and disable k_attractors
    v_pred = unet(x_t, t, h_potential=z_macro, k_attractors=None)
    
    # MSE Loss
    loss = torch.mean((v_pred - v_target)**2)
    return loss
