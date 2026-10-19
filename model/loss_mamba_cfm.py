import torch

def mamba_cfm_loss(mamba, hippo_oracle, x_context, x_target):
    """
    Phase 5 Loss: Train Mamba to predict the exact Hilbert space coefficients.
    Flow Match (Unet) is NOT needed here. Mamba is trained to align with the Oracle.
    
    x_context: (B, L_c, D) Continuous embeddings of Context + Retrieved Knowledge
    x_target: (B, L_t, D) Continuous embeddings of the target sentence
    """
    # 1. Oracle: What should the potential energy be for this target?
    with torch.no_grad():
        h_target = hippo_oracle(x_target, return_sequence=False) # (B, D)
        
    # 2. Mamba Prediction: What does Mamba think the potential energy will be?
    # Based strictly on the context, without seeing the future.
    h_pred = mamba(x_context) # (B, D)
    
    # 3. Loss: Force Mamba's predicted state to exactly match the target Legendre coefficients
    loss = torch.nn.functional.mse_loss(h_pred, h_target)
    return loss
