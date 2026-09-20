import torch
import torch.nn as nn

class DifferentialPlannerCUDA(nn.Module):
    """
    Autoregressive Differential Planner (PyTorch/CUDA Version).
    Predicts the Semantic Velocity (Delta Z) of the next logical chunk.
    """
    def __init__(self, d_model=1024, num_layers=4, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        
        from mamba_ssm import Mamba
        
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'mamba': Mamba(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                ),
                'norm': nn.LayerNorm(d_model)
            }) for _ in range(num_layers)
        ])
        
        self.norm_f = nn.LayerNorm(d_model)
        
        # Output Heads
        self.delta_head = nn.Linear(d_model, d_model)
        
    def forward(self, delta_z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            delta_z: (B, T, D) Sequence of semantic velocities \Delta Z
        Returns:
            pred_delta: (B, T, D)
        """
        B, T, D = delta_z.shape
        
        x = delta_z
        for layer in self.layers:
            residual = x
            x = layer['norm'](x)
            x = layer['mamba'](x)
            x = residual + x
            
        x = self.norm_f(x)
        
        pred_delta = self.delta_head(x)
        
        return pred_delta

def loss_fn_cuda(model, delta_z, valid_mask=None):
    """
    Loss function for the Differential Planner (PyTorch).
    """
    x_in = delta_z[:, :-1, :]
    tgt_delta = delta_z[:, 1:, :]
    
    pred_delta = model(x_in)
    
    # MSE Losses
    delta_loss = torch.mean((pred_delta - tgt_delta)**2, dim=-1) # (B, T-1)
    
    if valid_mask is not None:
        v_mask = valid_mask[:, 1:]
        delta_loss = torch.sum(delta_loss * v_mask) / (torch.sum(v_mask) + 1e-6)
    else:
        delta_loss = torch.mean(delta_loss)
        
    return delta_loss
