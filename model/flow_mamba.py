import torch
import torch.nn as nn

class MambaReader(nn.Module):
    """
    Reads a sequence of concept vectors H (Phase 1 output) and compresses 
    them into a single macro-thought vector z_macro.
    """
    def __init__(self, d_model=512, d_state=16, d_conv=4, expand=2, num_layers=4):
        super().__init__()
        self.d_model = d_model
        
        # In a real training environment, we would use:
        # from mamba_ssm import Mamba
        # For cross-platform compatibility (macOS/MPS), we can wrap it or use a fallback.
        try:
            from mamba_ssm import Mamba
            self.layers = nn.ModuleList([
                Mamba(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                ) for _ in range(num_layers)
            ])
            self.use_mamba_ssm = True
        except ImportError:
            # Fallback to a standard Transformer Encoder or simplified recurrent block 
            # if mamba-ssm is not installed (e.g., on macOS for local testing)
            print("Warning: mamba_ssm not found. Falling back to Transformer for local testing.")
            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8, batch_first=True)
            self.layers = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.use_mamba_ssm = False
            
        self.norm_f = nn.LayerNorm(d_model)
        
        # Projection to ensure z_macro has the exact desired latent dimension
        # (could be same as d_model, or a bottleneck e.g., 256)
        self.z_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        """
        x: (B, L, D) - Sequence of contextual vectors from Phase 1
        Returns: (B, D) - The summarized z_macro vector
        """
        if self.use_mamba_ssm:
            for layer in self.layers:
                x = layer(x)
        else:
            x = self.layers(x)
            
        x = self.norm_f(x)
        
        # We take the final state (causal summarization)
        # For Mamba, the last token contains the summary of the causal chain
        last_hidden_state = x[:, -1, :] 
        
        z_macro = self.z_proj(last_hidden_state)
        return z_macro
