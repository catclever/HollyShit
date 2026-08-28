import mlx.core as mx
import mlx.nn as nn

class MambaReaderMLX(nn.Module):
    """
    Reads a sequence of concept vectors H (Phase 1 output) and compresses 
    them into a single macro-thought vector z_macro.
    (MLX Version using a standard Transformer Encoder as a stable fallback 
    for Mac local testing, since mamba-ssm is CUDA-only).
    """
    def __init__(self, d_model=512, num_layers=4, num_heads=8):
        super().__init__()
        self.d_model = d_model
        
        # We use a standard Transformer Encoder for the MLX version to 
        # ensure it runs perfectly on Mac out-of-the-box.
        self.layers = [
            nn.TransformerEncoderLayer(dims=d_model, num_heads=num_heads)
            for _ in range(num_layers)
        ]
        
        self.norm_f = nn.LayerNorm(d_model)
        
        # Projection to ensure z_macro has the exact desired latent dimension
        self.z_proj = nn.Linear(d_model, d_model)
        
    def __call__(self, x: mx.array, mask: mx.array = None) -> mx.array:
        """
        x: (B, L, D) - Sequence of contextual vectors from Phase 1
        Returns: (B, D) - The summarized z_macro vector
        """
        for layer in self.layers:
            x = layer(x, mask=mask)
            
        x = self.norm_f(x)
        
        # We can just take the mean over the sequence length, 
        # or use the last token (causal)
        z_macro = mx.mean(x, axis=1)
        z_macro = self.z_proj(z_macro)
        
        return z_macro
