import mlx.core as mx
import mlx.nn as nn

class ContinuousToTextTranslatorMLX(nn.Module):
    """
    MLX native version of the ContinuousToTextTranslator for Phase 4.
    Translates the fully unrolled continuous manifold sequence (B, L, D) 
    back into human-readable discrete token probabilities (B, L, V).
    
    Uses a 1D convolution with a small receptive field (kernel_size=3) to capture 
    local trajectory velocity and curvature, eliminating character boundary jitter.
    """
    def __init__(self, d_model: int = 1024, vocab_size: int = 6000, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, vocab_size)
        
    def __call__(self, x_continuous: mx.array) -> mx.array:
        """
        x_continuous: (B, L, D) The continuous sequence curve.
        Returns: logits (B, L, V)
        """
        h_conv = self.act(self.conv(x_continuous))
        h = self.norm(x_continuous + h_conv)
        return self.fc(h)
        
    def decode_greedy(self, x_continuous: mx.array) -> mx.array:
        """
        Returns: token_ids (B, L)
        """
        logits = self(x_continuous)
        return mx.argmax(logits, axis=-1)
