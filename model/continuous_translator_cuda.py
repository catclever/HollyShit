import torch
import torch.nn as nn

class ContinuousToTextTranslatorCUDA(nn.Module):
    """
    A lightweight, local-window convolutional dictionary projector for Phase 4.
    Translates the fully unrolled continuous manifold sequence (B, L, D) 
    back into human-readable discrete token probabilities (B, L, V).
    
    Uses a 1D convolution with a small receptive field (kernel_size=3) to capture 
    local trajectory velocity and curvature. This eliminates character boundary jitter 
    and missing tokens ('镂空') without causing autoregressive posterior collapse.
    
    Maintains O(1) non-autoregressive parallel decoding.
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
        
    def forward(self, x_continuous: torch.Tensor) -> torch.Tensor:
        """
        x_continuous: (B, L, D) The continuous sequence curve (e.g. from Unet or Phase 1).
        Returns: logits (B, L, V)
        """
        # (B, L, D) -> (B, D, L) for Conv1d
        h_conv = self.conv(x_continuous.transpose(1, 2)).transpose(1, 2)
        h = self.norm(x_continuous + self.act(h_conv))
        return self.fc(h)
        
    @torch.no_grad()
    def decode_greedy(self, x_continuous: torch.Tensor) -> torch.Tensor:
        """
        x_continuous: (B, L, D)
        Returns: token_ids (B, L)
        """
        logits = self(x_continuous)
        return logits.argmax(dim=-1)
