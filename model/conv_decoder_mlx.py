import mlx.core as mx
import mlx.nn as nn

class Conv1DBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        # MLX Conv1d expects input shape [N, L, C]
        self.conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2, # Same padding
            bias=False
        )
        self.ln = nn.LayerNorm(channels, affine=False)
        self.gelu = nn.GELU()

    def __call__(self, x: mx.array) -> mx.array:
        # x is [B, L, C]
        residual = x
        x = self.conv(x)
        x = self.ln(x)
        x = self.gelu(x)
        return x + residual

from dataclasses import dataclass

@dataclass
class ConvDecoderConfig:
    vocab_size: int = 32000
    embed_dim: int = 1024
    hidden_dim: int = 1024
    num_layers: int = 4
    kernel_size: int = 5

class ParallelConvDecoder(nn.Module):
    """
    Phase 2.2: A fully parallel, Non-Autoregressive (NAT) Convolutional Decoder.
    It takes the Zero-Order Hold (ZOH) concept sequence and applies a few layers of 
    1D Convolution over the time axis to perform local syntactic smoothing.
    It generates all tokens simultaneously in a single forward pass.
    """
    def __init__(self, config: ConvDecoderConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        
        self.concept_proj = nn.Linear(config.embed_dim, config.hidden_dim) if config.embed_dim != config.hidden_dim else None
        
        # Stack of 1D convolutions for local syntactic smoothing
        self.blocks = [Conv1DBlock(config.hidden_dim, config.kernel_size) for _ in range(config.num_layers)]
        
        self.final_ln = nn.LayerNorm(config.hidden_dim, affine=False)
        self.fc = nn.Linear(config.hidden_dim, config.vocab_size)

    def get_config(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self.config)

    def __call__(self, concept_sequence: mx.array) -> mx.array:
        # concept_sequence: [B, L, embed_dim]
        if self.concept_proj is not None:
            x = self.concept_proj(concept_sequence)
        else:
            x = concept_sequence
            
        for block in self.blocks:
            x = block(x)
            
        x_final = self.final_ln(x)
        logits = self.fc(x_final) # [B, L, V]
        return logits

    def generate(self, concept_sequence: mx.array, temperature: float = 0.0) -> mx.array:
        """
        Parallel generation! No autoregressive loop.
        """
        logits = self(concept_sequence) # [B, L, V]
        
        if temperature > 0:
            logits = logits / temperature
            tokens = mx.random.categorical(logits) # [B, L]
        else:
            tokens = mx.argmax(logits, axis=-1) # [B, L]
            
        return tokens
