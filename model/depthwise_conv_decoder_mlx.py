import mlx.core as mx
import mlx.nn as nn
from dataclasses import dataclass

class DepthwiseConv1DBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 5, dilation: int = 1):
        super().__init__()
        
        # 1. Depthwise Convolution: Each dimension only mixes with itself across time
        # MLX Conv1d expects input shape [N, L, C]
        padding = ((kernel_size - 1) * dilation) // 2
        self.depthwise_conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=padding, # Same padding, accounts for dilation
            stride=1,
            dilation=dilation,
            groups=channels,          # <--- The magic that isolates dimensions!
            bias=False
        )
        self.dw_ln = nn.LayerNorm(channels, affine=False)
        self.dw_gelu = nn.GELU()
        
        # 2. Pointwise Convolution (1x1): Cross-dimension feature mixing
        self.pointwise_conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            bias=False
        )
        self.pw_ln = nn.LayerNorm(channels, affine=False)
        self.pw_gelu = nn.GELU()

    def __call__(self, x: mx.array) -> mx.array:
        # x is [B, L, C]
        residual = x
        
        # Phase 1: Natural Affinity (Time-tracking per orthogonal dimension)
        x = self.depthwise_conv(x)
        x = self.dw_ln(x)
        x = self.dw_gelu(x)
        
        # Phase 2: Feature Mixing (Across 1024 dimensions)
        x = self.pointwise_conv(x)
        x = self.pw_ln(x)
        x = self.pw_gelu(x)
        
        return x + residual

@dataclass
class DepthwiseConvDecoderConfig:
    vocab_size: int = 32000
    embed_dim: int = 1024
    hidden_dim: int = 1024
    num_layers: int = 4
    kernel_size: int = 5
    use_dilation: bool = False

class ParallelDepthwiseConvDecoder(nn.Module):
    """
    Phase 2.2.2: A fully parallel, Non-Autoregressive (NAT) Convolutional Decoder.
    It uses Depthwise Separable Convolutions to strictly enforce the logic that 
    the n-th dimension at t=1 is naturally affine to the n-th dimension at t=2.
    """
    def __init__(self, config: DepthwiseConvDecoderConfig = None, **kwargs):
        super().__init__()
        if config is None:
            config = DepthwiseConvDecoderConfig(**kwargs)
        self.config = config
        self.vocab_size = config.vocab_size
        
        self.concept_proj = nn.Linear(config.embed_dim, config.hidden_dim) if config.embed_dim != config.hidden_dim else None
        
        # Stack of Depthwise Separable 1D convolutions (Exponential Dilation)
        self.blocks = []
        for i in range(config.num_layers):
            dilation = (2 ** i) if config.use_dilation else 1
            self.blocks.append(DepthwiseConv1DBlock(config.hidden_dim, config.kernel_size, dilation=dilation))
        
        self.final_ln = nn.LayerNorm(config.hidden_dim, affine=False)
        self.fc = nn.Linear(config.hidden_dim, config.vocab_size)

    def get_config(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self.config)

    def __call__(self, concept_sequence_zoh: mx.array) -> mx.array:
        # concept_sequence_zoh: [B, L, embed_dim]
        if self.concept_proj is not None:
            x = self.concept_proj(concept_sequence_zoh)
        else:
            x = concept_sequence_zoh
            
        for block in self.blocks:
            x = block(x)
            
        x_final = self.final_ln(x)
        logits = self.fc(x_final) # [B, L, V]
        return logits

    def generate(self, concept_sequence_zoh: mx.array, temperature: float = 0.0) -> mx.array:
        """
        Parallel generation! No autoregressive loop.
        """
        logits = self(concept_sequence_zoh) # [B, L, V]
        
        if temperature > 0:
            logits = logits / temperature
            tokens = mx.random.categorical(logits) # [B, L]
        else:
            tokens = mx.argmax(logits, axis=-1) # [B, L]
            
        return tokens
