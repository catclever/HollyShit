import mlx.core as mx
import mlx.nn as nn
import math

class RotaryPositionalEmbeddings(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
        self.inv_freq = inv_freq
        
    def __call__(self, q, k):
        seq_len = q.shape[2]
        t = mx.arange(seq_len, dtype=mx.float32)
        freqs = mx.outer(t, self.inv_freq)
        emb = mx.concatenate((freqs, freqs), axis=-1)
        cos = mx.cos(emb)[None, None, :, :]
        sin = mx.sin(emb)[None, None, :, :]
        
        def apply_rotary(x):
            d1, d2 = mx.split(x, 2, axis=-1)
            x_half = mx.concatenate((-d2, d1), axis=-1)
            return (x * cos) + (x_half * sin)
            
        return apply_rotary(q), apply_rotary(k)

class RoPETransformerLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)
        
        self.rope = RotaryPositionalEmbeddings(self.head_dim)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )
        
    def __call__(self, x, mask=None):
        B, L, _ = x.shape
        
        res = x
        x = self.norm1(x)
        
        q = self.wq(x).reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.wk(x).reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.wv(x).reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        q, k = self.rope(q, k)
        
        scores = (q @ k.transpose(0, 1, 3, 2)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
            
        attn = mx.softmax(scores, axis=-1)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, L, -1)
        out = self.wo(out)
        
        x = res + out
        
        res = x
        x = self.norm2(x)
        x = self.net(x)
        x = res + x
        
        return x

from dataclasses import dataclass

@dataclass
class CharEncoderConfig:
    vocab_size: int = 32000
    d_model: int = 1024
    n_heads: int = 8
    n_layers: int = 6
    max_seq_len: int = 512
    z_dim: int = 1024

class TinyCharEncoder(nn.Module):
    """
    A lightweight, pure-native Text-to-Vector encoder.
    Distilled from the GodEncoder multi-view space.
    Upgraded: Uses RoPE (Rotary Position Embeddings) to allow infinite length extrapolation 
    without O(N^2) memory bloating during training!
    """
    def __init__(self, config: CharEncoderConfig = None, **kwargs):
        super().__init__()
        if config is None:
            config = CharEncoderConfig(**kwargs)
        self.config = config
        
        self.tok_emb = nn.Embedding(self.config.vocab_size, self.config.d_model)
        
        # A custom symmetric encoder explicitly built with RoPE logic
        self.layers = [RoPETransformerLayer(self.config.d_model, self.config.n_heads) for _ in range(self.config.n_layers)]
        self.final_ln = nn.LayerNorm(self.config.d_model)
        
        # Final projection from inner transformer dimension to the absolute GodEncoder semantic space dimension
        self.out_proj = nn.Linear(self.config.d_model, self.config.z_dim)

    def get_config(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self.config)
        
    def get_normalized_embeddings(self, x: mx.array) -> mx.array:
        """Helper to extract strictly spherical embeddings."""
        norm = mx.linalg.norm(self.tok_emb.weight, ord=2, axis=-1, keepdims=True)
        # Avoid division by zero
        normalized_weight = self.tok_emb.weight / mx.maximum(norm, 1e-8)
        return normalized_weight[x]

    def __call__(self, x: mx.array, attention_mask: mx.array = None, return_seq: bool = False) -> mx.array:
        """
        x: (B, L)
        attention_mask: (B, L) where 1 is valid, 0 is pad
        """
        B, L = x.shape
        
        # Pure Token Embeddings (Projected to Hypersphere)
        h = self.get_normalized_embeddings(x) # (B, L, d_model)
        
        # Create non-causal additive padding mask
        mask = None
        if attention_mask is not None:
            mask = mx.where(attention_mask[:, None, None, :] == 0, mx.array(-1e9), mx.array(0.0))
            
        # Pumping through the RoPE Stack
        for layer in self.layers:
            h = layer(h, mask)
            
        h = self.final_ln(h)
        
        if return_seq:
            return self.out_proj(h) # (B, L, z_dim)
        
        # Average Pooling (Masked) to extract the holistic sentence representation
        if attention_mask is not None:
            h_masked = h * attention_mask[:, :, None] # zero out padded positions
            # Safe division by sum
            h_pool = h_masked.sum(axis=1) / mx.maximum(attention_mask.sum(axis=1, keepdims=True), 1e-8)
        else:
            h_pool = h.mean(axis=1) # (B, d_model)
            
        # Project into the 1024-D Z_target space!
        z_pred = self.out_proj(h_pool) # (B, z_dim)
        
        return z_pred


