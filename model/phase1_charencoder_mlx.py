import mlx.core as mx
import mlx.nn as nn
import math

class RoPETransformerLayerMLX(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.query_proj = nn.Linear(d_model, d_model, bias=False)
        self.key_proj = nn.Linear(d_model, d_model, bias=False)
        self.value_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.rope = nn.RoPE(self.head_dim, traditional=True)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.fc1 = nn.Linear(d_model, 4 * d_model)
        self.fc2 = nn.Linear(4 * d_model, d_model)
        
    def __call__(self, x, attention_mask=None):
        B, L, _ = x.shape
        
        res = x
        x = self.norm1(x)
        
        q = self.query_proj(x).reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.key_proj(x).reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.value_proj(x).reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        q = self.rope(q)
        k = self.rope(k)
        
        scale = 1.0 / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            # attention_mask: (B, L) where 1 is valid, 0 is pad
            attn_mask = (attention_mask == 0).reshape(B, 1, 1, L) * -1e9
            out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=attn_mask)
        else:
            out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
            
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        out = self.out_proj(out)
        
        x = res + out
        
        res = x
        x = self.norm2(x)
        x = self.fc2(nn.gelu(self.fc1(x)))
        x = res + x
        
        return x

from dataclasses import dataclass

@dataclass
class CharEncoderConfig:
    vocab_size: int = 32000
    d_model: int = 1024
    n_heads: int = 8
    n_layers: int = 6
    z_dim: int = 1024

class TinyCharEncoderMLX(nn.Module):
    def __init__(self, config: CharEncoderConfig = None, **kwargs):
        super().__init__()
        if config is None:
            # Fallback for old calls without config
            config = CharEncoderConfig(**kwargs)
        self.config = config
        
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = [RoPETransformerLayerMLX(config.d_model, config.n_heads) for _ in range(config.n_layers)]
        
        self.final_ln = nn.LayerNorm(config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.z_dim)

    def get_config(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self.config)

    def get_normalized_embeddings(self, x: mx.array) -> mx.array:
        """Extract strictly spherical embeddings."""
        norms = mx.linalg.norm(self.tok_emb.weight, ord=2, axis=-1, keepdims=True)
        normalized_weight = self.tok_emb.weight / norms
        return normalized_weight[x]

    def __call__(self, x, attention_mask=None):
        h = self.get_normalized_embeddings(x)
        
        for layer in self.layers:
            h = layer(h, attention_mask)
            
        h = self.final_ln(h)
        
        if attention_mask is not None:
            h_masked = mx.where(attention_mask[..., None] > 0, h, mx.zeros_like(h))
            sum_mask = mx.maximum(attention_mask.sum(axis=-1, keepdims=True), mx.array([1e-8]))
            h_pool = h_masked.sum(axis=1) / sum_mask
        else:
            h_pool = h.mean(axis=1)
            
        return self.out_proj(h_pool)


