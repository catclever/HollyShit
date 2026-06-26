import mlx.core as mx
import mlx.nn as nn
import math

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations in MLX.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.frequency_embedding_size = frequency_embedding_size

    def timestep_embedding(self, t, dim, max_period=10000):
        # t shape: (B,) or (B, 1)
        half = dim // 2
        freqs = mx.exp(-math.log(max_period) * mx.arange(0, half, dtype=mx.float32) / half)
        if len(t.shape) == 1:
            t = mx.expand_dims(t, axis=1)
        args = t * freqs
        embedding = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
        if dim % 2 == 1:
            embedding = mx.concatenate([embedding, mx.zeros((t.shape[0], 1))], axis=-1)
        return embedding

    def __call__(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class MultiHeadAttention(nn.Module):
    """
    Standard Multi-Head Attention without RoPE, specifically for sequence flow matching.
    """
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def __call__(self, x: mx.array, mask: mx.array = None):
        B, L, _ = x.shape
        q = self.q_proj(x).reshape(B, L, self.n_heads, self.d_head)
        k = self.k_proj(x).reshape(B, L, self.n_heads, self.d_head)
        v = self.v_proj(x).reshape(B, L, self.n_heads, self.d_head)
        
        q = q.transpose(0, 2, 1, 3) # (B, H, L, D)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        
        out = mx.fast.scaled_dot_product_attention(
            q, k, v,
            scale=1.0 / (self.d_head ** 0.5),
            mask=mask
        )
        
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.out_proj(out)


class AdaLNTransformerLayer(nn.Module):
    """
    Transformer layer with Adaptive Layer Normalization (AdaLN) in MLX.
    """
    def __init__(self, d_model: int, n_heads: int, dim_cond: int):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        
        self.cond_proj = nn.Linear(dim_cond, d_model * 6)
        
    def __call__(self, x: mx.array, cond: mx.array) -> mx.array:
        ada_params = self.cond_proj(cond) # (B, d_model * 6)
        ada_params = mx.expand_dims(ada_params, axis=1) # (B, 1, d_model * 6)
        
        splits = mx.split(ada_params, 6, axis=-1)
        scale_a, shift_a, gate_a, scale_m, shift_m, gate_m = splits
        
        # Attention block with AdaLN
        x_norm = self.ln1(x)
        x_mod = x_norm * (1.0 + scale_a) + shift_a
        attn_out = self.attention(x_mod)
        x = x + gate_a * attn_out
        
        # MLP block with AdaLN
        x_norm2 = self.ln2(x)
        x_mod2 = x_norm2 * (1.0 + scale_m) + shift_m
        mlp_out = self.mlp(x_mod2)
        x = x + gate_m * mlp_out
        
        return x


class NARFlowMatcher(nn.Module):
    """
    Conditional Sequence Flow Matcher (MLX Version).
    """
    def __init__(self, z_dim: int, x_dim: int, d_model: int = 512, n_layers: int = 6, n_heads: int = 8, max_seq_len: int = 64):
        super().__init__()
        self.z_dim = z_dim
        self.x_dim = x_dim
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        
        # Map input features to d_model if dimensions don't match
        self.in_proj = nn.Linear(x_dim, d_model) if x_dim != d_model else nn.Identity()
        self.out_proj = nn.Linear(d_model, x_dim) if x_dim != d_model else nn.Identity()
        
        # Condition embeddings
        self.t_embedder = TimestepEmbedder(d_model)
        self.z_proj = nn.Sequential(
            nn.Linear(z_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )
        
        # Absolute position embeddings
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        
        # Transformer Layer Stack
        self.layers = [
            AdaLNTransformerLayer(d_model, n_heads, d_model) for _ in range(n_layers)
        ]
        self.final_ln = nn.LayerNorm(d_model)
        
    def __call__(self, x_t: mx.array, t: mx.array, z_macro: mx.array) -> mx.array:
        """
        x_t: (B, L, x_dim)
        t: (B,)
        z_macro: (B, z_dim)
        """
        B, L, _ = x_t.shape
        
        h = self.in_proj(x_t) # (B, L, d_model)
        
        positions = mx.arange(L)[None, :] # (1, L)
        h = h + self.pos_emb(positions) # (B, L, d_model)
        
        t_emb = self.t_embedder(t) # (B, d_model)
        z_emb = self.z_proj(z_macro) # (B, d_model)
        cond = t_emb + z_emb # (B, d_model)
        
        for layer in self.layers:
            h = layer(h, cond)
            
        h = self.final_ln(h)
        v_pred = self.out_proj(h) # (B, L, x_dim)
        return v_pred
