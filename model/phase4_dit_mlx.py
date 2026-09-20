import mlx.core as mx
import mlx.nn as nn
import math

class TimestepEmbedderMLX(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.freq_dim = frequency_embedding_size

    def __call__(self, t: mx.array) -> mx.array:
        # t: (B,)
        half_dim = self.freq_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = mx.exp(mx.arange(half_dim, dtype=mx.float32) * -emb)
        emb = mx.expand_dims(t, 1) * mx.expand_dims(emb, 0)
        emb = mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=-1)
        return self.mlp(emb)

class DiTBlockMLX(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, affine=False)
        self.attn = nn.MultiHeadAttention(hidden_size, num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, affine=False)
        mlp_hidden = int(hidden_size * 4)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, hidden_size)
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size)
        )

    def modulate(self, x, shift, scale):
        return x * (1 + mx.expand_dims(scale, 1)) + mx.expand_dims(shift, 1)

    def __call__(self, x, c):
        # c: [B, D]
        mod = self.adaLN_modulation(c)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mx.split(mod, 6, axis=-1)
        
        norm_x = self.modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out = self.attn(norm_x, norm_x, norm_x)
        x = x + mx.expand_dims(gate_msa, 1) * attn_out
        
        norm_x2 = self.modulate(self.norm2(x), shift_mlp, scale_mlp)
        mlp_out = self.mlp(norm_x2)
        x = x + mx.expand_dims(gate_mlp, 1) * mlp_out
        return x

class ConceptDiTMLX(nn.Module):
    """
    Diffusion Transformer for Concept Clouds (MLX Version).
    """
    def __init__(self, d_model=512, max_len=128, num_layers=6, num_heads=8):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        self.t_embedder = TimestepEmbedderMLX(d_model)
        
        # Positional embedding
        self.pos_embed = mx.random.normal((1, max_len, d_model)) * 0.02
        
        self.z_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )
        
        self.blocks = [DiTBlockMLX(d_model, num_heads) for _ in range(num_layers)]
        
        self.final_layer = nn.Sequential(
            nn.LayerNorm(d_model, affine=False),
            nn.Linear(d_model, d_model)
        )
        
    def __call__(self, x: mx.array, t: mx.array, z_macro: mx.array) -> mx.array:
        """
        x: (B, L, D) - Noisy concept cloud
        t: (B,) - Timesteps
        z_macro: (B, D) - Thought vector from Mamba
        """
        t_emb = self.t_embedder(t)
        z_emb = self.z_proj(z_macro)
        
        c = t_emb + z_emb
        
        x = x + self.pos_embed[:, :x.shape[1], :]
        
        for block in self.blocks:
            x = block(x, c)
            
        x = self.final_layer(x)
        return x
