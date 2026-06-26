import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        # t: (B,)
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        args = t.unsqueeze(1) * freqs.unsqueeze(0)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2 == 1:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class AdaLNTransformerLayerCUDA(nn.Module):
    """
    Transformer layer with Adaptive Layer Normalization (AdaLN) modulation.
    Uses PyTorch's native scaled_dot_product_attention (FlashAttention on CUDA).
    """
    def __init__(self, d_model, n_heads, dim_cond):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.ln2 = nn.LayerNorm(d_model, elementwise_affine=False)
        
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        
        # AdaLN parameter projection: predicts scale, shift, gate for both attention and MLP
        self.cond_proj = nn.Linear(dim_cond, d_model * 6)
        
        # Initialize cond_proj to zeros so that the network starts as an identity block
        nn.init.zeros_(self.cond_proj.weight)
        nn.init.zeros_(self.cond_proj.bias)

    def forward(self, x, cond):
        # x: (B, L, d_model)
        # cond: (B, dim_cond)
        ada_params = self.cond_proj(cond) # (B, d_model * 6)
        ada_params = ada_params.unsqueeze(1) # (B, 1, d_model * 6)
        scale_a, shift_a, gate_a, scale_m, shift_m, gate_m = torch.chunk(ada_params, 6, dim=-1)
        
        # Self-Attention Block with AdaLN
        x_norm = self.ln1(x)
        x_mod = x_norm * (1 + scale_a) + shift_a
        attn_out, _ = self.attn(x_mod, x_mod, x_mod)
        x = x + gate_a * attn_out
        
        # MLP Block with AdaLN
        x_norm2 = self.ln2(x)
        x_mod2 = x_norm2 * (1 + scale_m) + shift_m
        mlp_out = self.mlp(x_mod2)
        x = x + gate_m * mlp_out
        
        return x


class NARFlowMatcherCUDA(nn.Module):
    """
    Conditional Sequence Flow Matcher (PyTorch CUDA Version).
    Predicts the vector field v_t given intermediate flow state X_t, timestep t, and macro condition Z_macro.
    """
    def __init__(self, z_dim: int, x_dim: int, d_model: int = 512, n_layers: int = 6, n_heads: int = 8, max_seq_len: int = 64):
        super().__init__()
        self.z_dim = z_dim
        self.x_dim = x_dim # e.g. 1024, matching TinyCharEncoder's tok_emb dimension
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
        self.layers = nn.ModuleList([
            AdaLNTransformerLayerCUDA(d_model, n_heads, d_model) for _ in range(n_layers)
        ])
        self.final_ln = nn.LayerNorm(d_model)
        
    def forward(self, x_t, t, z_macro):
        """
        x_t: (B, L, x_dim) - intermediate flow state
        t: (B,) - scalar flow timesteps
        z_macro: (B, z_dim) - global macro condition
        Returns: Predicted vector field v_t (B, L, x_dim)
        """
        B, L, _ = x_t.shape
        
        # 1. Project input sequence to hidden dimension
        h = self.in_proj(x_t) # (B, L, d_model)
        
        # 2. Add position embeddings
        positions = torch.arange(L, device=x_t.device).unsqueeze(0) # (1, L)
        h = h + self.pos_emb(positions) # (B, L, d_model)
        
        # 3. Compute global condition vector
        t_emb = self.t_embedder(t) # (B, d_model)
        z_emb = self.z_proj(z_macro) # (B, d_model)
        cond = t_emb + z_emb # (B, d_model)
        
        # 4. Process through Transformer layers
        for layer in self.layers:
            h = layer(h, cond)
            
        h = self.final_ln(h)
        
        # 5. Project back to output dimension
        v_pred = self.out_proj(h) # (B, L, x_dim)
        return v_pred
