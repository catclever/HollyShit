"""
Archived on: 2026-07-27
Reason: Archived before implementing the new Mamba-DiT Next-Thought Flow Matching architecture. The original phase 2 was a pure DiT autoencoder without the Mamba macro-concept reader.
"""
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
        # Scale continuous t in [0,1] to [0, 1000] to give the sinusoidal embeddings proper high-frequency resolution.
        # Otherwise, for t in [0, 1], the arguments to sin/cos never even complete a single period!
        t_scaled = t * 1000.0
        
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        args = t_scaled.unsqueeze(1) * freqs.unsqueeze(0)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2 == 1:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    # xq, xk: (B, L, H, D_head)
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    freqs_cis = freqs_cis[:xq.shape[1]]
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2) # (1, L, 1, D_head/2)
    
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

def get_1d_sincos_pos_embed(embed_dim, length):
    """
    Creates 1D sinusoidal positional embeddings.
    Returns: [length, embed_dim]
    """
    pos = torch.arange(length, dtype=torch.float32)
    omega = torch.arange(embed_dim // 2, dtype=torch.float32)
    omega /= (embed_dim / 2.)
    omega = 1. / 10000**omega
    
    out = torch.einsum('m,d->md', pos, omega)
    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)
    emb = torch.cat([emb_sin, emb_cos], dim=1)
    return emb


class AdaLNTransformerLayerCUDA(nn.Module):
    """
    Transformer layer with Adaptive Layer Normalization (AdaLN) modulation.
    Uses PyTorch's scaled_dot_product_attention with RoPE injection.
    """
    def __init__(self, d_model, n_heads, dim_cond):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
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

    def forward(self, x, cond, freqs_cis=None):
        # x: (B, L, d_model)
        B, L, _ = x.shape
        ada_params = self.cond_proj(cond) # (B, d_model * 6)
        ada_params = ada_params.unsqueeze(1) # (B, 1, d_model * 6)
        scale_a, shift_a, gate_a, scale_m, shift_m, gate_m = torch.chunk(ada_params, 6, dim=-1)
        
        # Self-Attention Block with AdaLN
        x_norm = self.ln1(x)
        x_mod = x_norm * (1 + scale_a) + shift_a
        
        q = self.q_proj(x_mod).view(B, L, self.n_heads, self.head_dim)
        k = self.k_proj(x_mod).view(B, L, self.n_heads, self.head_dim)
        v = self.v_proj(x_mod).view(B, L, self.n_heads, self.head_dim)
        
        if freqs_cis is not None:
            q, k = apply_rotary_emb(q, k, freqs_cis)
            
        q = q.transpose(1, 2) # (B, H, L, D_head)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).reshape(B, L, self.d_model)
        attn_out = self.o_proj(attn_out)
        
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
        
        # Absolute Positional Embeddings (Crucial for breaking symmetry of pure noise at t=0)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, d_model), requires_grad=False)
        pos_embed = get_1d_sincos_pos_embed(d_model, max_seq_len)
        self.pos_embed.data.copy_(pos_embed.unsqueeze(0))
        
        # Condition embeddings
        self.t_embedder = TimestepEmbedder(d_model)
        self.z_proj = nn.Sequential(
            nn.Linear(z_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )
        # Precompute RoPE frequencies
        freqs_cis = precompute_freqs_cis(d_model // n_heads, max_seq_len * 4) # extra headroom for longer sequences during inference
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)
        
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
        
        # 2. Add Absolute Positional Embeddings
        # Without this, the model cannot distinguish between identical noise vectors at t=0
        h = h + self.pos_embed[:, :L, :]
        
        # 3. Compute global condition vector
        t_emb = self.t_embedder(t) # (B, d_model)
        z_emb = self.z_proj(z_macro) # (B, d_model)
        cond = t_emb + z_emb # (B, d_model)
        
        # 4. Process through Transformer layers
        for layer in self.layers:
            h = layer(h, cond, self.freqs_cis)
            
        h = self.final_ln(h)
        
        # 5. Project back to output dimension
        v_pred = self.out_proj(h) # (B, L, x_dim)
        return v_pred
