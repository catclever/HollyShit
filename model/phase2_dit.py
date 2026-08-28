import torch
import torch.nn as nn
import math

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.freq_dim = frequency_embedding_size

    def forward(self, t):
        # t: (B,)
        half_dim = self.freq_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=t.device) * -emb)
        emb = t.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)
        return self.mlp(emb)

class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_size * 4)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden, hidden_size)
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        
        norm_x = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + gate_msa.unsqueeze(1) * attn_out
        
        norm_x2 = modulate(self.norm2(x), shift_mlp, scale_mlp)
        mlp_out = self.mlp(norm_x2)
        x = x + gate_mlp.unsqueeze(1) * mlp_out
        return x

class ConceptDiT(nn.Module):
    """
    Diffusion Transformer for Concept Clouds.
    Generates the next concept cloud H_{t+1} given the thought vector z_macro.
    """
    def __init__(self, d_model=512, max_len=64, num_layers=6, num_heads=8):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        # Time embedding
        self.t_embedder = TimestepEmbedder(d_model)
        
        # Positional embedding for the target concept cloud
        # (Since DiT generates an ordered sequence of vectors)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        
        # To combine z_macro and t
        # z_macro is the "class" or "context" condition
        self.z_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )
        
        self.blocks = nn.ModuleList([
            DiTBlock(d_model, num_heads) for _ in range(num_layers)
        ])
        
        self.final_layer = nn.Sequential(
            nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6),
            nn.Linear(d_model, d_model)
        )
        
    def forward(self, x, t, z_macro):
        """
        x: (B, L, D) - Noisy concept cloud
        t: (B,) - Timesteps
        z_macro: (B, D) - Thought vector from Mamba
        """
        # Embed time and z_macro
        t_emb = self.t_embedder(t) # (B, D)
        z_emb = self.z_proj(z_macro) # (B, D)
        
        # Combine condition
        c = t_emb + z_emb # (B, D)
        
        # Add positional embedding to x
        x = x + self.pos_embed[:, :x.size(1), :]
        
        # Forward through blocks
        for block in self.blocks:
            x = block(x, c)
            
        x = self.final_layer(x)
        return x
