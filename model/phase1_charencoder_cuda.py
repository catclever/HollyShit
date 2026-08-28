import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RotaryPositionalEmbeddingsCUDA(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 4096):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len
        self._build_cache(max_seq_len)
        
    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=torch.float32, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)
        
    def forward(self, q, k):
        seq_len = q.size(2)
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
            self.max_seq_len = seq_len
            
        cos = self.cos_cached[:, :, :seq_len, ...]
        sin = self.sin_cached[:, :, :seq_len, ...]
        
        def apply_rotary(x):
            d1, d2 = x.chunk(2, dim=-1)
            x_half = torch.cat((-d2, d1), dim=-1)
            return (x * cos) + (x_half * sin)
            
        return apply_rotary(q), apply_rotary(k)

class RoPETransformerLayerCUDA(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.query_proj = nn.Linear(d_model, d_model, bias=False)
        self.key_proj = nn.Linear(d_model, d_model, bias=False)
        self.value_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.rope = RotaryPositionalEmbeddingsCUDA(self.head_dim)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.fc1 = nn.Linear(d_model, 4 * d_model)
        self.fc2 = nn.Linear(4 * d_model, d_model)
        
    def forward(self, x, attention_mask=None):
        B, L, _ = x.size()
        
        res = x
        x = self.norm1(x)
        
        q = self.query_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.key_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.value_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        
        q, k = self.rope(q, k)
        
        attn_mask = None
        if attention_mask is not None:
            attn_mask = attention_mask.view(B, 1, 1, L)
            attn_mask = (attn_mask == 1.0)
            
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        
        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        out = self.out_proj(out)
        
        x = res + out
        
        res = x
        x = self.norm2(x)
        x = self.fc2(F.gelu(self.fc1(x)))
        x = res + x
        
        return x

class TinyCharEncoderCUDA(nn.Module):
    """
    The PyTorch Clone of the Teacher-Distilled Character Flow Encoder.
    Completely refactored for native CUDA performance, utilizing pure PyTorch primitives.
    """
    def __init__(self, vocab_size: int, d_model: int = 1024, n_heads: int = 8, n_layers: int = 6, max_seq_len: int = 512, z_dim: int = 1024):
        super().__init__()
        
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([RoPETransformerLayerCUDA(d_model, n_heads) for _ in range(n_layers)])
        
        self.final_ln = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, z_dim)
        
        # Better initialization for embedding table
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=d_model**-0.5)

    def get_normalized_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Helper to extract strictly spherical embeddings."""
        normalized_weight = F.normalize(self.tok_emb.weight, p=2, dim=-1)
        return F.embedding(x, normalized_weight)

    def forward(self, x, attention_mask=None):
        h = self.get_normalized_embeddings(x)
        
        for layer in self.layers:
            h = layer(h, attention_mask)
            
        h = self.final_ln(h)
        
        if attention_mask is not None:
            # attention_mask: (B, L) where 1 is valid, 0 is pad
            # Safely mask out padding without risking 0 * NaN = NaN propagation
            h_masked = torch.where(attention_mask.unsqueeze(-1) > 0, h, torch.zeros_like(h))
            
            # Safe avg pool
            sum_mask = attention_mask.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            h_pool = h_masked.sum(dim=1) / sum_mask
        else:
            h_pool = h.mean(dim=1)
            
        return self.out_proj(h_pool)


