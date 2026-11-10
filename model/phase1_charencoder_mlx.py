from dataclasses import dataclass
import math
from typing import Dict, Optional
import mlx.core as mx
import mlx.nn as nn


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

    def __call__(self, x: mx.array, attention_mask: Optional[mx.array] = None) -> mx.array:
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


@dataclass
class CharEncoderConfig:
    vocab_size: int = 32000
    d_model: int = 1024
    n_heads: int = 8
    n_layers: int = 6
    max_seq_len: int = 512
    z_dim: int = 1024
    d_sub: int = 32
    pad_token_id: int = 0


class TinyCharEncoderMLX(nn.Module):
    """
    Phase 1 v2 Dual-Head Masked Character Encoder (MLX Mirror).
    Enforces automatic <PAD> masking when `attention_mask is None` and provides
    `encode_tokens`, `encode_char_subspace`, `encode_dual`, and `orthogonal_subspace_loss`.
    """

    def __init__(self, config: Optional[CharEncoderConfig] = None, **kwargs):
        super().__init__()
        if config is None:
            config = CharEncoderConfig(**kwargs)
        self.config = config
        self.vocab_size = config.vocab_size
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.n_layers = config.n_layers
        self.max_seq_len = config.max_seq_len
        self.z_dim = config.z_dim
        self.d_sub = config.d_sub
        self.pad_token_id = int(config.pad_token_id)

        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = [
            RoPETransformerLayerMLX(config.d_model, config.n_heads)
            for _ in range(config.n_layers)
        ]

        self.final_ln = nn.LayerNorm(config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.z_dim)

        self.char_subspace_proj = nn.Linear(config.d_model, config.d_sub, bias=False)
        self.char_token_mean = mx.zeros((config.d_model,), dtype=mx.float32)

    def get_config(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self.config)

    def _resolve_attention_mask(
        self, x: mx.array, attention_mask: Optional[mx.array] = None
    ) -> mx.array:
        if attention_mask is None:
            return (x != self.pad_token_id).astype(mx.float32)
        return (attention_mask > 0).astype(mx.float32)

    def get_normalized_embeddings(self, x: mx.array) -> mx.array:
        """Extract strictly spherical embeddings."""
        norms = mx.linalg.norm(self.tok_emb.weight, ord=2, axis=-1, keepdims=True)
        normalized_weight = self.tok_emb.weight / mx.maximum(norms, 1e-8)
        return normalized_weight[x]

    def encode_tokens(
        self, x: mx.array, attention_mask: Optional[mx.array] = None
    ) -> mx.array:
        mask = self._resolve_attention_mask(x, attention_mask)
        h = self.get_normalized_embeddings(x)

        for layer in self.layers:
            h = layer(h, mask)

        h = self.final_ln(h)
        h = mx.where(mask[..., None] > 0, h, mx.zeros_like(h))
        return h

    def _project_char_subspace_from_tokens(
        self, h: mx.array, mask: mx.array, center: bool = True
    ) -> mx.array:
        h_in = (h - self.char_token_mean) if center else h
        h_sub = self.char_subspace_proj(h_in)
        h_sub = mx.where(mask[..., None] > 0, h_sub, mx.zeros_like(h_sub))
        return h_sub

    def encode_char_subspace(
        self,
        x: mx.array,
        attention_mask: Optional[mx.array] = None,
        center: bool = True,
    ) -> mx.array:
        mask = self._resolve_attention_mask(x, attention_mask)
        h = self.encode_tokens(x, mask)
        return self._project_char_subspace_from_tokens(h, mask, center=center)

    def encode_dual(
        self,
        x: mx.array,
        attention_mask: Optional[mx.array] = None,
        return_hidden: bool = False,
        center_subspace: bool = True,
    ) -> Dict[str, mx.array]:
        mask = self._resolve_attention_mask(x, attention_mask)
        h = self.encode_tokens(x, mask)

        sum_mask = mx.maximum(mask.sum(axis=-1, keepdims=True), mx.array([1e-8]))
        h_pool = h.sum(axis=1) / sum_mask
        z_god = self.out_proj(h_pool)
        h_sub = self._project_char_subspace_from_tokens(h, mask, center=center_subspace)

        out = {
            "z_god": z_god,
            "s_pool": h_pool,
            "h_sub": h_sub,
            "attention_mask": mask,
        }
        if return_hidden:
            out["h_tokens"] = h
        return out

    def orthogonal_subspace_loss(self) -> mx.array:
        w = self.char_subspace_proj.weight
        gram = w @ w.T
        eye = mx.eye(self.d_sub)
        diff = gram - eye
        return mx.sum(diff * diff)

    def __call__(
        self, x: mx.array, attention_mask: Optional[mx.array] = None
    ) -> mx.array:
        mask = self._resolve_attention_mask(x, attention_mask)
        h = self.encode_tokens(x, mask)
        sum_mask = mx.maximum(mask.sum(axis=-1, keepdims=True), mx.array([1e-8]))
        h_pool = h.sum(axis=1) / sum_mask
        return self.out_proj(h_pool)

    def forward(
        self, x: mx.array, attention_mask: Optional[mx.array] = None
    ) -> mx.array:
        return self.__call__(x, attention_mask=attention_mask)
