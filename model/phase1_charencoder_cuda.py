import os
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def load_mlx_safetensors_into_torch(torch_module: nn.Module, safetensor_path: str):
    """
    Loads an MLX-generated .safetensors or PyTorch .pt checkpoint into a PyTorch nn.Module
    without requiring MLX to be imported at module load time.
    """
    if not os.path.exists(safetensor_path):
        raise FileNotFoundError(f"Checkpoint not found at {safetensor_path}")

    if safetensor_path.endswith(".pt") or safetensor_path.endswith(".pth"):
        state_dict = torch.load(safetensor_path, map_location="cpu", weights_only=True)
    else:
        from safetensors.torch import load_file
        state_dict = load_file(safetensor_path)

    target_keys = set(torch_module.state_dict().keys())
    remapped = {}
    for k, v in state_dict.items():
        new_k = k
        new_k = new_k.replace(".wq.", ".query_proj.")
        new_k = new_k.replace(".wk.", ".key_proj.")
        new_k = new_k.replace(".wv.", ".value_proj.")
        new_k = new_k.replace(".wo.", ".out_proj.")

        if new_k not in target_keys:
            test_k1 = new_k.replace(".net.layers.0.", ".fc1.").replace(".net.layers.2.", ".fc2.")
            test_k1 = test_k1.replace("net.layers.0.", "fc1.").replace("net.layers.2.", "fc2.")
            test_k2 = new_k.replace(".net.layers.0.", ".linear1.").replace(".net.layers.2.", ".linear2.")
            if test_k1 in target_keys:
                new_k = test_k1
            elif test_k2 in target_keys:
                new_k = test_k2

        remapped[new_k] = v

    torch_module.load_state_dict(remapped, strict=False)
    return torch_module


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

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.size(2)
        if seq_len > self.max_seq_len or self.cos_cached.device != q.device:
            self._build_cache(max(seq_len, self.max_seq_len))
            self.max_seq_len = max(seq_len, self.max_seq_len)

        cos = self.cos_cached[:, :, :seq_len, ...].to(dtype=q.dtype, device=q.device)
        sin = self.sin_cached[:, :, :seq_len, ...].to(dtype=q.dtype, device=q.device)

        def apply_rotary(x: torch.Tensor) -> torch.Tensor:
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

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, _ = x.size()

        res = x
        x = self.norm1(x)

        q = self.query_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.key_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.value_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        q, k = self.rope(q, k)

        attn_mask = None
        if attention_mask is not None:
            valid_key = (attention_mask > 0).view(B, 1, 1, L)
            # Guard against all-PAD sequences producing NaN in softmax
            has_any_valid = valid_key.any(dim=-1, keepdim=True)
            attn_mask = torch.where(has_any_valid, valid_key, torch.ones_like(valid_key))

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
    Phase 1 v2 Dual-Head Masked Character Encoder (PyTorch/CUDA).
    Upgrades:
    1. Mandatory <PAD> attention masking: If `attention_mask is None` in `forward`,
       `encode_tokens`, `encode_char_subspace`, or `encode_dual`, automatically builds
       `attention_mask = (x != self.pad_token_id).float()` so Transformer layers never
       attend to <PAD> tokens.
    2. Dual-Head Architecture:
       - Clause Semantic Planning Head: masked mean pool `s_pool` -> `out_proj(s_pool)` (`z_god`).
       - Native Orthogonal 32-D Character Subspace Head: `char_subspace_proj(h - char_token_mean)`
         with persistent `char_token_mean` buffer and Stiefel manifold orthogonality penalty.
    3. 100% backward compatibility with legacy `student.pt` checkpoints.
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 1024,
        n_heads: int = 8,
        n_layers: int = 6,
        max_seq_len: int = 512,
        z_dim: int = 1024,
        d_sub: int = 32,
        pad_token_id: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.z_dim = z_dim
        self.d_sub = d_sub
        self.pad_token_id = int(pad_token_id)

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([RoPETransformerLayerCUDA(d_model, n_heads) for _ in range(n_layers)])

        self.final_ln = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, z_dim)

        # Native Orthogonal 32-D Character Anchor Head + persistent token-mean buffer
        self.char_subspace_proj = nn.Linear(d_model, d_sub, bias=False)
        nn.init.orthogonal_(self.char_subspace_proj.weight)
        self.register_buffer("char_token_mean", torch.zeros(d_model, dtype=torch.float32))

        # Spherical initialization for embedding table
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=d_model**-0.5)

    def get_config(self) -> dict:
        return {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "max_seq_len": self.max_seq_len,
            "z_dim": self.z_dim,
            "d_sub": self.d_sub,
            "pad_token_id": self.pad_token_id,
        }

    def _resolve_attention_mask(
        self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Ensures `attention_mask` is always present and float32 with 1.0 for valid tokens
        and 0.0 for <PAD> tokens. Never allows unmasked <PAD> leakage.
        """
        if attention_mask is None:
            return (x != self.pad_token_id).to(dtype=torch.float32)
        return (attention_mask > 0).to(dtype=torch.float32, device=x.device)

    def get_normalized_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Helper to extract strictly spherical embeddings."""
        normalized_weight = F.normalize(self.tok_emb.weight, p=2, dim=-1)
        return F.embedding(x, normalized_weight)

    def encode_tokens(
        self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Runs RoPE Transformer layers with mandatory attention_mask, applies `final_ln`,
        and strictly zeros out padded positions (`attention_mask == 0`).
        Returns Tensor[B, L, d_model].
        """
        mask = self._resolve_attention_mask(x, attention_mask)
        h = self.get_normalized_embeddings(x)

        for layer in self.layers:
            h = layer(h, mask)

        h = self.final_ln(h)
        valid_mask = (mask > 0).unsqueeze(-1)
        h = torch.where(valid_mask, h, torch.zeros_like(h))
        return h

    def _project_char_subspace_from_tokens(
        self, h: torch.Tensor, mask: torch.Tensor, center: bool = True
    ) -> torch.Tensor:
        valid_mask = (mask > 0).unsqueeze(-1)
        if center:
            mean_vec = self.char_token_mean.detach().to(dtype=h.dtype, device=h.device)
            h_in = h - mean_vec
        else:
            h_in = h
        h_sub = self.char_subspace_proj(h_in)
        h_sub = torch.where(valid_mask, h_sub, torch.zeros_like(h_sub))
        return h_sub

    def encode_char_subspace(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        center: bool = True,
    ) -> torch.Tensor:
        """
        Projects `h = encode_tokens(x, attention_mask)` through the native orthogonal 32-D
        character subspace head `char_subspace_proj(h - char_token_mean)` and zeros out
        padded positions (`attention_mask == 0`).
        Returns Tensor[B, L, d_sub].
        """
        mask = self._resolve_attention_mask(x, attention_mask)
        h = self.encode_tokens(x, mask)
        return self._project_char_subspace_from_tokens(h, mask, center=center)

    def encode_dual(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_hidden: bool = False,
        center_subspace: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Single masked forward pass returning both clause-level semantic vectors and
        token-level 32-D character subspace trajectories:
          - `z_god`: Tensor[B, z_dim]
          - `s_pool`: Tensor[B, d_model]
          - `h_sub`: Tensor[B, L, d_sub]
          - `attention_mask`: Tensor[B, L]
          - `h_tokens`: Tensor[B, L, d_model] (if `return_hidden=True`)
        """
        mask = self._resolve_attention_mask(x, attention_mask)
        h = self.encode_tokens(x, mask)

        sum_mask = mask.sum(dim=-1, keepdim=True).clamp(min=1e-8).to(dtype=h.dtype)
        h_pool = h.sum(dim=1) / sum_mask
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

    def forward(
        self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Backward-compatible forward pass returning `z_god = self.out_proj(h_pool)`
        with automatic <PAD> masking when `attention_mask is None`.
        """
        mask = self._resolve_attention_mask(x, attention_mask)
        h = self.encode_tokens(x, mask)
        sum_mask = mask.sum(dim=-1, keepdim=True).clamp(min=1e-8).to(dtype=h.dtype)
        h_pool = h.sum(dim=1) / sum_mask
        return self.out_proj(h_pool)

    def orthogonal_subspace_loss(self) -> torch.Tensor:
        """
        Stiefel manifold penalty ||W_sub W_sub^T - I_{d_sub}||_F^2 keeping the 32
        character subspace basis rows strictly orthonormal.
        """
        w = self.char_subspace_proj.weight.float()
        gram = torch.matmul(w, w.t())
        eye = torch.eye(self.d_sub, device=w.device, dtype=w.dtype)
        return ((gram - eye) ** 2).sum()

    @torch.no_grad()
    def update_char_token_mean(
        self,
        h_tokens: torch.Tensor,
        attention_mask: torch.Tensor,
        momentum: float = 0.05,
    ) -> torch.Tensor:
        """
        Updates persistent buffer `char_token_mean` via EMA over valid tokens
        (`attention_mask > 0`) during Phase 1 v2 training.
        """
        valid_mask = attention_mask > 0
        if not valid_mask.any():
            return self.char_token_mean
        batch_mean = h_tokens.detach().float()[valid_mask].mean(dim=0)
        if self.char_token_mean.abs().max() < 1e-12:
            self.char_token_mean.copy_(batch_mean.to(dtype=self.char_token_mean.dtype))
        else:
            self.char_token_mean.lerp_(batch_mean.to(dtype=self.char_token_mean.dtype), float(momentum))
        return self.char_token_mean
