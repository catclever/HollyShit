from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttentionCUDA(nn.Module):
    """MLX-compatible key layout: attention.{query,key,value,out}_proj.weight"""

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model({d_model}) must be divisible by num_heads({num_heads})")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.query_proj = nn.Linear(d_model, d_model, bias=False)
        self.key_proj = nn.Linear(d_model, d_model, bias=False)
        self.value_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        x = x.view(bsz, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)  # (B, H, T, Dh)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        q = self._reshape_heads(self.query_proj(x))
        k = self._reshape_heads(self.key_proj(x))
        v = self._reshape_heads(self.value_proj(x))

        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], self.d_model)
        return self.out_proj(attn_out)


class TransformerLayerCUDA(nn.Module):
    """MLX-compatible key layout for each transformer layer."""

    def __init__(self, d_model: int, num_heads: int, mlp_dim: int):
        super().__init__()
        self.attention = CausalSelfAttentionCUDA(d_model, num_heads)
        self.linear1 = nn.Linear(d_model, mlp_dim)
        self.linear2 = nn.Linear(mlp_dim, d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.ln1(x), attn_mask)
        ffn = self.linear2(F.gelu(self.linear1(self.ln2(x))))
        x = x + ffn
        return x


class TransformerStackCUDA(nn.Module):
    def __init__(self, num_layers: int, d_model: int, num_heads: int, mlp_dim: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerLayerCUDA(d_model=d_model, num_heads=num_heads, mlp_dim=mlp_dim) for _ in range(num_layers)]
        )
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, attn_mask)
        return self.ln(x)


class WeakDecoderCUDA(nn.Module):
    def __init__(self, z_dim: int, vocab_size: int, d_model: int = 128, n_layers: int = 2, n_heads: int = 4):
        super().__init__()
        self.z_proj = nn.Linear(z_dim, d_model)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = TransformerStackCUDA(
            num_layers=n_layers,
            d_model=d_model,
            num_heads=n_heads,
            mlp_dim=d_model * 4,
        )
        self.out_proj = nn.Linear(d_model, vocab_size)

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        # Broadcastable to (B, H, T, T) in scaled_dot_product_attention
        return torch.triu(
            torch.full((1, 1, seq_len, seq_len), float("-inf"), device=device),
            diagonal=1,
        )

    def forward(self, z_target: torch.Tensor, token_inputs: torch.Tensor) -> torch.Tensor:
        z_projected = self.z_proj(z_target)
        x = self.embedding(token_inputs) + z_projected[:, None, :]
        mask = self._causal_mask(x.shape[1], x.device)
        x = self.transformer(x, attn_mask=mask)
        return self.out_proj(x)

    @torch.no_grad()
    def generate(
        self,
        z_target: torch.Tensor,
        start_token: int,
        eos_token: Optional[int] = None,
        max_tokens: int = 50,
        temperature: float = 0.7,
    ) -> List[int]:
        z_projected = self.z_proj(z_target)
        tokens = [int(start_token)]

        for _ in range(max_tokens):
            token_tensor = torch.tensor([tokens], dtype=torch.long, device=z_target.device)
            x = self.embedding(token_tensor) + z_projected[:, None, :]
            mask = self._causal_mask(x.shape[1], x.device)
            x = self.transformer(x, attn_mask=mask)
            logits = self.out_proj(x[:, -1, :])

            if temperature == 0:
                next_token = int(torch.argmax(logits, dim=-1).item())
            else:
                probs = torch.softmax(logits / max(temperature, 1e-6), dim=-1)
                next_token = int(torch.multinomial(probs, num_samples=1).item())

            tokens.append(next_token)
            if eos_token is not None and next_token == eos_token:
                break

        return tokens
