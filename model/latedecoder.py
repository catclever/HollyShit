import mlx.core as mx
import mlx.nn as nn
import math

class CausalTransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim, affine=False)
        self.ln2 = nn.LayerNorm(hidden_dim, affine=False)
        self.attn = nn.MultiHeadAttention(hidden_dim, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
    def __call__(self, x: mx.array, mask: mx.array = None, cache=None) -> mx.array:
        # If cache is provided, use it for efficient auto-regressive generation
        if cache is not None:
            x_norm = self.ln1(x)
            attn_out, cache = self.attn(x_norm, x_norm, x_norm, mask=mask, cache=cache)
            x = x + attn_out
            x = x + self.mlp(self.ln2(x))
            return x, cache
        else:
            x_norm = self.ln1(x)
            attn_out = self.attn(x_norm, x_norm, x_norm, mask=mask)
            x = x + attn_out
            x = x + self.mlp(self.ln2(x))
            return x

from dataclasses import dataclass

@dataclass
class LateFusionConfig:
    vocab_size: int = 32000
    embed_dim: int = 1024
    hidden_dim: int = 1024
    num_layers: int = 4
    num_heads: int = 8
    max_len: int = 512
    window_size: int = None

class LateFusionDecoder(nn.Module):
    def __init__(self, config: LateFusionConfig = None, **kwargs):
        super().__init__()
        if config is None:
            config = LateFusionConfig(**kwargs)
        self.config = config
        
        self.window_size = config.window_size
        self.vocab_size = config.vocab_size
        self.hidden_dim = config.hidden_dim
        self.max_len = config.max_len
        
        # Pure Text Encoder
        self.char_embed = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.pos_embed = mx.random.normal((1, config.max_len, config.hidden_dim)) * 0.02
        
        self.blocks = [CausalTransformerBlock(config.hidden_dim, config.num_heads) for _ in range(config.num_layers)]
        self.final_ln = nn.LayerNorm(config.hidden_dim, affine=False)
        
        # Late Fusion MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
        
        # Vocab Output
        self.fc = nn.Linear(config.hidden_dim, config.vocab_size)
        
    def get_config(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self.config)

    def _create_causal_mask(self, L: int, dtype=mx.float32) -> mx.array:
        indices = mx.arange(L)
        causal_mask = indices.reshape(L, 1) < indices.reshape(1, L)
        if self.window_size is not None:
            window_mask = indices.reshape(1, L) <= (indices.reshape(L, 1) - self.window_size)
            mask = causal_mask | window_mask
        else:
            mask = causal_mask
        return mask.astype(dtype) * -1e9
        
    def __call__(self, target_tokens: mx.array, target_z: mx.array) -> mx.array:
        # target_tokens: [B, L], target_z: [B, L, D]
        # target_z contains the target continuous vectors aligned with the output tokens
        B, L = target_tokens.shape
        L = L - 1 # shifted
        
        x = target_tokens[:, :-1]
        x = self.char_embed(x) + self.pos_embed[:, :L, :]
        
        mask = self._create_causal_mask(L, dtype=x.dtype)
        
        # Pure Causal Text Processing
        for block in self.blocks:
            x = block(x, mask)
            
        h_text = self.final_ln(x)
        
        # Late Fusion
        # Concatenate text history (h_text) with target continuous concept (target_z)
        fused = mx.concatenate([h_text, target_z], axis=-1)
        fused_out = self.fusion_mlp(fused)
        
        logits = self.fc(fused_out)
        return logits

    def generate(self, encoder_out: mx.array, bos_token_id: int, temperature: float = 0.0) -> mx.array:
        # encoder_out: [B, SeqLen, D]
        B, SeqLen, _ = encoder_out.shape
        current_tokens = mx.full((B, 1), bos_token_id, dtype=mx.int32)
        generated_tokens = []
        
        # Note: We need a KV cache for efficient generation. For now we just run the full sequence autoregressively.
        # It's O(L^2) but it's just for evaluation of 1 sequence.
        
        for t in range(SeqLen - 1):
            # 1. Process text history
            x = self.char_embed(current_tokens) + self.pos_embed[:, :t+1, :]
            mask = self._create_causal_mask(t+1, dtype=x.dtype)
            
            for block in self.blocks:
                x = block(x, mask)
                
            h_text = self.final_ln(x)
            
            # We only care about the last time step for prediction
            h_t = h_text[:, -1:, :] # [B, 1, D]
            z_target = encoder_out[:, t+1:t+2, :] # [B, 1, D], the target concept for t+1
            
            # Late Fusion
            fused = mx.concatenate([h_t, z_target], axis=-1)
            fused_out = self.fusion_mlp(fused)
            logits = self.fc(fused_out) # [B, 1, V]
            
            if temperature > 0:
                probs = mx.softmax(logits[:, -1, :] / temperature, axis=-1)
                next_token = mx.random.categorical(mx.log(probs)).reshape(B, 1)
            else:
                next_token = mx.argmax(logits[:, -1, :], axis=-1).reshape(B, 1)
                
            generated_tokens.append(next_token)
            current_tokens = mx.concatenate([current_tokens, next_token], axis=1)
            
        return current_tokens

