import mlx.core as mx
import mlx.nn as nn
import math

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.head_dim = d_model // n_heads
        
        self.query_proj = nn.Linear(d_model, d_model, bias=False)
        self.key_proj = nn.Linear(d_model, d_model, bias=False)
        self.value_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def __call__(self, x: mx.array, memory: mx.array, mask: mx.array = None) -> mx.array:
        B, L_q, _ = x.shape
        _, L_k, _ = memory.shape
        
        q = self.query_proj(x).reshape(B, L_q, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.key_proj(memory).reshape(B, L_k, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.value_proj(memory).reshape(B, L_k, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

        scores = mx.matmul(q, k.transpose(0, 1, 3, 2)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
            
        attn = mx.softmax(scores, axis=-1)
        out = mx.matmul(attn, v)
        out = out.transpose(0, 2, 1, 3).reshape(B, L_q, self.d_model)
        return self.out_proj(out)

class TransformerDecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.self_attn = nn.MultiHeadAttention(d_model, n_heads)
        self.cross_attn = MultiHeadCrossAttention(d_model, n_heads)
        
        self.ln1 = nn.LayerNorm(d_model, affine=False)
        self.ln2 = nn.LayerNorm(d_model, affine=False)
        self.ln3 = nn.LayerNorm(d_model, affine=False)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4, bias=False),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model, bias=False)
        )

    def __call__(self, x: mx.array, memory: mx.array, causal_mask: mx.array = None) -> mx.array:
        # 1. Causal Self-Attention
        x = x + self.self_attn(self.ln1(x), self.ln1(x), self.ln1(x), causal_mask)
        # 2. Cross-Attention
        x = x + self.cross_attn(self.ln2(x), memory, mask=None)
        # 3. FFN
        x = x + self.ffn(self.ln3(x))
        return x

from dataclasses import dataclass

@dataclass
class StandardDecoderConfig:
    vocab_size: int = 32000
    embed_dim: int = 1024
    hidden_dim: int = 1024
    num_layers: int = 4
    num_heads: int = 8
    max_len: int = 512

class StandardCrossAttnDecoder(nn.Module):
    \"\"\"
    Phase 3.1.3: The textbook classic Transformer Decoder.
    No pointer networks, no residual injection. Just pure cross-attention.
    \"\"\"
    def __init__(self, config: StandardDecoderConfig = None, **kwargs):
        super().__init__()
        if config is None:
            config = StandardDecoderConfig(**kwargs)
        self.config = config
        
        self.vocab_size = config.vocab_size
        self.char_embed = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.pos_embed = mx.random.normal((1, config.max_len, config.hidden_dim)) * 0.02
        
        self.concept_proj = nn.Linear(config.embed_dim, config.hidden_dim) if config.embed_dim != config.hidden_dim else None
        
        self.blocks = [TransformerDecoderBlock(config.hidden_dim, config.num_heads) for _ in range(config.num_layers)]
        
        self.final_ln = nn.LayerNorm(config.hidden_dim, affine=False)
        self.fc = nn.Linear(config.hidden_dim, config.vocab_size)

    def get_config(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self.config)

    def _create_causal_mask(self, L: int, dtype: mx.Dtype) -> mx.array:
        indices = mx.arange(L)
        mask = indices.reshape(-1, 1) < indices.reshape(1, -1)
        return mask.astype(dtype) * -1e9
        
    def __call__(self, concept_sequence_zoh: mx.array, target_tokens: mx.array) -> mx.array:
        B, L = target_tokens.shape
        L = L - 1
        
        x = target_tokens[:, :-1]
        x_embed = self.char_embed(x) + self.pos_embed[:, :L, :]
        
        if self.concept_proj is not None:
            memory = self.concept_proj(concept_sequence_zoh)
        else:
            memory = concept_sequence_zoh
            
        mask = self._create_causal_mask(L, dtype=x_embed.dtype)
        
        x = x_embed
        for block in self.blocks:
            x = block(x, memory, causal_mask=mask)
            
        x_final = self.final_ln(x)
        logits = self.fc(x_final)
        
        return logits

    def generate(self, concept_sequence_zoh: mx.array, bos_token_id: int, max_len: int = 64, temperature: float = 0.8, repetition_penalty: float = 1.2) -> mx.array:
        B = concept_sequence_zoh.shape[0]
        current_tokens = mx.full((B, 1), bos_token_id, dtype=mx.int32)
        generated_tokens = []
        
        if self.concept_proj is not None:
            memory = self.concept_proj(concept_sequence_zoh)
        else:
            memory = concept_sequence_zoh
            
        for step in range(max_len):
            L = current_tokens.shape[1]
            x_embed = self.char_embed(current_tokens) + self.pos_embed[:, :L, :]
            mask = self._create_causal_mask(L, dtype=x_embed.dtype)
            
            x = x_embed
            for block in self.blocks:
                x = block(x, memory, causal_mask=mask)
                
            x_final = self.final_ln(x[:, -1:, :])
            logits_vocab = self.fc(x_final)
            next_token_logits = logits_vocab
                
            if repetition_penalty != 1.0 and len(generated_tokens) > 0:
                past_tokens = mx.concatenate(generated_tokens, axis=1)
                for b in range(B):
                    b_past = past_tokens[b]
                    
                    import numpy as np
                    b_past_np = np.unique(np.array(b_past))
                    mask_np = np.zeros(next_token_logits.shape[-1], dtype=bool)
                    mask_np[b_past_np] = True
                    mask_mx = mx.array(mask_np)
                    
                    logits_b = next_token_logits[b, 0]
                    penalties = mx.where(logits_b > 0, 1.0 / repetition_penalty, repetition_penalty)
                    logits_b = mx.where(mask_mx, logits_b * penalties, logits_b)
                    next_token_logits[b, 0] = logits_b
                    
            next_token_logits = next_token_logits[:, -1, :]
            
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
                next_token = mx.random.categorical(next_token_logits)
                next_token = next_token.reshape(-1, 1)
            else:
                next_token = mx.argmax(next_token_logits, axis=-1, keepdims=True)
            
            generated_tokens.append(next_token)
            current_tokens = mx.concatenate([current_tokens, next_token], axis=1)
            
        return mx.concatenate(generated_tokens, axis=1)
