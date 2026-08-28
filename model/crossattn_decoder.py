import mlx.core as mx
import mlx.nn as nn
import math

class TransformerDecoderBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.self_attn = nn.MultiHeadAttention(hidden_dim, num_heads)
        
        self.ln1 = nn.LayerNorm(hidden_dim, affine=False)
        self.ln3 = nn.LayerNorm(hidden_dim, affine=False)
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
    def __call__(self, x, mask=None):
        nx = self.ln1(x)
        x = x + self.self_attn(nx, nx, nx, mask)
        
        nx3 = self.ln3(x)
        x = x + self.mlp(nx3)
        return x

from dataclasses import dataclass

@dataclass
class DecoderConfig:
    vocab_size: int = 32000
    embed_dim: int = 1024
    hidden_dim: int = 1024
    num_layers: int = 4
    num_heads: int = 8
    max_len: int = 512

class StrongConceptDecoder(nn.Module):
    """
    Phase 1.5 Decoder: Sentence-Level Short-Range Decoding.
    Decouples ZOH from token embeddings and uses Cross-Attention with a sliding window,
    acting as a true Encoder-Decoder Translation model.
    Also incorporates a Pointer-Generator for exact copying of anchor tokens.
    """
    def __init__(self, config: DecoderConfig = None, **kwargs):
        super().__init__()
        if config is None:
            config = DecoderConfig(**kwargs)
        self.config = config
        
        self.vocab_size = config.vocab_size
        self.char_embed = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.pos_embed = mx.random.normal((1, config.max_len, config.hidden_dim)) * 0.02
        
        self.concept_proj = nn.Linear(config.embed_dim, config.hidden_dim) if config.embed_dim != config.hidden_dim else None
        
        self.blocks = [TransformerDecoderBlock(config.hidden_dim, config.num_heads) for _ in range(config.num_layers)]
        
        self.final_ln = nn.LayerNorm(config.hidden_dim, affine=False)
        self.fc = nn.Linear(config.hidden_dim, config.vocab_size)
        
        # Dual Head for Pointer Network
        self.p_gen_proj = nn.Linear(config.hidden_dim, 1)

    def get_config(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self.config)

    def _create_causal_mask(self, L: int, dtype: mx.Dtype) -> mx.array:
        indices = mx.arange(L)
        mask = indices.reshape(-1, 1) < indices.reshape(1, -1)
        return mask.astype(dtype) * -1e9
        
    def __call__(self, concept_sequence_zoh: mx.array, target_tokens: mx.array, anchors_h: mx.array = None, anchors_tokens: mx.array = None) -> mx.array:
        B, L = target_tokens.shape
        L = L - 1 # shifted
        
        x = target_tokens[:, :-1]
        x_embed = self.char_embed(x) + self.pos_embed[:, :L, :]
        
        if self.concept_proj is not None:
            concept_sequence_zoh = self.concept_proj(concept_sequence_zoh)
            
        # ZOH channel: direct residual injection
        x = x_embed + concept_sequence_zoh
        
        mask = self._create_causal_mask(L, dtype=x.dtype)
        
        for block in self.blocks:
            x = block(x, mask)
            
        x_final = self.final_ln(x)
        
        # 1. Standard Vocab generation (Head 1)
        logits_vocab = self.fc(x_final) # [B, L, V]
        
        # 2. Pointer Network Copy mechanism (Head 2)
        if anchors_h is not None and anchors_tokens is not None:
            p_gen = mx.sigmoid(self.p_gen_proj(x_final)) # [B, L, 1]
            
            K = anchors_h.shape[1]
            anchors_proj = self.concept_proj(anchors_h) if self.concept_proj is not None else anchors_h
            attn_scores = mx.matmul(x_final, anchors_proj.transpose(0, 2, 1)) / (x_final.shape[-1] ** 0.5) # [B, L, K]
            
            pad_mask = (anchors_tokens == -1).reshape(B, 1, K)
            attn_scores = mx.where(pad_mask, -1e9, attn_scores)
            
            attn_dist = mx.softmax(attn_scores, axis=-1) # [B, L, K]
            
            probs_vocab = mx.softmax(logits_vocab, axis=-1)
            final_probs = p_gen * probs_vocab
            
            V = logits_vocab.shape[-1]
            valid_mask = anchors_tokens != -1
            safe_tokens = mx.where(valid_mask, anchors_tokens, 0)
            
            # Create one-hot mapping [B, K, V]
            one_hot = (safe_tokens[..., None] == mx.arange(V)).astype(final_probs.dtype)
            one_hot = mx.where(valid_mask[..., None], one_hot, 0.0)
            
            # Matrix multiply to scatter anchor probabilities [B, L, K] @ [B, K, V] -> [B, L, V]
            anchor_probs = mx.matmul(attn_dist, one_hot)
            
            final_probs = final_probs + (1 - p_gen) * anchor_probs
            
            final_probs = mx.maximum(final_probs, 1e-8)
            return mx.log(final_probs)
            
        return logits_vocab

    def generate(self, concept_sequence_zoh: mx.array, bos_token_id: int, max_len: int = 64, temperature: float = 0.8, repetition_penalty: float = 1.2, anchors_h: mx.array = None, anchors_tokens: mx.array = None) -> mx.array:
        B = concept_sequence_zoh.shape[0]
        current_tokens = mx.full((B, 1), bos_token_id, dtype=mx.int32)
        generated_tokens = []
        
        if self.concept_proj is not None:
            concept_sequence_zoh = self.concept_proj(concept_sequence_zoh)
            
        L_zoh = concept_sequence_zoh.shape[1]
        
        for step in range(max_len):
            L = current_tokens.shape[1]
            x_embed = self.char_embed(current_tokens) + self.pos_embed[:, :L, :]
            
            if L <= L_zoh:
                aligned_zoh = concept_sequence_zoh[:, :L, :]
            else:
                last_zoh = concept_sequence_zoh[:, -1:, :]
                padding_zoh = mx.repeat(last_zoh, L - L_zoh, axis=1)
                aligned_zoh = mx.concatenate([concept_sequence_zoh, padding_zoh], axis=1)
                
            x = x_embed + aligned_zoh
            mask = self._create_causal_mask(L, dtype=x.dtype)
            
            for block in self.blocks:
                x = block(x, mask)
                
            x_final = self.final_ln(x[:, -1:, :])
            logits_vocab = self.fc(x_final)
            
            if anchors_h is not None and anchors_tokens is not None:
                p_gen = mx.sigmoid(self.p_gen_proj(x_final))
                K = anchors_h.shape[1]
                anchors_proj = self.concept_proj(anchors_h) if self.concept_proj is not None else anchors_h
                attn_scores = mx.matmul(x_final, anchors_proj.transpose(0, 2, 1)) / (x_final.shape[-1] ** 0.5)
                
                pad_mask = (anchors_tokens == -1).reshape(B, 1, K)
                attn_scores = mx.where(pad_mask, -1e9, attn_scores)
                attn_dist = mx.softmax(attn_scores, axis=-1)
                
                probs_vocab = mx.softmax(logits_vocab, axis=-1)
                final_probs = p_gen * probs_vocab
                
                V = logits_vocab.shape[-1]
                valid_mask = anchors_tokens != -1
                safe_tokens = mx.where(valid_mask, anchors_tokens, 0)
                
                # Create one-hot mapping [B, K, V]
                one_hot = (safe_tokens[..., None] == mx.arange(V)).astype(final_probs.dtype)
                one_hot = mx.where(valid_mask[..., None], one_hot, 0.0)
                
                # Matrix multiply to scatter anchor probabilities [B, L, K] @ [B, K, V] -> [B, L, V]
                anchor_probs = mx.matmul(attn_dist, one_hot)
                
                final_probs = final_probs + (1 - p_gen) * anchor_probs
                            
                next_token_logits = mx.log(mx.maximum(final_probs, 1e-8))
            else:
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
                    
            # next_token_logits is [B, 1, V], squeeze the seq dim to [B, V]
            next_token_logits = next_token_logits[:, -1, :]
            
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
                next_token = mx.random.categorical(next_token_logits)  # [B]
                next_token = next_token.reshape(-1, 1)  # [B, 1]
            else:
                next_token = mx.argmax(next_token_logits, axis=-1, keepdims=True)  # [B, 1]
            
            generated_tokens.append(next_token)
            current_tokens = mx.concatenate([current_tokens, next_token], axis=1)
            
        return mx.concatenate(generated_tokens, axis=1)
