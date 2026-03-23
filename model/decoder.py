import mlx.core as mx
import mlx.nn as nn

class WeakDecoder(nn.Module):
    """
    The Micro Camp: responsible ONLY for translating a semantic coordinate
    (z_target) directly into a sequence of tokens.
    
    Gradient from this module must NOT flow back into the Macro camp.
    """
    def __init__(self, z_dim: int, vocab_size: int, d_model: int = 128, n_layers: int = 2):
        super().__init__()
        self.z_dim = z_dim
        self.vocab_size = vocab_size
        
        # Project spatial target into decoder's internal dimension
        self.z_proj = nn.Linear(z_dim, d_model)
        
        # A very shallow transformer decoder
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # We use TransformerEncoder with a causal mask to act as a decoder
        self.transformer = nn.TransformerEncoder(
            num_layers=n_layers, 
            dims=d_model, 
            num_heads=4, 
            mlp_dims=d_model * 4
        )
        self.out_proj = nn.Linear(d_model, vocab_size)

    def __call__(self, z_target: mx.array, token_inputs: mx.array):
        """
        z_target: (Batch, z_dim) - The exact target semantic anchor for the current clause.
        token_inputs: (Batch, SeqLen) - Shifted right tokens for teacher forcing.
        
        Returns:
            logits: (Batch, SeqLen, vocab_size)
        """
        # --- ENDOGENOUS FEEDBACK (No Stop Gradient) ---
        # The decoder acts as the ultimate judge. Its reconstruction loss
        # must flow freely back to the God Encoder to shape the spatial layout.
        z_projected = self.z_proj(z_target) # (Batch, d_model)
        
        # Embed the tokens
        x = self.embedding(token_inputs) # (Batch, SeqLen, d_model)
        
        # Inject the semantic space coordinate into every token 
        # (conditioning the generation on the macro point)
        x = x + z_projected[:, None, :] 
        
        # Autoregressive causal mask
        seq_len = x.shape[1]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
        
        # Pass through shallow transformer
        x = self.transformer(x, mask)
        
        # Project to vocabulary
        logits = self.out_proj(x)
        return logits

    def generate(self, z_target: mx.array, start_token: int, eos_token: int = None, max_tokens: int = 50, temperature: float = 0.7):
        """
        Auto-regressive generation for inference from a purely spatial coordinate.
        z_target: (1, z_dim)
        """
        z_projected = self.z_proj(z_target) # (1, d_model)
        
        # We hold the sequence of generated tokens (Batch=1)
        tokens = mx.array([[start_token]]) 
        result_tokens = [start_token]
        
        for i in range(max_tokens):
            x = self.embedding(tokens) # (1, seq_len, d_model)
            
            # Inject spatial macro point as static bias
            x = x + z_projected[:, None, :]
            
            # Causal mask for autoregressive step
            seq_len = x.shape[1]
            mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
            
            # Shallow forward pass
            x = self.transformer(x, mask)
            
            # Get logits of the newly predicted LAST token
            logits = self.out_proj(x[:, -1, :]) # (1, vocab_size)
            
            if temperature == 0:
                next_token = mx.argmax(logits, axis=-1).item()
            else:
                # Stochastic sampling based on temperature
                next_token = mx.random.categorical(logits * (1.0 / temperature)).item()
                
            result_tokens.append(next_token)
            
            # If the model explicitly signals end of text, cleanly stop predicting garbage
            if eos_token is not None and next_token == eos_token:
                break
            
            # Append to the sequence for the next timestep
            tokens = mx.array([result_tokens])
            
        return result_tokens
