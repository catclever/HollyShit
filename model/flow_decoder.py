import mlx.core as mx
import mlx.nn as nn
import math

class FlowTimeEmbedding(nn.Module):
    """
    Standard Sinusoidal Time Embedding for Flow Matching / Diffusion.
    Projects scalar time t in [0, 1] to a high-dimensional vector.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim)
        )

    def __call__(self, t: mx.array):
        # t shape: (B, 1) or (B,)
        # sinusoidal encoding
        half_dim = self.dim // 2
        
        # Calculate frequencies: e^(-log(10000) * i / half_dim)
        freqs = mx.exp(
            -math.log(10000.0) * mx.arange(half_dim, dtype=mx.float32) / half_dim
        )
        
        # Compute angles
        args = t * freqs
        
        # Concatenate sin and cos
        embedding = mx.concatenate([mx.sin(args), mx.cos(args)], axis=-1)
        
        # If dim is odd, pad with zeros to match target dim exactly
        if self.dim % 2 != 0:
            embedding = mx.pad(embedding, [(0, 0), (0, 1)])
            
        # Pass through MLP
        return self.mlp(embedding)


class CharPositionalEncoding(nn.Module):
    """
    Absolute Positional Encoding for the Sequence.
    Gives the continuous noises a spatial coordinate system.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def __call__(self, x: mx.array):
        # x shape: (B, L, d_model)
        L = x.shape[1]
        half_dim = self.dim // 2
        
        # position indices: (L,)
        pos = mx.arange(L, dtype=mx.float32)
        
        # frequencies
        freqs = mx.exp(-math.log(10000.0) * mx.arange(half_dim, dtype=mx.float32) / half_dim)
        
        # (L, half_dim)
        args = pos[:, None] * freqs[None, :]
        
        # concatenate sin and cos -> (L, dim)
        pe = mx.concatenate([mx.sin(args), mx.cos(args)], axis=-1)
        
        if self.dim % 2 != 0:
            pe = mx.pad(pe, [(0, 0), (0, 1)])
            
        # Broadcast add to x: (B, L, d_model) + (1, L, d_model)
        return x + pe[None, :, :]


class FlowDecoder(nn.Module):
    """
    Continuous Vector Field Network for Optimal Transport Flow Matching.
    Replaces the Autoregressive WeakDecoder.
    
    Inputs:
    - x_t: (B, L, d_model) - The noisy character embeddings at time t
    - t: (B, 1) - The continuous time step [0.0, 1.0]
    - z_target: (B, d_model) - The absolute semantic anchor from GodEncoder
    
    Output:
    - v_pred: (B, L, d_model) - The predicted velocity vector for each character
    """
    def __init__(self, z_dim: int, d_model: int, vocab_size: int, n_layers: int = 4, n_heads: int = 8):
        super().__init__()
        self.d_model = d_model
        
        # 1. The Continuous Universe's Blueprint (The Target Embedding Space)
        # We need this to define the ultimate ground truth destination for the flow
        self.char_embedding = nn.Embedding(vocab_size, d_model)
        
        # 2. Time & Semantic Conditioning Projections
        self.time_embed = FlowTimeEmbedding(d_model)
        self.z_proj = nn.Linear(z_dim, d_model)
        
        # Absolute Positional Encoding for the flow field
        self.pos_embed = CharPositionalEncoding(d_model)
        
        # 3. The Core Wind Tunnel (Bidirectional Non-Causal Transformer)
        # Because Flow Matching sees all noise at once, it doesn't need to be causal!
        self.transformer = nn.TransformerEncoder(
            num_layers=n_layers,
            dims=d_model,
            num_heads=n_heads,
            mlp_dims=d_model * 4
        )
        
        # 4. Out-Projection to Velocity
        # The output is NOT logits over 8000 chars! It's a continuous velocity vector [d_model]!
        self.velocity_head = nn.Linear(d_model, d_model)

    def __call__(self, x_t: mx.array, t: mx.array, z_target: mx.array, mask: mx.array = None):
        """
        Calculates the velocity field v_pred for the ODE flow.
        """
        B, L, _ = x_t.shape
        
        # 1. Embed Time and Z
        # t is (B, 1), t_emb becomes (B, d_model)
        t_emb = self.time_embed(t)
        
        # z_target is (B, z_dim), z_emb becomes (B, d_model)
        z_emb = self.z_proj(z_target)
        
        # 2. Condition the input flow
        # We broadcast add the Time and Z context globally to every position in the sequence
        # (B, 1, d_model) broadcasted over length L
        condition = mx.expand_dims(t_emb + z_emb, 1) 
        
        x_conditioned = x_t + condition
        
        # Add absolute positional encodings to give the noise a spatial direction
        x_conditioned = self.pos_embed(x_conditioned)
        
        # 3. Transformer Processing (Fully bidirectional, letting particles "see" each other's noise)
        # Convert (B, L) padding mask → additive attention mask for MLX TransformerEncoder
        # MLX expects additive mask where 0 = attend, -inf = ignore
        attn_mask = None
        if mask is not None:
            # mask shape: (B, L), values 1=valid 0=pad
            # → (B, 1, 1, L) so it broadcasts against (B, heads, L, L)
            attn_mask = mx.where(
                mask[:, None, None, :] > 0,
                mx.zeros_like(mask[:, None, None, :]),
                mx.full(mask[:, None, None, :].shape, float('-inf'))
            )
        memory = self.transformer(x_conditioned, mask=attn_mask)
        
        # 4. Extract continuous velocity
        # Shape: (B, L, d_model)
        v_pred = self.velocity_head(memory)
        
        return v_pred

    def embed_text(self, token_ids: mx.array):
        """
        Converts text tokens to their continuous Euclidean coordinates.
        This provides x_1 (the physical destination) for Flow Matching.
        Removed standard sqrt(d_model) scaling to avoid flow scale mismatch.
        """
        return self.char_embedding(token_ids)
        
    def generate_euler(self, z_target: mx.array, target_length: int, steps: int = 20):
        """
        Inference / Decoding: Solves the Flow Differential Equation (Euler Method).
        The 'Cheat' parameter target_length tells it exactly how much noise to sample!
        """
        B = z_target.shape[0]
        
        # Start at absolute chaos (t=0)
        # Sample x_0 from standard Normal distribution N(0, 1)
        x = mx.random.normal(shape=(B, target_length, self.d_model))
        
        dt = 1.0 / steps
        
        # Integrate from t=0.0 to t=1.0
        for step in range(steps):
            # t must be shape (B, 1)
            t = mx.full((B, 1), step * dt, dtype=mx.float32)
            
            # Predict the velocity field
            v = self(x, t, z_target)
            
            # Euler step: x_{t+dt} = x_t + v_t * dt
            x = x + v * dt
            
        # At t=1.0, x should perfectly approximate the char_embedding of the target sentence!
        # Final trick: Snap to the nearest discrete token in the dictionary
        # Compute L2 distance or Cosine Similarity to self.char_embedding.weight
        
        emb_bank = self.char_embedding.weight
        
        # Simple Euclidean Distance Argmin (Broadcasting)
        # Distance^2 = (A-B)^2 = A^2 + B^2 - 2AB
        x_sq = mx.sum(mx.square(x), axis=-1, keepdims=True)            # (B, L, 1)
        emb_sq = mx.sum(mx.square(emb_bank), axis=-1).reshape(1, 1, -1) # (1, 1, V)
        dot_prod = mx.matmul(x, emb_bank.T)                            # (B, L, V)
        
        distances = x_sq + emb_sq - 2 * dot_prod                       # (B, L, V)
        
        # Select the vocabulary ID with the smallest distance
        # Shape: (B, L)
        final_token_ids = mx.argmin(distances, axis=-1)
        
        return final_token_ids
