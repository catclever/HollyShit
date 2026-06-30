import mlx.core as mx
import mlx.nn as nn

class VAEBottleneck(nn.Module):
    """
    Implements the variational reparameterization trick.
    Used for sampling an absolute coordinate z from the distribution μ, σ.
    
    During training (Phase 1), the micro decoder directly uses Target Emb,
    so this bottleneck is primarily used when evaluating / generating new text
    or exploring the spatial clusters of semantics.
    """
    def __init__(self):
        super().__init__()
        
    def __call__(self, mu: mx.array, logvar: mx.array, sample: bool = True):
        """
        mu: (B, L, z_dim) center of distribution
        logvar: (B, L, z_dim) log-variance of distribution
        sample: If true, sample stochastically. If false, return mode (mu).
        
        Returns:
            z: Sampled coordinate
            std: The computed standard deviation (for loss calculation reference)
        """
        std = mx.exp(0.5 * logvar)
        
        if sample:
            # Reparameterization trick: z = mu + eps * std
            eps = mx.random.normal(shape=std.shape)
            z = mu + eps * std
            return z, std
            
        return mu, std
