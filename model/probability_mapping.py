import mlx.core as mx
import mlx.nn as nn

class ProbabilityMappingLayer(nn.Module):
    """
    Independent layer that maps Mamba's deterministic hidden state (h_t) 
    into a probability net (μ, log_var) to evaluate spatial layout uncertainty.
    """
    def __init__(self, d_model: int, z_dim: int):
        super().__init__()
        self.d_model = d_model
        self.z_dim = z_dim
        
        # Intermediate reasoning mapping
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
        )
        
        self.mu_proj = nn.Linear(d_model, z_dim)
        self.logvar_proj = nn.Linear(d_model, z_dim)
        
    def __call__(self, h_t: mx.array):
        """
        h_t: (B, L, d_model) - Mamba's dynamic momentum state
        
        Returns:
            mu_net: (B, L, z_dim) - Center of the probability net
            logvar_net: (B, L, z_dim) - Log-variance of the probability net
        """
        h_mapped = self.mlp(h_t)
        mu_net = self.mu_proj(h_mapped)
        logvar_net = self.logvar_proj(h_mapped)
        
        return mu_net, logvar_net
