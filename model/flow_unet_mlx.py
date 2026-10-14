import mlx.core as mx
import mlx.nn as nn
import math

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_dim, frequency_embedding_size=256):
        super().__init__()
        self.freq_dim = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def __call__(self, t: mx.array) -> mx.array:
        half_dim = self.freq_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = mx.exp(mx.arange(half_dim, dtype=mx.float32) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=-1)
        return self.mlp(emb)

class AdaLN(nn.Module):
    def __init__(self, hidden_dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, affine=False)
        self.linear = nn.Linear(cond_dim, 2 * hidden_dim)

    def __call__(self, x: mx.array, h_potential: mx.array) -> mx.array:
        x_norm = self.norm(x)
        c = self.linear(h_potential)
        shift, scale = mx.split(c, 2, axis=-1)
        return x_norm * (1.0 + scale[:, None, :]) + shift[:, None, :]

class OrthogonalAttractor(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.norm_x = nn.LayerNorm(hidden_dim)
        self.norm_k = nn.LayerNorm(hidden_dim)
        self.scale = 1.0 / math.sqrt(hidden_dim)

    def __call__(self, x: mx.array, k: mx.array) -> mx.array:
        x_norm = self.norm_x(x)
        k_norm = self.norm_k(k)
        # x_norm: (B, L_x, D), k_norm: (B, L_k, D)
        # inner product: x_norm @ k_norm^T -> (B, L_x, L_k)
        k_norm_t = mx.transpose(k_norm, (0, 2, 1))
        activation = mx.matmul(x_norm, k_norm_t) * self.scale
        activation = mx.maximum(activation, mx.array(0.0)) # ReLU
        
        # force: activation @ k -> (B, L_x, D)
        force = mx.matmul(activation, k)
        return force

class Gated1DResBlock(nn.Module):
    def __init__(self, hidden_dim: int, cond_dim: int, kernel_size: int = 5, dilation: int = 1):
        super().__init__()
        
        self.time_proj = nn.Linear(hidden_dim, hidden_dim)
        self.adaLN1 = AdaLN(hidden_dim, cond_dim)
        self.attractor = OrthogonalAttractor(hidden_dim)
        
        # MLX Conv1d expects (N, H, W, C) -> (B, L, D) natively, no need to transpose like PyTorch
        padding = ((kernel_size - 1) * dilation) // 2
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding, dilation=dilation)
        self.act1 = nn.SiLU()
        
        self.adaLN2 = AdaLN(hidden_dim, cond_dim)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding, dilation=dilation)
        
    def __call__(self, x: mx.array, t_emb: mx.array, h_potential: mx.array, k_attractors: mx.array = None) -> mx.array:
        residual = x
        
        t_spatial = mx.expand_dims(self.time_proj(t_emb), 1)
        x = x + t_spatial
        
        x = self.adaLN1(x, h_potential)
        
        if k_attractors is not None:
            force = self.attractor(x, k_attractors)
            x = x + force
            
        # MLX Conv1d natively takes (Batch, Length, Channels)
        x = self.conv1(x)
        x = self.act1(x)
        
        x = self.adaLN2(x, h_potential)
        x = self.conv2(x)
        
        return residual + x

class FlowMatching1DUnet(nn.Module):
    def __init__(self, hidden_dim: int = 1024, cond_dim: int = 1024, num_layers: int = 6):
        super().__init__()
        self.t_embedder = TimestepEmbedder(hidden_dim)
        
        self.blocks = [
            Gated1DResBlock(
                hidden_dim=hidden_dim, 
                cond_dim=cond_dim, 
                kernel_size=5, 
                dilation=2**(i % 4)
            ) for i in range(num_layers)
        ]
        
        self.final_norm = AdaLN(hidden_dim, cond_dim)
        self.final_proj = nn.Linear(hidden_dim, hidden_dim)

    def __call__(self, x_t: mx.array, t: mx.array, h_potential: mx.array, k_attractors: mx.array = None) -> mx.array:
        t_emb = self.t_embedder(t)
        
        x = x_t
        for block in self.blocks:
            x = block(x, t_emb, h_potential, k_attractors)
            
        x = self.final_norm(x, h_potential)
        v_pred = self.final_proj(x)
        
        return v_pred
