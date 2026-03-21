import mlx.core as mx
import mlx.nn as nn

class AdapterLayer(nn.Module):
    """
    Adapter layer to map from external pre-trained embeddings (e.g. BGE, OpenAI)
    into the internal model continuous semantic space.
    """
    def __init__(self, input_dim: int, d_model: int):
        super().__init__()
        # A simple linear projection. 
        # For a more complex adapter, this could be an MLP.
        self.proj = nn.Linear(input_dim, d_model)
        
    def __call__(self, x: mx.array) -> mx.array:
        """
        x: (batch_size, seq_len, input_dim)
        returns: (batch_size, seq_len, d_model)
        """
        return self.proj(x)
