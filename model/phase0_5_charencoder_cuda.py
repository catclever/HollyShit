import torch
import torch.nn as nn

class TinyCharEncoderCUDA(nn.Module):
    """
    The PyTorch Clone of the Teacher-Distilled Character Flow Encoder.
    Completely refactored for native CUDA performance, utilizing pure PyTorch primitives.
    """
    def __init__(self, vocab_size: int, d_model: int = 1024, n_heads: int = 8, n_layers: int = 6, max_seq_len: int = 512, z_dim: int = 1024):
        super().__init__()
        
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([RoPETransformerLayerCUDA(d_model, n_heads) for _ in range(n_layers)])
        
        self.final_ln = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, z_dim)
        
        # Better initialization for embedding table
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=d_model**-0.5)

    def forward(self, x, attention_mask=None):
        h = self.tok_emb(x)
        
        for layer in self.layers:
            h = layer(h, attention_mask)
            
        h = self.final_ln(h)
        
        if attention_mask is not None:
            # attention_mask: (B, L) where 1 is valid, 0 is pad
            # Safely mask out padding without risking 0 * NaN = NaN propagation
            h_masked = torch.where(attention_mask.unsqueeze(-1) > 0, h, torch.zeros_like(h))
            
            # Safe avg pool
            sum_mask = attention_mask.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            h_pool = h_masked.sum(dim=1) / sum_mask
        else:
            h_pool = h.mean(dim=1)
            
        return self.out_proj(h_pool)

def load_mlx_safetensors_into_torch(torch_module, safetensor_path):
    """
    Seamlessly loads an MLX-generated safetensors file into a PyTorch nn.Module.
    MLX and PyTorch nn.Linear weight shapes identically map to (out_features, in_features).
    """
    from safetensors.torch import load_file
    state_dict = load_file(safetensor_path)
    target_keys = set(torch_module.state_dict().keys())
    remapped = {}
    for k, v in state_dict.items():
        new_k = k
        new_k = new_k.replace(".net.layers.0.", ".fc1.")
        new_k = new_k.replace(".net.layers.2.", ".fc2.")
        new_k = new_k.replace("net.layers.0.", "fc1.")
        new_k = new_k.replace("net.layers.2.", "fc2.")
        new_k = new_k.replace(".norm.", ".norm.") # Unchanged, just to be explicit
        remapped[new_k] = v
    state_dict = remapped
    # Add support for WeakDecoder if any key needs remapping (it perfectly matches)
    for k in list(state_dict.keys()):
        if k.endswith('.wq.weight'):
            state_dict[k.replace('.wq.', '.query_proj.')] = state_dict.pop(k)
        elif k.endswith('.wk.weight'):
            state_dict[k.replace('.wk.', '.key_proj.')] = state_dict.pop(k)
        elif k.endswith('.wv.weight'):
            state_dict[k.replace('.wv.', '.value_proj.')] = state_dict.pop(k)
        elif k.endswith('.wo.weight'):
            state_dict[k.replace('.wo.', '.out_proj.')] = state_dict.pop(k)

