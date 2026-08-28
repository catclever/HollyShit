import mlx.core as mx
import mlx.nn as nn
import os

def load_torch_weights_into_mlx(mlx_model: nn.Module, torch_ckpt_path: str, strict: bool = False):
    """
    Generic utility to load PyTorch saved weights into an MLX model.
    Handles standard Linear, LayerNorm, and Embedding mappings.
    """
    import torch
    if not os.path.exists(torch_ckpt_path):
        raise FileNotFoundError(f"PyTorch checkpoint not found at {torch_ckpt_path}")
        
    state_dict = torch.load(torch_ckpt_path, map_location='cpu', weights_only=True)
    
    mlx_state_list = []
    for k, v in state_dict.items():
        if "inv_freq" in k or "cos_cached" in k or "sin_cached" in k:
            continue
        v_np = v.numpy()
        mlx_state_list.append((k, mx.array(v_np)))
        
    import mlx.utils
    tree = mlx.utils.tree_unflatten(mlx_state_list)
    
    # Attempt to update
    try:
        mlx_model.update(tree)
        mx.eval(mlx_model.parameters())
        print(f"[WeightUtils] Successfully loaded PyTorch weights into MLX model from {torch_ckpt_path}")
    except Exception as e:
        print(f"[WeightUtils] Error loading weights: {e}")
        if strict:
            raise e

def load_mlx_safetensors_into_torch(torch_module, safetensor_path: str):
    """
    Seamlessly loads an MLX-generated safetensors file into a PyTorch nn.Module.
    Handles standard key remappings for QKV and FFN layers if needed.
    """
    from safetensors.torch import load_file
    if not os.path.exists(safetensor_path):
        raise FileNotFoundError(f"MLX safetensors not found at {safetensor_path}")
        
    state_dict = load_file(safetensor_path)
    target_keys = set(torch_module.state_dict().keys())
    
    remapped = {}
    for k, v in state_dict.items():
        new_k = k
        
        # QKV Attention variants
        new_k = new_k.replace('.wq.', '.query_proj.')
        new_k = new_k.replace('.wk.', '.key_proj.')
        new_k = new_k.replace('.wv.', '.value_proj.')
        new_k = new_k.replace('.wo.', '.out_proj.')
        
        # If the direct match isn't found, try common alternatives
        if new_k not in target_keys:
            test_k1 = new_k.replace(".net.layers.0.", ".fc1.").replace(".net.layers.2.", ".fc2.")
            test_k1 = test_k1.replace("net.layers.0.", "fc1.").replace("net.layers.2.", "fc2.")
            test_k2 = new_k.replace(".net.layers.0.", ".linear1.").replace(".net.layers.2.", ".linear2.")
            
            if test_k1 in target_keys:
                new_k = test_k1
            elif test_k2 in target_keys:
                new_k = test_k2
                
        remapped[new_k] = v
        
    missing = target_keys - set(remapped.keys())
    if len(missing) > 0:
        print(f"[WeightUtils] [Warning] Missing keys when loading {safetensor_path} into {torch_module.__class__.__name__}:")
        for m in list(missing)[:10]:
            print(f"  - {m}")
        if len(missing) > 10:
            print(f"  ... and {len(missing)-10} more.")
            
    torch_module.load_state_dict(remapped, strict=False)
    print(f"[WeightUtils] Successfully loaded MLX safetensors into PyTorch model from {safetensor_path}")
