import os
import argparse
import torch
import mlx.core as mx
from safetensors.torch import load_file as load_pt_safetensors
from safetensors.torch import save_file as save_pt_safetensors

def _pt_to_mlx_key(pt_key: str) -> str:
    """Mapping PyTorch keys to MLX keys."""
    # 1. TinyCharEncoder Forward (Wrapping attention)
    if "layers" in pt_key and "attention" not in pt_key:
        if "q_proj" in pt_key: pt_key = pt_key.replace("q_proj", "attention.q_proj")
        elif "k_proj" in pt_key: pt_key = pt_key.replace("k_proj", "attention.k_proj")
        elif "v_proj" in pt_key: pt_key = pt_key.replace("v_proj", "attention.v_proj")
        elif "out_proj" in pt_key: pt_key = pt_key.replace("out_proj", "attention.out_proj")
        
    # 2. Legacy PyTorch Models Remapping -> MLX (e.g. distilled_emb/model_cuda.py uses fc1/fc2)
    k = pt_key
    if ".fc1." in k: k = k.replace(".fc1.", ".net.layers.0.")
    elif ".fc2." in k: k = k.replace(".fc2.", ".net.layers.2.")
    elif k.startswith("fc1."): k = k.replace("fc1.", "net.layers.0.")
    elif k.startswith("fc2."): k = k.replace("fc2.", "net.layers.2.")
    return k

def _mlx_to_pt_key(mlx_key: str) -> str:
    """Mapping MLX keys back to PyTorch keys."""
    k = mlx_key
    # 1. TinyCharEncoder Reverse (Unwrapping attention)
    if "attention.q_proj" in k: return k.replace("attention.q_proj", "q_proj")
    if "attention.k_proj" in k: return k.replace("attention.k_proj", "k_proj")
    if "attention.v_proj" in k: return k.replace("attention.v_proj", "v_proj")
    if "attention.out_proj" in k: return k.replace("attention.out_proj", "out_proj")
        
    # We NO LONGER force-remap GodEncoder/WeakDecoder to fc1/wq here, 
    # because the new CUDA models (model/*_cuda.py) natively use MLX-compatible layouts 
    # (net.layers.0 / query_proj). If loading into legacy models like distilled_emb/model_cuda.py,
    # those models handle compatibility via their own custom load_state_dict logics.
    return k

def convert_pt_to_mlx(input_path: str, output_path: str):
    print(f"🔄 [PT -> MLX] Converting PyTorch weights to Apple MLX format...")
    mlx_state = {}
    
    if input_path.endswith(('.pt', '.pth', '.bin')):
        pt_state = torch.load(input_path, map_location="cpu")
        for k, v in pt_state.items():
            # Safeguard against optimizer states saved in .pt files
            if not isinstance(v, torch.Tensor): continue
            new_k = _pt_to_mlx_key(k)
            
            # Safely handle bfloat16 without numpy crash
            if v.dtype == torch.bfloat16:
                v_int16 = v.view(torch.int16).numpy()
                mlx_state[new_k] = mx.array(v_int16).view(mx.bfloat16)
            else:
                mlx_state[new_k] = mx.array(v.numpy())
    else:
        # For .safetensors, directly load natively via MLX, skipping PyTorch overhead
        pt_state = mx.load(input_path)
        for k, v in pt_state.items():
            new_k = _pt_to_mlx_key(k)
            mlx_state[new_k] = v
            
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    mx.save_safetensors(output_path, mlx_state)
    print(f"✅ Success! Mapped {len(mlx_state)} tensors. Saved to {output_path}")

def convert_mlx_to_pt(input_path: str, output_path: str):
    print(f"🔄 [MLX -> PT] Converting Apple MLX weights to PyTorch safetensors...")
    # Because MLX safetensors is fully standard, we can load it straight into PyTorch CPU tensors
    mlx_state = load_pt_safetensors(input_path)
    
    pt_state = {}
    for k, v in mlx_state.items():
        new_k = _mlx_to_pt_key(k)
        pt_state[new_k] = v
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_pt_safetensors(pt_state, output_path)
    print(f"✅ Success! Mapped {len(pt_state)} tensors. Saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bi-directional Weight Converter for PyTorch <-> Apple MLX")
    parser.add_argument("direction", choices=["pt2mlx", "mlx2pt"], help="Direction of conversion")
    parser.add_argument("--input", type=str, required=True, help="Input weight file path")
    parser.add_argument("--output", type=str, required=True, help="Output weight file path")
    
    args = parser.parse_args()
    if args.direction == "pt2mlx":
        convert_pt_to_mlx(args.input, args.output)
    elif args.direction == "mlx2pt":
        convert_mlx_to_pt(args.input, args.output)
