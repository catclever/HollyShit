import os
import sys

# [BUG FIX]
# (1) Specific Problem: Running this inference script on servers/macs via python command raises 
#     "ModuleNotFoundError: No module named 'training'" or similar because the workspace root is not in Python path.
# (2) Method to Resolve: Explicitly insert the parent directory of this script (workspace root) 
#     into the beginning of sys.path before importing local project packages (model, training, distilled_emb).
# (3) Caveats: Ensure any other newly created root-level script or nested script imports 
#     local packages using absolute workspace imports by executing this path resolution first.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import mlx.core as mx
import mlx.nn as nn
from training.core.char_tokenizer import CharTokenizer
from distilled_emb.model import TinyCharEncoder
from model.pre_phase1_dit import NARFlowMatcher

def load_pt_checkpoint(pt_path):
    """
    Load PyTorch checkpoint state dict using torch CPU backend if available,
    and convert parameters to MLX arrays.
    """
    import torch
    print(f"Loading weights from PyTorch checkpoint: {pt_path}")
    checkpoint = torch.load(pt_path, map_location='cpu')
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    mlx_params = {}
    for k, v in state_dict.items():
        if "attn.in_proj_weight" in k:
            q_w, k_w, v_w = v.chunk(3, dim=0)
            mlx_params[k.replace("attn.in_proj_weight", "attention.q_proj.weight")] = mx.array(q_w.numpy())
            mlx_params[k.replace("attn.in_proj_weight", "attention.k_proj.weight")] = mx.array(k_w.numpy())
            mlx_params[k.replace("attn.in_proj_weight", "attention.v_proj.weight")] = mx.array(v_w.numpy())
            continue
        elif "attn.in_proj_bias" in k:
            q_b, k_b, v_b = v.chunk(3, dim=0)
            mlx_params[k.replace("attn.in_proj_bias", "attention.q_proj.bias")] = mx.array(q_b.numpy())
            mlx_params[k.replace("attn.in_proj_bias", "attention.k_proj.bias")] = mx.array(k_b.numpy())
            mlx_params[k.replace("attn.in_proj_bias", "attention.v_proj.bias")] = mx.array(v_b.numpy())
            continue
        elif "attn.out_proj." in k:
            mlx_params[k.replace("attn.", "attention.")] = mx.array(v.numpy())
            continue
            
        arr = mx.array(v.detach().numpy())
        new_k = k
        new_k = new_k.replace("mlp.0.", "mlp.layers.0.")
        new_k = new_k.replace("mlp.2.", "mlp.layers.2.")
        new_k = new_k.replace("z_proj.0.", "z_proj.layers.0.")
        new_k = new_k.replace("z_proj.2.", "z_proj.layers.2.")
        
        mlx_params[new_k] = arr
        
    import mlx.utils as mu
    return mu.tree_unflatten(list(mlx_params.items()))

def generate_flow(encoder, flow_matcher, tokenizer, prompt, steps=20, max_seq_len=64, scale_factor=1.0):
    """
    Instantly reconstructs sentence from raw text prompt by finding its Z_macro 
    and integrating the continuous flow from Gaussian noise.
    """
    # 1. Encode prompt
    ids = tokenizer.encode(prompt, add_special_tokens=False)[:max_seq_len]
    pad_len = max_seq_len - len(ids)
    input_ids = mx.array([ids + [tokenizer.pad_token_id] * pad_len])
    mask = mx.array([[1] * len(ids) + [0] * pad_len])
    
    # 2. Extract macro condition Z_macro (pooled sentence representation)
    z_macro = encoder(input_ids, attention_mask=mask) # (1, z_dim)
    
    # 3. Sample initial gaussian noise sequence X_0
    x_dim = encoder.tok_emb.weight.shape[1] # e.g. 1024
    x_t = mx.random.normal((1, max_seq_len, x_dim))
    
    # 4. Integrate ODE trajectory using Euler method
    dt = 1.0 / steps
    for i in range(steps):
        t_val = float(i) / steps
        t_arr = mx.array([t_val]) # (1,)
        
        # Predict velocity field vector
        v_pred = flow_matcher(x_t, t_arr, z_macro)
        
        # Euler Step
        x_t = x_t + v_pred * dt
        
    # 5. Project final sequence back to vocabulary space (using frozen tok_emb weights)
    # tok_emb.weight shape: (vocab_size, x_dim)
    emb_weights = encoder.tok_emb.weight
    
    # Scale x_t back down to the original embedding scale before projection
    x_t = x_t / scale_factor
    
    # Pointwise dot product similarity: (1, L, vocab_size)
    logits = mx.matmul(x_t, emb_weights.T)
    
    # Decode argmax tokens
    predicted_tokens = mx.argmax(logits, axis=-1)[0].tolist()
    
    decoded_chars = []
    for token in predicted_tokens:
        if token == tokenizer.eos_token_id:
            break
        decoded_chars.append(tokenizer.decode([token]))
        
    return "".join(decoded_chars)

def load_training_args(flow_ckpt_path):
    """
    Attempts to load training_args.json from the parent directory of the flow checkpoint.
    Returns a dictionary of found arguments or empty dict.
    """
    import json
    ckpt_dir = os.path.dirname(flow_ckpt_path)
    config_path = os.path.join(ckpt_dir, "training_args.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            print(f"Loaded companion training configuration from {config_path}")
            return config
        except Exception as e:
            print(f"Warning: Failed to parse training_args.json at {config_path}: {e}")
    return {}

def get_args():
    parser = argparse.ArgumentParser(description="Sequence Flow Matcher Local Inference (MLX)")
    parser.add_argument("--tinybert_ckpt", type=str, required=True, help="Path to frozen TinyCharEncoder weights")
    parser.add_argument("--flow_ckpt", type=str, required=True, help="Path to trained flow matcher PyTorch checkpoint")
    parser.add_argument("--prompt", type=str, default="今天天气真好，我们出去散步吧。", help="Prompt to test the reconstruction pipeline")
    parser.add_argument("--steps", type=int, default=20, help="Number of Euler integration steps")
    parser.add_argument("--max_seq_len", type=int, default=64)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--n_heads", type=int, default=8)
    return parser.parse_args()

def main():
    args = get_args()
    tokenizer = CharTokenizer()
    
    # 1. Load TinyCharEncoder
    print(f"Loading TinyCharEncoder from {args.tinybert_ckpt}")
    if args.tinybert_ckpt.endswith(".safetensors"):
        encoder = TinyCharEncoder(vocab_size=tokenizer.vocab_size)
        encoder.load_weights(args.tinybert_ckpt)
    else:
        # Bridge PyTorch pt weights if needed
        import torch
        pt_weights = torch.load(args.tinybert_ckpt, map_location='cpu', weights_only=True)
        sniffed_x_dim = pt_weights['tok_emb.weight'].shape[1]
        sniffed_z_dim = pt_weights['out_proj.weight'].shape[0]
        encoder = TinyCharEncoder(vocab_size=tokenizer.vocab_size, d_model=sniffed_x_dim, z_dim=sniffed_z_dim)
        
        mlx_enc_params = {}
        for k, v in pt_weights.items():
            new_k = k
            if "layers." in new_k:
                new_k = new_k.replace(".q_proj.", ".attention.q_proj.")
                new_k = new_k.replace(".k_proj.", ".attention.k_proj.")
                new_k = new_k.replace(".v_proj.", ".attention.v_proj.")
                # the transformer layer has its own out_proj
                # wait, let's be safe and check if it's layers.X.out_proj
                # yes, layers.0.out_proj.weight -> layers.0.attention.out_proj.weight
                new_k = new_k.replace(".out_proj.", ".attention.out_proj.")
            mlx_enc_params[new_k] = mx.array(v.numpy())
        import mlx.utils as mu
        encoder.update(mu.tree_unflatten(list(mlx_enc_params.items())))
        encoder.update(mu.tree_unflatten(list(mlx_enc_params.items())))
        
    # Auto-load hyperparams from companion training_args.json
    config = load_training_args(args.flow_ckpt)
    d_model = config.get("d_model", args.d_model)
    n_layers = config.get("n_layers", args.n_layers)
    n_heads = config.get("n_heads", args.n_heads)
    max_seq_len = config.get("max_seq_len", args.max_seq_len)
    
    emb_scale_factor = config.get("emb_scale_factor", 1.0)
    print(f"Model parameters: d_model={d_model}, n_layers={n_layers}, n_heads={n_heads}, max_seq_len={max_seq_len}, scale_factor={emb_scale_factor:.2f}")
    
    # 2. Instantiate Flow Matcher
    flow_matcher = NARFlowMatcher(
        z_dim=encoder.out_proj.weight.shape[0],
        x_dim=encoder.tok_emb.weight.shape[1],
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        max_seq_len=max_seq_len
    )
    
    # Load Flow Matcher weights
    mlx_flow_params = load_pt_checkpoint(args.flow_ckpt)
    flow_matcher.update(mlx_flow_params)
    
    print("\n--- Running Flow Matching Reconstruction Test ---")
    print(f"Original Text: {args.prompt}")
    
    reconstructed = generate_flow(
        encoder=encoder,
        flow_matcher=flow_matcher,
        tokenizer=tokenizer,
        prompt=args.prompt,
        steps=args.steps,
        max_seq_len=max_seq_len,
        scale_factor=emb_scale_factor
    )
    
    print(f"Reconstructed: {reconstructed}")
    print("-------------------------------------------------")

if __name__ == "__main__":
    main()
