import os
import sys

# [BUG FIX]
# (1) Specific Problem: Running this inference script on servers via python command raises 
#     "ModuleNotFoundError: No module named 'training'" or similar because the workspace root is not in Python path.
# (2) Method to Resolve: Explicitly insert the parent directory of this script (workspace root) 
#     into the beginning of sys.path before importing local project packages (model, training, distilled_emb).
# (3) Caveats: Ensure any other newly created root-level script or nested script imports 
#     local packages using absolute workspace imports by executing this path resolution first.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import torch
import torch.nn as nn
from training.core.char_tokenizer import CharTokenizer
from distilled_emb.model_cuda import TinyCharEncoderCUDA
from model.pre_phase1_dit_cuda import NARFlowMatcherCUDA

def generate_flow_cuda(encoder, flow_matcher, tokenizer, prompt, steps=20, max_seq_len=64, device="cuda"):
    encoder.eval()
    flow_matcher.eval()
    
    # 1. Encode prompt
    ids = tokenizer.encode(prompt, add_special_tokens=False)[:max_seq_len]
    pad_len = max_seq_len - len(ids)
    input_ids = torch.tensor([ids + [tokenizer.pad_token_id] * pad_len], dtype=torch.long, device=device)
    mask = torch.tensor([[1] * len(ids) + [0] * pad_len], dtype=torch.float32, device=device)
    
    # 2. Extract macro condition Z_macro (pooled sentence representation)
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            z_macro = encoder(input_ids, attention_mask=mask) # (1, z_dim)
        
    # 3. Sample initial gaussian noise sequence X_0
    x_dim = encoder.tok_emb.weight.shape[1] # e.g. 1024
    x_t = torch.randn((1, max_seq_len, x_dim), device=device)
    
    # 4. Integrate ODE trajectory using Euler method
    dt = 1.0 / steps
    for i in range(steps):
        t_val = float(i) / steps
        t_tensor = torch.tensor([t_val], device=device, dtype=torch.float32) # (1,)
        
        # Predict velocity field vector
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                v_pred = flow_matcher(x_t, t_tensor, z_macro)
            
        # Euler Step
        x_t = x_t + v_pred * dt
        
    # 5. Project final sequence back to vocabulary space (using frozen tok_emb weights)
    # tok_emb.weight shape: (vocab_size, x_dim)
    emb_weights = encoder.tok_emb.weight
    
    # Pointwise dot product similarity: (1, L, vocab_size)
    logits = torch.matmul(x_t, emb_weights.T)
    
    # Decode argmax tokens
    predicted_tokens = torch.argmax(logits, dim=-1)[0].tolist()
    
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
    parser = argparse.ArgumentParser(description="Sequence Flow Matcher Inference (CUDA/PyTorch)")
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = CharTokenizer()
    
    # 1. Load TinyCharEncoder
    print(f"Loading TinyCharEncoderCUDA from {args.tinybert_ckpt}")
    if args.tinybert_ckpt.endswith(".safetensors"):
        from safetensors.torch import load_file
        weights = load_file(args.tinybert_ckpt)
    else:
        weights = torch.load(args.tinybert_ckpt, map_location='cpu', weights_only=True)
        
    sniffed_x_dim = weights['tok_emb.weight'].shape[1]
    sniffed_z_dim = weights['out_proj.weight'].shape[0]
    
    encoder = TinyCharEncoderCUDA(vocab_size=tokenizer.vocab_size, d_model=sniffed_x_dim, z_dim=sniffed_z_dim)
    encoder.load_state_dict(weights, strict=False)
    encoder.to(device).eval()
    
    # Auto-load hyperparams from companion training_args.json
    config = load_training_args(args.flow_ckpt)
    d_model = config.get("d_model", args.d_model)
    n_layers = config.get("n_layers", args.n_layers)
    n_heads = config.get("n_heads", args.n_heads)
    max_seq_len = config.get("max_seq_len", args.max_seq_len)
    
    print(f"Model parameters: d_model={d_model}, n_layers={n_layers}, n_heads={n_heads}, max_seq_len={max_seq_len}")
    
    # 2. Instantiate Flow Matcher
    flow_matcher = NARFlowMatcherCUDA(
        z_dim=sniffed_z_dim,
        x_dim=sniffed_x_dim,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        max_seq_len=max_seq_len
    )
    
    print(f"Loading Flow Matcher weights from: {args.flow_ckpt}")
    checkpoint = torch.load(args.flow_ckpt, map_location='cpu')
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    flow_matcher.load_state_dict(state_dict)
    flow_matcher.to(device).eval()
    
    print("\n--- Running CUDA Flow Matching Reconstruction Test ---")
    print(f"Original Text: {args.prompt}")
    
    reconstructed = generate_flow_cuda(
        encoder=encoder,
        flow_matcher=flow_matcher,
        tokenizer=tokenizer,
        prompt=args.prompt,
        steps=args.steps,
        max_seq_len=max_seq_len,
        device=device
    )
    
    print(f"Reconstructed: {reconstructed}")
    print("-------------------------------------------------")

if __name__ == "__main__":
    main()
