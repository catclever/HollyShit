"""
Archived on: 2026-07-27
Reason: Archived before implementing the new Mamba-DiT Next-Thought Flow Matching architecture. The original phase 2 was a pure DiT autoencoder without the Mamba macro-concept reader.
"""
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
from model.phase1_charencoder_cuda import TinyCharEncoderCUDA
from model.phase2_dit_cuda import NARFlowMatcherCUDA

def generate_flow_cuda(encoder, flow_matcher, tokenizer, prompt, steps=20, max_seq_len=64, scale_factor=1.0, cfg_scale=1.0, device="cuda"):
    encoder.eval()
    flow_matcher.eval()
    
    # 1. Encode prompt and pad to max_seq_len
    ids = tokenizer.encode(prompt, add_special_tokens=True)
    
    # Truncate and Pad to exactly max_seq_len
    seq_trunc = ids[:max_seq_len]
    pad_len = max_seq_len - len(seq_trunc)
    padded_seq = seq_trunc + [tokenizer.pad_token_id] * pad_len
    mask_list = [1] * len(seq_trunc) + [0] * pad_len
    
    input_ids = torch.tensor([padded_seq], dtype=torch.long, device=device)
    mask = torch.tensor([mask_list], dtype=torch.float32, device=device)
    
    # 2. Extract macro condition Z_macro (pooled sentence representation)
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            z_macro = encoder(input_ids, attention_mask=mask) # (1, z_dim)
        
    # 3. Sample initial gaussian noise sequence X_0 and project to hypersphere (always use max_seq_len)
    x_dim = encoder.tok_emb.weight.shape[1] # e.g. 1024
    x_t = torch.randn((1, max_seq_len, x_dim), device=device)
    import torch.nn.functional as F
    x_t = F.normalize(x_t, p=2, dim=-1) * scale_factor
    
    # 4. Integrate ODE trajectory using Spherical Euler method
    dt = 1.0 / steps
    for i in range(steps):
        t_val = float(i) / steps
        t_tensor = torch.tensor([t_val], device=device, dtype=torch.float32) # (1,)
        
        # Predict velocity field vector
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                if cfg_scale == 1.0:
                    v_pred = flow_matcher(x_t, t_tensor, z_macro)
                else:
                    v_cond = flow_matcher(x_t, t_tensor, z_macro)
                    z_uncond = torch.zeros_like(z_macro)
                    v_uncond = flow_matcher(x_t, t_tensor, z_uncond)
                    v_pred = v_uncond + cfg_scale * (v_cond - v_uncond)
            
        # Euler Step along the tangent
        x_t = x_t + v_pred * dt
        # Re-project to stay strictly on the hypersphere
        x_t = F.normalize(x_t, p=2, dim=-1) * scale_factor
        
    # 5. Project final sequence back to vocabulary space (using frozen tok_emb weights)
    # tok_emb.weight shape: (vocab_size, x_dim)
    emb_weights = encoder.tok_emb.weight
    
    # Scale x_t back down to the original embedding scale before projection
    x_t = x_t / scale_factor
    
    # Cosine similarity for decoding
    import torch.nn.functional as F
    x_norm = F.normalize(x_t, p=2, dim=-1)
    emb_norm = F.normalize(emb_weights, p=2, dim=-1)
    # (1, L, D) @ (D, vocab_size) -> (1, L, vocab_size)
    logits = torch.matmul(x_norm, emb_norm.T)
    
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
    parser.add_argument("--prompt", type=str, default="今天天气不错", help="Input prompt to encode into z_macro")
    parser.add_argument("--steps", type=int, default=50, help="Number of Euler integration steps")
    parser.add_argument("--max_seq_len", type=int, default=64, help="Must match training max_seq_len")
    parser.add_argument("--cfg_scale", type=float, default=1.0, help="Classifier-Free Guidance scale")
    parser.add_argument("--split_mode", type=str, choices=["none", "punct", "length"], default="none", help="How to split long prompts")
    parser.add_argument("--split_len", type=int, default=60, help="Max length for 'length' split mode")
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
    
    # 2. Load Flow Matcher Configuration
    config = load_training_args(args.flow_ckpt)
    if not config:
        print("Warning: training_args.json not found! Using fallback defaults (d_model=512, n_layers=6, n_heads=8, max_seq_len=64)")
        
    d_model = config.get("d_model", 512)
    n_layers = config.get("n_layers", 6)
    n_heads = config.get("n_heads", 8)
    max_seq_len = config.get("max_seq_len", 64)
    emb_scale_factor = config.get("emb_scale_factor", 1.0)
    print(f"Model parameters loaded from config: d_model={d_model}, n_layers={n_layers}, n_heads={n_heads}, max_seq_len={max_seq_len}, scale_factor={emb_scale_factor:.2f}")
    
    # 3. Instantiate and Load Flow Matcher
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
    
    # --- Split Logic ---
    chunks = []
    if args.split_mode == "none":
        chunks = [args.prompt]
    elif args.split_mode == "punct":
        import re
        # Split by Chinese and English punctuation
        raw_chunks = re.split(r'([，。！？；：、,.!?;:])', args.prompt)
        # Re-attach punctuation
        chunks = []
        for i in range(0, len(raw_chunks)-1, 2):
            if raw_chunks[i]:
                chunks.append(raw_chunks[i] + raw_chunks[i+1])
        if len(raw_chunks) % 2 != 0 and raw_chunks[-1]:
            chunks.append(raw_chunks[-1])
    elif args.split_mode == "length":
        s_len = args.split_len
        chunks = [args.prompt[i:i+s_len] for i in range(0, len(args.prompt), s_len)]
        
    print(f"\nSplit Mode: {args.split_mode}")
    print(f"Chunks generated: {len(chunks)}")
    for i, c in enumerate(chunks):
        print(f"  Chunk {i+1}: {c}")
        
    print("\n--- Generation Results ---")
    final_output = ""
    for idx, chunk in enumerate(chunks):
        if not chunk.strip():
            continue
            
        reconstructed = generate_flow_cuda(
            encoder=encoder,
            flow_matcher=flow_matcher,
            tokenizer=tokenizer,
            prompt=chunk,
            steps=args.steps,
            max_seq_len=args.max_seq_len,
            scale_factor=emb_scale_factor,
            cfg_scale=args.cfg_scale,
            device=device
        )
        
        # Clean up special tokens for printing
        clean_text = reconstructed.replace("<BOS>", "").replace("<EOS>", "").replace("<PAD>", "").strip()
        final_output += clean_text
        print(f"Chunk {idx+1} Output: {clean_text}")

    print("\n==============================")
    print(f"Final Assembled: {final_output}")
    print("==============================")
    print("-------------------------------------------------")

if __name__ == "__main__":
    main()
