import os
import argparse
import mlx.core as mx
import mlx.nn as nn
from training.core.char_tokenizer import CharTokenizer
from distilled_emb.model import TinyCharEncoder
from model.phase1_dit import NARFlowMatcher

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
        # Transfer PyTorch tensor to MLX array
        arr = mx.array(v.detach().numpy())
        
        # Map PyTorch module names to MLX structures
        new_k = k
        # PyTorch Sequential layers mapping
        new_k = new_k.replace("mlp.0.", "mlp.layers.0.")
        new_k = new_k.replace("mlp.2.", "mlp.layers.2.")
        new_k = new_k.replace("z_proj.0.", "z_proj.layers.0.")
        new_k = new_k.replace("z_proj.2.", "z_proj.layers.2.")
        
        mlx_params[new_k] = arr
        
    return mlx_params

def generate_flow(encoder, flow_matcher, tokenizer, prompt, steps=20, max_seq_len=64):
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
        
        mlx_enc_params = {k: mx.array(v.numpy()) for k, v in pt_weights.items()}
        encoder.update(mlx_enc_params)
        
    # 2. Instantiate Flow Matcher
    flow_matcher = NARFlowMatcher(
        z_dim=encoder.out_proj.weight.shape[0],
        x_dim=encoder.tok_emb.weight.shape[1],
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        max_seq_len=args.max_seq_len
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
        max_seq_len=args.max_seq_len
    )
    
    print(f"Reconstructed: {reconstructed}")
    print("-------------------------------------------------")

if __name__ == "__main__":
    main()
