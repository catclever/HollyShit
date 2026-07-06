import os
import sys
import argparse
import json
import pandas as pd
import mlx.core as mx

# Ensure parent directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from training.core.char_tokenizer import CharTokenizer
from distilled_emb.model import TinyCharEncoder
from model.pre_phase1_dit import NARFlowMatcher

def load_pt_checkpoint(pt_path):
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

def load_training_args(flow_ckpt_path):
    ckpt_dir = os.path.dirname(flow_ckpt_path)
    config_path = os.path.join(ckpt_dir, "training_args.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            print(f"Loaded companion training configuration from {config_path}")
            return config
        except Exception as e:
            print(f"Warning: Failed to parse training_args.json: {e}")
    return {}

def evaluate(args):
    tokenizer = CharTokenizer()
    
    # 1. Load TinyCharEncoder
    print(f"Loading TinyCharEncoder from {args.tinybert_ckpt}")
    if args.tinybert_ckpt.endswith(".safetensors"):
        encoder = TinyCharEncoder(vocab_size=tokenizer.vocab_size)
        encoder.load_weights(args.tinybert_ckpt)
        sniffed_x_dim = encoder.tok_emb.weight.shape[1]
        sniffed_z_dim = encoder.out_proj.weight.shape[0]
    else:
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
                new_k = new_k.replace(".out_proj.", ".attention.out_proj.")
            mlx_enc_params[new_k] = mx.array(v.numpy())
        import mlx.utils as mu
        encoder.update(mu.tree_unflatten(list(mlx_enc_params.items())))
        encoder.update(mu.tree_unflatten(list(mlx_enc_params.items())))
        
    # 2. Load Flow Matcher Configuration
    config = load_training_args(args.flow_ckpt)
    d_model = config.get("d_model", args.d_model)
    n_layers = config.get("n_layers", args.n_layers)
    n_heads = config.get("n_heads", args.n_heads)
    max_seq_len = config.get("max_seq_len", args.max_seq_len)
    emb_scale_factor = config.get("emb_scale_factor", 1.0)
    print(f"Model parameters: d_model={d_model}, n_layers={n_layers}, n_heads={n_heads}, max_seq_len={max_seq_len}, scale_factor={emb_scale_factor:.2f}")
    
    # 3. Instantiate and Load Flow Matcher
    flow_matcher = NARFlowMatcher(
        z_dim=sniffed_z_dim,
        x_dim=sniffed_x_dim,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        max_seq_len=max_seq_len
    )
    mlx_flow_params = load_pt_checkpoint(args.flow_ckpt)
    flow_matcher.update(mlx_flow_params)
    
    # 4. Load Parquet Data subset
    print(f"Loading data from {args.parquet_path}...")
    df = pd.read_parquet(args.parquet_path)
    chunks = df['chunks'].explode().dropna().tolist()
    
    num_samples = min(args.num_samples, len(chunks))
    test_chunks = chunks[:num_samples]
    print(f"Evaluating reconstruction quality on {num_samples} samples...")
    
    batch_size = min(32, num_samples)
    total_chars = 0
    correct_chars = 0
    total_sentences = 0
    exact_match_sentences = 0
    total_cosine_sim = 0.0
    cosine_count = 0
    
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        texts = test_chunks[start_idx:end_idx]
        B = len(texts)
        
        enc_ids_list = [tokenizer.encode(t, add_special_tokens=True) for t in texts]
        e_ids_t, e_mask_t, d_ids_t = [], [], []
        for seq in enc_ids_list:
            seq_trunc = seq[:max_seq_len]
            pad_len = max_seq_len - len(seq_trunc)
            
            padded_seq = seq_trunc + [tokenizer.pad_token_id] * pad_len
            mask = [1] * len(seq_trunc) + [0] * pad_len
            
            e_ids_t.append(padded_seq)
            e_mask_t.append(mask)
            d_ids_t.append(padded_seq)
            
        enc_ids = mx.array(e_ids_t)
        enc_mask = mx.array(e_mask_t)
        target_ids = mx.array(d_ids_t)
        
        target_mask = (target_ids != tokenizer.pad_token_id)
        
        # 1. Encode prompt
        z_macro = encoder(enc_ids, attention_mask=enc_mask)
        X_1 = encoder.tok_emb(target_ids)
        
        # 2. ODE Solver (Euler Integration)
        x_t = mx.random.normal((B, max_seq_len, sniffed_x_dim))
        dt = 1.0 / args.steps
        for i in range(args.steps):
            t_val = float(i) / args.steps
            t_tensor = mx.array([t_val] * B)
            v_pred = flow_matcher(x_t, t_tensor, z_macro)
            x_t = x_t + v_pred * dt
            
        # 3. Project to vocab and compute accuracy
        emb_weights = encoder.tok_emb.weight
        
        # Scale back to original embedding space
        x_t = x_t / emb_scale_factor
        
        # Cosine similarity for decoding
        x_norm = x_t / (mx.linalg.norm(x_t, axis=-1, keepdims=True) + 1e-8)
        emb_norm = emb_weights / (mx.linalg.norm(emb_weights, axis=-1, keepdims=True) + 1e-8)
        
        logits = mx.matmul(x_norm, emb_norm.T) # (B, L, V)
        predicted_tokens = mx.argmax(logits, axis=-1) # (B, L)
        
        correct_mask = (predicted_tokens == target_ids) * target_mask
        
        # Convert back to native python scalars/lists for easy counting
        correct_mask_np = correct_mask.tolist()
        target_mask_np = target_mask.tolist()
        
        for b in range(B):
            sent_mask = target_mask_np[b]
            sent_correct = correct_mask_np[b]
            
            s_len = sum(sent_mask)
            s_corr = sum([1 for c, m in zip(sent_correct, sent_mask) if c and m])
            
            correct_chars += s_corr
            total_chars += s_len
            total_sentences += 1
            if s_corr == s_len:
                exact_match_sentences += 1
                
        # Compute cosine similarity
        dot_product = (x_t * X_1).sum(axis=-1)
        x_t_norm = mx.linalg.norm(x_t, axis=-1)
        X_1_norm = mx.linalg.norm(X_1, axis=-1)
        cosine_sims = dot_product / (x_t_norm * X_1_norm + 1e-8)
        
        # Masked sum for cosine
        cosine_sims_np = cosine_sims.tolist()
        for b in range(B):
            sent_mask = target_mask_np[b]
            sent_sim = cosine_sims_np[b]
            total_cosine_sim += sum([s for s, m in zip(sent_sim, sent_mask) if m])
            cosine_count += sum(sent_mask)

    char_acc = (correct_chars / total_chars) * 100 if total_chars > 0 else 0
    em_rate = (exact_match_sentences / total_sentences) * 100 if total_sentences > 0 else 0
    avg_cos = (total_cosine_sim / cosine_count) if cosine_count > 0 else 0
    
    print("\n" + "="*50)
    print("      Flow Matcher Reconstruction Evaluation (MLX)    ")
    print("="*50)
    print(f"Checkpoint:       {os.path.basename(args.flow_ckpt)}")
    print(f"Tested Samples:   {num_samples}")
    print(f"Steps:            {args.steps}")
    print("-" * 50)
    print(f"Avg Cosine Sim:   {avg_cos:.4f} (Closer to 1.0 is better)")
    print(f"Char Accuracy:    {char_acc:.2f}% (Token-level matching)")
    print(f"Sentence EM:      {em_rate:.2f}% (Entire sentence matching)")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Flow Matcher Reconstruction Accuracy (MLX)")
    parser.add_argument("--tinybert_ckpt", type=str, required=True, help="Path to frozen TinyCharEncoder weights")
    parser.add_argument("--flow_ckpt", type=str, required=True, help="Path to trained flow matcher PyTorch checkpoint")
    parser.add_argument("--parquet_path", type=str, required=True, help="Path to evaluation dataset")
    parser.add_argument("--num_samples", type=int, default=200, help="Number of samples to evaluate on")
    parser.add_argument("--steps", type=int, default=20, help="Integration steps")
    parser.add_argument("--max_seq_len", type=int, default=64)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--n_heads", type=int, default=8)
    
    args = parser.parse_args()
    evaluate(args)
