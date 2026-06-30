import os
import sys
import argparse
import json
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset

# Ensure parent directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from training.core.char_tokenizer import CharTokenizer
from distilled_emb.model_cuda import TinyCharEncoderCUDA
from model.pre_phase1_dit_cuda import NARFlowMatcherCUDA

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
    d_model = config.get("d_model", args.d_model)
    n_layers = config.get("n_layers", args.n_layers)
    n_heads = config.get("n_heads", args.n_heads)
    max_seq_len = config.get("max_seq_len", args.max_seq_len)
    print(f"Model parameters: d_model={d_model}, n_layers={n_layers}, n_heads={n_heads}, max_seq_len={max_seq_len}")
    
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
    
    # 4. Load Parquet Data subset
    print(f"Loading data from {args.parquet_path}...")
    df = pd.read_parquet(args.parquet_path)
    chunks = df['chunks'].explode().dropna().tolist()
    
    num_samples = min(args.num_samples, len(chunks))
    test_chunks = chunks[:num_samples]
    print(f"Evaluating reconstruction quality on {num_samples} samples...")
    
    # Process batch-by-batch
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
        
        # Prepare inputs
        enc_ids_list = [tokenizer.encode(t, add_special_tokens=False) for t in texts]
        e_ids_t, e_mask_t, d_ids_t = [], [], []
        for seq in enc_ids_list:
            e_seq = seq[:max_seq_len]
            e_pad = max_seq_len - len(e_seq)
            e_ids_t.append(e_seq + [tokenizer.pad_token_id] * e_pad)
            e_mask_t.append([1] * len(e_seq) + [0] * e_pad)
            
            d_seq = seq[:max_seq_len-1] + [tokenizer.eos_token_id]
            d_pad = max_seq_len - len(d_seq)
            d_ids_t.append(d_seq + [tokenizer.pad_token_id] * d_pad)
            
        enc_ids = torch.tensor(e_ids_t, dtype=torch.long, device=device)
        enc_mask = torch.tensor(e_mask_t, dtype=torch.long, device=device)
        target_ids = torch.tensor(d_ids_t, dtype=torch.long, device=device)
        
        # Target mask (ignore PAD tokens in evaluations)
        target_mask = (target_ids != tokenizer.pad_token_id).float().to(device)
        
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                # 1. Encode prompt
                z_macro = encoder(enc_ids, attention_mask=enc_mask)
                # Get ground truth embedding X_1
                X_1 = encoder.tok_emb(target_ids)
                
            # 2. ODE Solver (Euler Integration)
            x_t = torch.randn((B, max_seq_len, sniffed_x_dim), device=device)
            dt = 1.0 / args.steps
            for i in range(args.steps):
                t_val = float(i) / args.steps
                t_tensor = torch.tensor([t_val] * B, device=device, dtype=torch.float32)
                with torch.cuda.amp.autocast():
                    v_pred = flow_matcher(x_t, t_tensor, z_macro)
                x_t = x_t + v_pred * dt
                
            # 3. Project to vocab and compute accuracy
            emb_weights = encoder.tok_emb.weight
            with torch.cuda.amp.autocast():
                logits = torch.matmul(x_t, emb_weights.T) # (B, L, V)
            predicted_tokens = torch.argmax(logits, dim=-1) # (B, L)
            
            # Check accuracy per token (masked)
            correct_mask = (predicted_tokens == target_ids).float() * target_mask
            correct_chars += correct_mask.sum().item()
            total_chars += target_mask.sum().item()
            
            # Check sentence exact match (all active tokens correct)
            for b in range(B):
                sent_mask = target_mask[b]
                sent_len = sent_mask.sum().item()
                sent_correct = correct_mask[b].sum().item()
                total_sentences += 1
                if sent_correct == sent_len:
                    exact_match_sentences += 1
                    
            # Compute cosine similarity between generated representation and X_1
            dot_product = (x_t * X_1).sum(dim=-1) # (B, L)
            x_t_norm = torch.norm(x_t, dim=-1).clamp(min=1e-8)
            X_1_norm = torch.norm(X_1, dim=-1).clamp(min=1e-8)
            cosine_sims = dot_product / (x_t_norm * X_1_norm) # (B, L)
            
            # Masked average cosine similarity
            total_cosine_sim += (cosine_sims * target_mask).sum().item()
            cosine_count += target_mask.sum().item()
            
    char_acc = (correct_chars / total_chars) * 100 if total_chars > 0 else 0
    em_rate = (exact_match_sentences / total_sentences) * 100 if total_sentences > 0 else 0
    avg_cos = (total_cosine_sim / cosine_count) if cosine_count > 0 else 0
    
    print("\n" + "="*50)
    print("      Flow Matcher Reconstruction Evaluation      ")
    print("="*50)
    print(f"Checkpoint:       {os.path.basename(args.flow_ckpt)}")
    print(f"Tested Samples:   {num_samples}")
    print(f"Steps:            {args.steps}")
    print("-"*50)
    print(f"Avg Cosine Sim:   {avg_cos:.4f} (Closer to 1.0 is better)")
    print(f"Char Accuracy:    {char_acc:.2f}% (Token-level matching)")
    print(f"Sentence EM:      {em_rate:.2f}% (Entire sentence matching)")
    print("="*50)
    print("\nStopping Criterion Reference:")
    print(" - Char Accuracy > 90% and Sentence EM > 50%: Model is fully converged and ready to stop.")
    print(" - If Char Accuracy plateaus for 1 Epoch: Can stop and proceed to Phase 2.")
    print("="*50 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Flow Matcher Reconstruction Accuracy (CUDA)")
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
