import os
import sys

# [BUG FIX]
# (1) Specific Problem: Running this training script on servers via python command raises 
#     "ModuleNotFoundError: No module named 'training'" because the workspace root is not in Python path.
# (2) Method to Resolve: Explicitly insert the parent directory of this script (workspace root) 
#     into the beginning of sys.path before importing local project packages (model, training).
# (3) Caveats: Ensure any other newly created root-level script or nested script imports 
#     local packages using absolute workspace imports by executing this path resolution first.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# [BUG FIX]
# (1) Specific Problem: The training loss plateaus early and evaluation yields random noise. 
#     This occurs because Flow Matching uses X_0 ~ N(0, 1), but the target embeddings X_1 
#     have a very small standard deviation (e.g., 0.032). This scale mismatch causes the velocity 
#     field to be dominated by noise, preventing the model from learning the target data distribution.
# (2) Method to Resolve: Dynamically calculate the standard deviation of the target embeddings (X_1) 
#     and compute a `scale_factor` to upscale X_1 to have a variance of ~1.0 during training.
#     This factor must be saved to `training_args.json` and applied inversely during inference/evaluation.
# (3) Caveats: NEVER attempt Flow Matching or Diffusion without matching the scale of the target 
#     data to the scale of the initial noise distribution! Failure to do so mathematically 
#     dooms the model to predict pure noise reduction.

import argparse
import time
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from training.core.char_tokenizer import CharTokenizer
from training.core.args import get_training_parser
from training.core.checkpoint import Checkpointer

from distilled_emb.model_cuda import TinyCharEncoderCUDA
from model.pre_phase1_dit_cuda import NARFlowMatcherCUDA

class ChunkTextDataset(Dataset):
    def __init__(self, parquet_path):
        print(f"Loading parquet from {parquet_path}")
        df = pd.read_parquet(parquet_path)
        self.chunks = df['chunks'].explode().dropna().tolist()
        print(f"Loaded {len(self.chunks)} text chunks.")
        
    def __len__(self):
        return len(self.chunks)
        
    def __getitem__(self, idx):
        return self.chunks[idx]

def get_args():
    # 1. Inherit from the unified training arguments template
    parser = get_training_parser(description="Sequence Flow Matcher Training")
    
    # 2. Append flow-matcher specific arguments
    parser.add_argument("--parquet_path", type=str, default="data/Basic_ZH/chunked_mixed_wiki.parquet", help="Path to chunked text dataset")
    parser.add_argument("--tinybert_ckpt", type=str, required=True, help="Path to frozen TinyCharEncoder pt/safetensors file")
    parser.add_argument("--max_seq_len", type=int, default=64, help="Max length of target sequence")
    parser.add_argument("--d_model", type=int, default=512, help="Hidden dimension of Flow Matcher")
    parser.add_argument("--n_layers", type=int, default=6, help="Number of Transformer layers")
    parser.add_argument("--n_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--log_steps", type=int, default=50, help="Log step interval")
    parser.add_argument("--max_steps", type=int, default=-1, help="Max training steps (for quick testing)")
    
    # 3. Override default settings for flow-matcher training
    parser.set_defaults(
        out_dir="checkpoints/flow_matcher",
        batch_size=256,
        epochs=10,
        lr=1e-4
    )
    
    return parser.parse_args()

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    
    tokenizer = CharTokenizer()

    # 1. Load Dataset
    dataset = ChunkTextDataset(args.parquet_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    
    # 2. Load Frozen TinyCharEncoder
    print(f"Loading Frozen TinyCharEncoder from {args.tinybert_ckpt}")
    if args.tinybert_ckpt.endswith(".safetensors"):
        from safetensors.torch import load_file
        weights = load_file(args.tinybert_ckpt)
    else:
        weights = torch.load(args.tinybert_ckpt, map_location='cpu', weights_only=True)
    
    sniffed_z_dim = weights['out_proj.weight'].shape[0]
    sniffed_x_dim = weights['tok_emb.weight'].shape[1] # e.g. 1024
    
    tiny_encoder = TinyCharEncoderCUDA(vocab_size=tokenizer.vocab_size, d_model=sniffed_x_dim, z_dim=sniffed_z_dim)
    tiny_encoder.load_state_dict(weights, strict=False)
    for param in tiny_encoder.parameters():
        param.requires_grad = False
    tiny_encoder.eval()
    tiny_encoder.to(device)
    
    # Calculate scale factor for Flow Matching
    std_x = tiny_encoder.tok_emb.weight.std().item()
    scale_factor = 1.0 / max(std_x, 1e-5)
    args.emb_scale_factor = scale_factor
    print(f"Calculated target embedding std: {std_x:.6f}, using scale_factor: {scale_factor:.2f}")
    
    # 3. Initialize Sequence Flow Matcher
    print(f"Initializing NARFlowMatcherCUDA: layers={args.n_layers}, d_model={args.d_model}, z_dim={sniffed_z_dim}, x_dim={sniffed_x_dim}")
    decoder = NARFlowMatcherCUDA(
        z_dim=sniffed_z_dim,
        x_dim=sniffed_x_dim,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        max_seq_len=args.max_seq_len
    ).to(device)
    
    optimizer = optim.AdamW(decoder.parameters(), lr=args.lr, weight_decay=1e-2)
    scaler = torch.cuda.amp.GradScaler()
    
    # 4. Initialize Unified Checkpointer
    checkpointer = Checkpointer(
        out_dir=args.out_dir,
        prefix=args.ckpt_prefix,
        keep_last_k=args.keep_last_k
    )
    checkpointer.register_model("flow_matcher", decoder)
    checkpointer.register_args(args)
    
    global_step = 0
    start_epoch = 0
    load_dir = None
    
    # Handle Resume logic via Checkpointer
    if args.resume_from:
        global_step = checkpointer.load(args.resume_from)
        load_dir = args.resume_from
    elif args.auto_resume:
        global_step = checkpointer.load_latest()
        if global_step > 0:
            base_name = f"step_{global_step}"
            folder_name = f"{checkpointer.prefix}_{base_name}" if checkpointer.prefix else base_name
            load_dir = os.path.join(args.out_dir, folder_name)
            
    # Load PyTorch specific optimizer/scaler states if resumed
    if global_step > 0 and load_dir is not None:
        opt_path = os.path.join(load_dir, "optimizer.pt")
        scaler_path = os.path.join(load_dir, "scaler.pt")
        if os.path.exists(opt_path):
            optimizer.load_state_dict(torch.load(opt_path, map_location='cpu', weights_only=True))
            print(f"Resumed PyTorch optimizer state from {opt_path}")
        if os.path.exists(scaler_path):
            scaler.load_state_dict(torch.load(scaler_path, map_location='cpu'))
            print(f"Resumed PyTorch scaler state from {scaler_path}")
            
        start_epoch = global_step // len(dataloader)
        print(f"Resumed training at Epoch {start_epoch + 1}, Step {global_step}")
    
    print("Starting Sequence Flow Matcher Training...")
    
    for epoch in range(start_epoch, args.epochs):
        decoder.train()
        for batch_idx, texts in enumerate(dataloader):
            global_step += 1
            
            # --- Prepare text IDs & Masks ---
            enc_ids_list = [tokenizer.encode(t, add_special_tokens=False) for t in texts]
            
            e_ids_t, e_mask_t, d_ids_t = [], [], []
            for seq in enc_ids_list:
                # Encoder input sequence
                e_seq = seq[:args.max_seq_len]
                e_pad = args.max_seq_len - len(e_seq)
                e_ids_t.append(e_seq + [tokenizer.pad_token_id] * e_pad)
                e_mask_t.append([1] * len(e_seq) + [0] * e_pad)
                
                # Target sequence (with EOS and PAD)
                d_seq = seq[:args.max_seq_len-1] + [tokenizer.eos_token_id]
                d_pad = args.max_seq_len - len(d_seq)
                d_ids_t.append(d_seq + [tokenizer.pad_token_id] * d_pad)
                
            enc_ids = torch.tensor(e_ids_t, dtype=torch.long, device=device)
            enc_mask = torch.tensor(e_mask_t, dtype=torch.long, device=device)
            target_ids = torch.tensor(d_ids_t, dtype=torch.long, device=device)
            
            # Create target mask (1 for valid/EOS, 0 for PAD)
            target_mask = (target_ids != tokenizer.pad_token_id).float().to(device)
            
            # 1. Forward frozen encoder to get conditions
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    z_truth = tiny_encoder(enc_ids, attention_mask=enc_mask) # (B, z_dim)
                    # Get ground truth static word embeddings as target X_1
                    X_1 = tiny_encoder.tok_emb(target_ids) # (B, L, x_dim)
                    X_1 = X_1 * scale_factor # Scale target to match noise variance
                    
            # 2. Sample flow matching variables
            B, L, D = X_1.shape
            t = torch.rand((B,), device=device)
            X_0 = torch.randn_like(X_1) # High-gaussian noise shape (B, L, D)
            
            # Interpolated flow state: X_t = t * X_1 + (1 - t) * X_0
            t_expand = t.view(B, 1, 1)
            X_t = t_expand * X_1 + (1.0 - t_expand) * X_0
            
            # Ground truth velocity vector field: U_t = X_1 - X_0
            U_t = X_1 - X_0
            
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                # Predict velocity field
                V_t = decoder(X_t, t, z_truth) # (B, L, x_dim)
                
                # Compute MSE loss masked for PAD tokens
                loss_elements = F.mse_loss(V_t, U_t, reduction='none') # (B, L, x_dim)
                loss_per_token = loss_elements.mean(dim=-1) # (B, L)
                
                # Apply padding mask
                loss = (loss_per_token * target_mask).sum() / target_mask.sum().clamp(min=1e-8)
                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            if global_step % args.log_steps == 0:
                print(f"Epoch {epoch+1}/{args.epochs} | Step {global_step} | Loss: {loss.item():.4f}")
                
            if global_step % args.save_steps == 0:
                checkpointer.save(global_step)
                
                base_name = f"step_{global_step}"
                folder_name = f"{checkpointer.prefix}_{base_name}" if checkpointer.prefix else base_name
                save_dir = os.path.join(args.out_dir, folder_name)
                
                torch.save(optimizer.state_dict(), os.path.join(save_dir, "optimizer.pt"))
                torch.save(scaler.state_dict(), os.path.join(save_dir, "scaler.pt"))
                print(f"Saved PyTorch optimizer & scaler states to {save_dir}")
 
            if args.max_steps > 0 and global_step >= args.max_steps:
                print(f"Reached max_steps ({args.max_steps}). Stopping training early.")
                break
                
        if args.max_steps > 0 and global_step >= args.max_steps:
            break

        # Save epoch checkpoint via Checkpointer
        checkpointer.save(global_step)
        
        base_name = f"step_{global_step}"
        folder_name = f"{checkpointer.prefix}_{base_name}" if checkpointer.prefix else base_name
        save_dir = os.path.join(args.out_dir, folder_name)
        
        torch.save(optimizer.state_dict(), os.path.join(save_dir, "optimizer.pt"))
        torch.save(scaler.state_dict(), os.path.join(save_dir, "scaler.pt"))
        print(f"Saved Epoch {epoch+1} checkpoint to {save_dir}")

if __name__ == "__main__":
    main()
