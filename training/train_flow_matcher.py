import os
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

from distilled_emb.model_cuda import TinyCharEncoderCUDA
from model.flow_matcher_cuda import NARFlowMatcherCUDA

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
    
    global_step = 0
    start_epoch = 0
    
    if args.resume_from:
        print(f"Resuming training from {args.resume_from}")
        checkpoint = torch.load(args.resume_from, map_location='cpu')
        decoder.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        global_step = checkpoint['global_step']
        start_epoch = checkpoint['epoch']
        print(f"Resumed at Epoch {start_epoch}, Step {global_step}")
    
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
                save_path = os.path.join(args.out_dir, f"flow_matcher_step_{global_step}.pt")
                torch.save({
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': decoder.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict()
                }, save_path)
                print(f"Saved checkpoint to {save_path}")
 
            if args.max_steps > 0 and global_step >= args.max_steps:
                print(f"Reached max_steps ({args.max_steps}). Stopping training early.")
                break
                
        if args.max_steps > 0 and global_step >= args.max_steps:
            break

        save_path = os.path.join(args.out_dir, f"flow_matcher_epoch_{epoch+1}.pt")
        torch.save({
            'epoch': epoch + 1,
            'global_step': global_step,
            'model_state_dict': decoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict()
        }, save_path)
        print(f"Saved Epoch {epoch+1} checkpoint to {save_path}")

if __name__ == "__main__":
    main()
