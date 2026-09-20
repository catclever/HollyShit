import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import sys

# Auto-resolve the parent workspace root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir in sys.path and os.path.exists(os.path.join(script_dir, "model.py")):
    sys.path = [p for p in sys.path if p != script_dir] + [script_dir]
    
from training.core.char_tokenizer import CharTokenizer
from training.data.mixed_dataset import MixedDataset
from model.phase1_charencoder_cuda import TinyCharEncoderCUDA
from model.concept_chunker import ConceptChunker
from model.phase2_mamba import MambaReader
from model.phase2_dit import ConceptDiT

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    tokenizer = CharTokenizer()
    dataset = MixedDataset(
        wiki_parquet_path="data/Basic_ZH/wiki_filtered.parquet",
        dialogues_jsonl_path="data/Basic_ZH/dialogues_254k.jsonl",
        tokenizer=tokenizer,
        seq_len=args.seq_len
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Load Phase 1 Encoder (Frozen)
    encoder = TinyCharEncoderCUDA(vocab_size=tokenizer.vocab_size).to(device)
    ckpt_path = "checkpoints/basencoder_step_115000/student.pt"
    if os.path.exists(ckpt_path):
        encoder.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        print("Loaded frozen TinyCharEncoder.")
    else:
        print(f"Warning: Could not find checkpoint {ckpt_path}. Using random init for Encoder.")
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
        
    # Chunker (Deterministic, no parameters)
    chunker = ConceptChunker(threshold=0.5).to(device)
    
    # Phase 2 Models
    mamba = MambaReader(d_model=1024, num_layers=4).to(device)
    dit = ConceptDiT(d_model=1024, max_len=128, num_layers=6, num_heads=8).to(device)
    
    # Optimize both together
    params = list(mamba.parameters()) + list(dit.parameters())
    optimizer = optim.AdamW(params, lr=2e-4, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    epochs = 3
    global_step = 0
    start_epoch = 0
    
    ckpt_dir = "checkpoints/phase2_thoughtengine"
    os.makedirs(ckpt_dir, exist_ok=True)
    import glob
    existing_ckpts = glob.glob(os.path.join(ckpt_dir, "mamba_epoch_*.pt"))
    if existing_ckpts:
        existing_ckpts.sort(key=lambda x: int(x.split('_epoch_')[1].split('.pt')[0]))
        latest_mamba = existing_ckpts[-1]
        epoch_num = int(latest_mamba.split('_epoch_')[1].split('.pt')[0])
        latest_dit = os.path.join(ckpt_dir, f"dit_epoch_{epoch_num}.pt")
        meta_path = os.path.join(ckpt_dir, f"meta_epoch_{epoch_num}.pt")
        
        if os.path.exists(latest_dit) and os.path.exists(meta_path):
            print(f"Resuming from {latest_mamba}...")
            mamba.load_state_dict(torch.load(latest_mamba, map_location=device, weights_only=True))
            dit.load_state_dict(torch.load(latest_dit, map_location=device, weights_only=True))
            
            meta = torch.load(meta_path, map_location=device, weights_only=True)
            optimizer.load_state_dict(meta['optimizer_state_dict'])
            start_epoch = meta['epoch'] + 1
            global_step = meta['global_step']
            
    mamba.train()
    dit.train()
    
    print("Starting Phase 2 Thought Engine Training...")
    
    for epoch in range(start_epoch, epochs):
        for batch_idx, (w_t, w_tgt) in enumerate(dataloader):
            w_t = w_t.to(device)
            w_tgt = w_tgt.to(device)
            
            # 1. Forward through frozen encoder & chunker
            with torch.no_grad():
                h_t = encoder.get_normalized_embeddings(w_t)
                h_tgt = encoder.get_normalized_embeddings(w_tgt)
                for layer in encoder.layers:
                    h_t = layer(h_t, None)
                    h_tgt = layer(h_tgt, None)
                h_t = encoder.final_ln(h_t)
                h_tgt = encoder.final_ln(h_tgt)
                
                # Step-function representation
                x_in = chunker(h_t)
                x_tgt = chunker(h_tgt)
            
            # 2. Extract Macro Thought with Mamba
            # x_in: [B, L, D] -> z_macro: [B, D]
            z_macro = mamba(x_in)
            
            # 3. Optimal Transport Conditional Flow Matching (OT-CFM)
            # Sample t ~ U(0, 1)
            B, L, D = x_tgt.shape
            t = torch.rand(B, device=device)
            
            # Expand t for broadcasting
            t_expand = t.view(B, 1, 1).expand(B, L, D)
            
            # Sample noise
            x_0 = torch.randn_like(x_tgt)
            
            # Flow interpolation: x_t = t * x_tgt + (1 - t) * x_0
            x_t = t_expand * x_tgt + (1 - t_expand) * x_0
            
            # Target velocity field v = x_tgt - x_0
            v_target = x_tgt - x_0
            
            # 4. Predict Velocity Field with DiT
            optimizer.zero_grad()
            v_pred = dit(x_t, t, z_macro)
            
            # Compute loss
            loss = criterion(v_pred, v_target)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            
            global_step += 1
            
            if global_step % 50 == 0:
                print(f"Epoch {epoch+1} | Step {global_step} | CFM Loss: {loss.item():.4f}")

        # Save checkpoint
        torch.save(mamba.state_dict(), f"{ckpt_dir}/mamba_epoch_{epoch+1}.pt")
        torch.save(dit.state_dict(), f"{ckpt_dir}/dit_epoch_{epoch+1}.pt")
        torch.save({
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'global_step': global_step
        }, f"{ckpt_dir}/meta_epoch_{epoch+1}.pt")
        print(f"Saved checkpoint for epoch {epoch+1}.")

if __name__ == "__main__":
    train()
