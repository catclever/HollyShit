import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import time
import os
import sys
import json
import torch
from training.core.char_tokenizer import CharTokenizer
from training.data.phase2_mixed_dataset import Phase2_MixedDataset
from model.phase2_mamba_mlx import MambaReaderMLX
from model.phase2_dit_mlx import ConceptDiTMLX
from model.concept_chunker import ConceptChunker
from model.phase1_charencoder_cuda import TinyCharEncoderCUDA
from training.core.args import get_training_parser
from training.core.checkpoint import Checkpointer
from torch.utils.data import DataLoader

def loss_fn(mamba, dit, x_in, x_tgt):
    """
    OT-CFM Flow Matching Loss
    """
    B, L, D = x_tgt.shape
    
    # Extract z_macro
    z_macro = mamba(x_in) # [B, D]
    
    # Sample t ~ U(0, 1)
    t = mx.random.uniform(shape=(B,))
    t_expand = mx.expand_dims(mx.expand_dims(t, 1), 2) # [B, 1, 1]
    
    # Sample noise
    x_0 = mx.random.normal(x_tgt.shape)
    
    # Interpolation
    x_t = t_expand * x_tgt + (1 - t_expand) * x_0
    
    # Target velocity field
    v_target = x_tgt - x_0
    
    # Predict velocity field
    v_pred = dit(x_t, t, z_macro)
    
    # MSE Loss
    loss = mx.mean(mx.square(v_pred - v_target))
    return loss

class Phase2ModelMLX(nn.Module):
    """Wrapper to easily optimize both Mamba and DiT together"""
    def __init__(self, mamba, dit):
        super().__init__()
        self.mamba = mamba
        self.dit = dit

    def __call__(self, x_in, x_tgt):
        return loss_fn(self.mamba, self.dit, x_in, x_tgt)

def train():
    parser = get_training_parser("Phase 2 Thought Engine Training (MLX)")
    args = parser.parse_args()
    
    tokenizer = CharTokenizer()
    dataset = Phase2_MixedDataset(
        wiki_parquet_path="data/Basic_ZH/wiki_filtered.parquet",
        dialogues_jsonl_path="data/Basic_ZH/dialogues_254k.jsonl",
        tokenizer=tokenizer,
        seq_len=args.seq_len
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # We use PyTorch to run the frozen CharEncoder and Chunker on CPU/GPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
        
    encoder = TinyCharEncoderCUDA(vocab_size=tokenizer.vocab_size).to(device)
    ckpt_path = "checkpoints/basencoder_step_115000/student.pt"
    if os.path.exists(ckpt_path):
        encoder.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        print("Loaded frozen TinyCharEncoder (PyTorch).")
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
        
    chunker = ConceptChunker(threshold=0.5).to(device)
    
    # Phase 2 MLX Models
    mamba = MambaReaderMLX(d_model=1024, num_layers=4)
    dit = ConceptDiTMLX(d_model=1024, max_len=128, num_layers=6, num_heads=8)
    
    model = Phase2ModelMLX(mamba, dit)
    mx.eval(model.parameters())
    
    def model_forward(m, x, y):
        return m(x, y)
        
    optimizer = optim.AdamW(learning_rate=args.lr)
    loss_and_grad_fn = nn.value_and_grad(model, model_forward)
    
    @mx.compile
    def train_step(x, y):
        return loss_and_grad_fn(model, x, y)

    
    checkpointer = Checkpointer(out_dir=args.out_dir, prefix=args.ckpt_prefix, keep_last_k=args.keep_last_k)
    checkpointer.register_model("mamba", mamba)
    checkpointer.register_model("dit", dit)
    checkpointer.register_optimizer("engine_opt", optimizer)
    checkpointer.register_args(args)
    
    global_step = 0
    if args.resume_from:
        args = checkpointer.load_args(args.resume_from, args)
        global_step = checkpointer.load(args.resume_from)
        print(f'Resumed at global step {global_step}')
    elif args.auto_resume:
        global_step = checkpointer.load_latest()
    
        
    start_epoch = global_step // len(dataloader) if len(dataloader) > 0 else 0
    
    print("Starting Phase 2 Thought Engine Training (MLX)...")
    
    for epoch in range(start_epoch, args.epochs):
        resumed_loader = checkpointer.wrap_dataloader(dataloader, epoch, global_step)
        for batch_idx, (w_t, w_tgt) in enumerate(resumed_loader):
            w_t = w_t.to(device)
            w_tgt = w_tgt.to(device)
            
            # Extract features via frozen PyTorch frontend
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
            
            # Move to MLX
            mx_x_in = mx.array(x_in.cpu().numpy())
            mx_x_tgt = mx.array(x_tgt.cpu().numpy())
            
            # Train step in MLX (Compiled loss and grad)
            loss, grads = train_step(mx_x_in, mx_x_tgt)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)
            
            global_step += 1
            
            if global_step % 50 == 0:
                print(f"Epoch {epoch+1} | Step {global_step} | CFM Loss: {loss.item():.4f}")

        checkpointer.save(global_step)
        print(f"Saved checkpoint for epoch {epoch+1} at global step {global_step}.")

if __name__ == "__main__":
    train()
