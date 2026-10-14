import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys

# Auto-resolve the parent workspace root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir in sys.path and os.path.exists(os.path.join(script_dir, "model.py")):
    sys.path = [p for p in sys.path if p != script_dir] + [script_dir]

from training.core.char_tokenizer import CharTokenizer
from training.data.mixed_dataset import MixedDataset
from model.flow_mamba import MambaReader
from model.flow_unet_cuda import FlowMatching1DUnet
from model.loss_legacy_joint import ot_cfm_loss_legacy
from model.concept_chunker import ConceptChunker
from model.phase1_charencoder_cuda import TinyCharEncoderCUDA
from training.core.args import get_training_parser
from training.core.checkpoint import Checkpointer
from torch.utils.data import DataLoader

def train():
    parser = get_training_parser("Phase 4 Flow Matching Latent Engine (CUDA)")
    args = parser.parse_args()
    
    tokenizer = CharTokenizer()
    dataset = MixedDataset(
        wiki_parquet_path="data/Basic_ZH/wiki_filtered.parquet",
        dialogues_jsonl_path="data/Basic_ZH/dialogues_254k.jsonl",
        tokenizer=tokenizer,
        seq_len=args.seq_len
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    encoder = TinyCharEncoderCUDA(vocab_size=tokenizer.vocab_size).to(device)
    ckpt_path = "checkpoints/basencoder_step_115000/student.pt"
    if os.path.exists(ckpt_path):
        encoder.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        print("Loaded frozen TinyCharEncoder (PyTorch).")
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
        
    chunker = ConceptChunker(threshold=0.5).to(device)
    
    # Phase 4 CUDA Models
    mamba = MambaReader(d_model=1024, num_layers=4).to(device)
    unet = FlowMatching1DUnet(hidden_dim=1024, cond_dim=1024, num_layers=6).to(device)
    
    # Combine parameters for optimizer
    optimizer = optim.AdamW(list(mamba.parameters()) + list(unet.parameters()), lr=args.lr)
    
    checkpointer = Checkpointer(out_dir=args.out_dir, prefix=args.ckpt_prefix, keep_last_k=args.keep_last_k)
    checkpointer.register_model("mamba", mamba)
    checkpointer.register_model("unet", unet)
    checkpointer.register_optimizer("engine_opt", optimizer)
    checkpointer.register_args(args)
    
    global_step = 0
    if args.resume_from:
        global_step = checkpointer.load(args.resume_from)
        print(f'Resumed at global step {global_step}')
    elif args.auto_resume:
        global_step = checkpointer.load_latest()
    
    args = checkpointer._args
        
    start_epoch = global_step // len(dataloader) if len(dataloader) > 0 else 0
    
    print("Starting Phase 4 Latent Space Flow Matching Training (CUDA)...")
    
    for epoch in range(start_epoch, args.epochs):
        resumed_loader = checkpointer.wrap_dataloader(dataloader, epoch, global_step)
        
        mamba.train()
        unet.train()
        
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
            
            # Forward and Loss
            optimizer.zero_grad()
            loss = ot_cfm_loss_legacy(unet, mamba, x_in, x_tgt)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            global_step += 1
            
            if global_step % 50 == 0:
                print(f"Epoch {epoch+1} | Step {global_step} | OT-CFM Loss: {loss.item():.4f}")

        checkpointer.save(global_step)
        print(f"Saved checkpoint for epoch {epoch+1} at global step {global_step}.")

if __name__ == "__main__":
    train()
