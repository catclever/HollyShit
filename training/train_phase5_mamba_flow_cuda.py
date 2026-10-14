import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir in sys.path and os.path.exists(os.path.join(script_dir, "model.py")):
    sys.path = [p for p in sys.path if p != script_dir] + [script_dir]

from training.core.char_tokenizer import CharTokenizer
from training.data.mixed_dataset import MixedDataset
from model.flow_mamba import MambaReader
from model.hippo_oracle import HiPPOOracleCUDA
from model.loss_mamba_cfm import mamba_cfm_loss
from model.phase1_charencoder_cuda import TinyCharEncoderCUDA
from training.core.args import get_training_parser
from training.core.checkpoint import Checkpointer
from torch.utils.data import DataLoader

def train():
    parser = get_training_parser("Phase 5 Mamba Predictive Planner (CUDA)")
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
        
    # Phase 1 Encoder (Frozen)
    encoder = TinyCharEncoderCUDA(vocab_size=tokenizer.vocab_size).to(device)
    ckpt_path = "checkpoints/basencoder_step_115000/student.pt"
    if os.path.exists(ckpt_path):
        encoder.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    encoder.eval()
    for p in encoder.parameters(): p.requires_grad = False
        
    # Phase 4 Oracle (Frozen)
    hippo_oracle = HiPPOOracleCUDA(hidden_dim=1024, N=64).to(device)
    hippo_oracle.eval()
    
    # Phase 5 Mamba (Trainable)
    mamba = MambaReader(d_model=1024, num_layers=4).to(device)
    optimizer = optim.AdamW(mamba.parameters(), lr=args.lr)
    
    checkpointer = Checkpointer(out_dir=args.out_dir, prefix=args.ckpt_prefix, keep_last_k=args.keep_last_k)
    checkpointer.register_model("mamba", mamba)
    checkpointer.register_optimizer("engine_opt", optimizer)
    checkpointer.register_args(args)
    
    global_step = checkpointer.load_latest() if args.auto_resume else 0
    start_epoch = global_step // len(dataloader) if len(dataloader) > 0 else 0
    
    print("Starting Phase 5 Mamba Latent Prediction Training...")
    
    for epoch in range(start_epoch, args.epochs):
        resumed_loader = checkpointer.wrap_dataloader(dataloader, epoch, global_step)
        mamba.train()
        
        for batch_idx, (w_t, w_tgt) in enumerate(resumed_loader):
            w_t = w_t.to(device)
            w_tgt = w_tgt.to(device)
            
            with torch.no_grad():
                h_t = encoder.get_normalized_embeddings(w_t)
                for layer in encoder.layers: h_t = layer(h_t, None)
                x_context = encoder.final_ln(h_t)
                
                h_tgt = encoder.get_normalized_embeddings(w_tgt)
                for layer in encoder.layers: h_tgt = layer(h_tgt, None)
                x_target = encoder.final_ln(h_tgt)
            
            optimizer.zero_grad()
            loss = mamba_cfm_loss(mamba, hippo_oracle, x_context, x_target)
            loss.backward()
            optimizer.step()
            
            global_step += 1
            if global_step % 50 == 0:
                print(f"Epoch {epoch+1} | Step {global_step} | Phase 5 Loss: {loss.item():.4f}")

        checkpointer.save(global_step)

if __name__ == "__main__":
    train()
