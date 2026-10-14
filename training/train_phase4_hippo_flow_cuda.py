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
from training.data.moss_dataset import MOSSDataset
from model.hippo_oracle import HiPPOOracleCUDA
from model.flow_unet_cuda import FlowMatching1DUnet
from model.loss_hippo_cfm import hippo_cfm_loss
from model.phase1_charencoder_cuda import TinyCharEncoderCUDA
from training.core.args import get_training_parser
from training.core.checkpoint import Checkpointer
from torch.utils.data import DataLoader

def train():
    parser = get_training_parser("Phase 4 HiPPO Oracle + Flow Match Engine (CUDA)")
    args = parser.parse_args()
    
    tokenizer = CharTokenizer()
    dataset = MOSSDataset(
        jsonl_path="data/Basic_ZH/moss_dialogues.jsonl",
        tokenizer=tokenizer,
        seq_len=args.seq_len
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    # Phase 1 Encoder (Frozen Continuous Geometric Mapper)
    encoder = TinyCharEncoderCUDA(vocab_size=tokenizer.vocab_size).to(device)
    ckpt_path = "checkpoints/basencoder_step_115000/student.pt"
    if os.path.exists(ckpt_path):
        encoder.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    encoder.eval()
    for p in encoder.parameters(): p.requires_grad = False
        
    # Phase 4 Specifics
    # 1. HiPPO Oracle (Frozen)
    hippo_oracle = HiPPOOracleCUDA(hidden_dim=1024, N=64).to(device)
    hippo_oracle.eval()
    
    # 2. Flow Match Unet (Trainable)
    unet = FlowMatching1DUnet(hidden_dim=1024, cond_dim=1024, num_layers=6).to(device)
    
    optimizer = optim.AdamW(unet.parameters(), lr=args.lr)
    
    checkpointer = Checkpointer(out_dir=args.out_dir, prefix=args.ckpt_prefix, keep_last_k=args.keep_last_k)
    checkpointer.register_model("unet", unet)
    checkpointer.register_optimizer("engine_opt", optimizer)
    checkpointer.register_args(args)
    
    global_step = checkpointer.load_latest() if args.auto_resume else 0
    start_epoch = global_step // len(dataloader) if len(dataloader) > 0 else 0
    
    print("Starting Phase 4 HiPPO + Flow Match Training (Curriculum Learning)...")
    
    # Curriculum Schedule: We increase the probability of using knowledge attractors over time.
    # At start (0), it's 0% chance. At the end, it caps at 50% chance.
    max_steps = args.epochs * len(dataloader)
    
    for epoch in range(start_epoch, args.epochs):
        resumed_loader = checkpointer.wrap_dataloader(dataloader, epoch, global_step)
        unet.train()
        
        for batch_idx, (w_t, w_tgt) in enumerate(resumed_loader):
            w_t = w_t.to(device)
            w_tgt = w_tgt.to(device)
            
            with torch.no_grad():
                h_t = encoder.get_normalized_embeddings(w_t)
                for layer in encoder.layers: h_t = layer(h_t, None)
                x_prompt = encoder.final_ln(h_t)
                
                h_tgt = encoder.get_normalized_embeddings(w_tgt)
                for layer in encoder.layers: h_tgt = layer(h_tgt, None)
                x_target = encoder.final_ln(h_tgt)
            
            optimizer.zero_grad()
            
            # Curriculum: probability of injecting knowledge increases linearly from 0 to 0.5
            p_attractor = min(0.5, (global_step / max_steps) * 1.0)
            use_attractor = torch.rand(1).item() < p_attractor
            
            loss = hippo_cfm_loss(unet, hippo_oracle, x_prompt, x_target, use_knowledge_attractor=use_attractor)
            loss.backward()
            optimizer.step()
            
            global_step += 1
            if global_step % 50 == 0:
                print(f"Epoch {epoch+1} | Step {global_step} | Phase 4 Loss: {loss.item():.4f} | Attractor Prob: {p_attractor:.2f}")

        checkpointer.save(global_step)

if __name__ == "__main__":
    train()
