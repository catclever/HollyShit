import torch
import torch.nn as nn
import torch.optim as optim
import os
import math
from torch.utils.data import DataLoader
from training.core.char_tokenizer import CharTokenizer
from training.data.omni_dataset import OmniDataset
from model.phase4_differential_planner_cuda import DifferentialPlannerCUDA, loss_fn_cuda
from model.concept_chunker import ConceptChunker
from model.phase1_charencoder_cuda import TinyCharEncoderCUDA
from training.core.args import get_training_parser
from training.core.checkpoint import Checkpointer

def prepare_differential_batch_cuda(encoder, chunker, w_t, max_trajectory_len=128):
    """
    Extracts chunks, computes Delta Z and lengths, and pads to max_trajectory_len (Pure PyTorch).
    """
    with torch.no_grad():
        h_t = encoder.get_normalized_embeddings(w_t)
        for layer in encoder.layers:
            h_t = layer(h_t, None)
        h_t = encoder.final_ln(h_t)
        
        B, L, D = h_t.shape
        h_norm = torch.nn.functional.normalize(h_t, p=2, dim=-1)
        
        all_delta_z = []
        all_masks = []
        device = h_t.device
        
        punct_ids = {262, 838, 317, 474, 10, 46, 33, 63, 1562}
        
        for b in range(B):
            h_seq = h_t[b]
            tok_seq = w_t[b].tolist()
            
            centroids = []
            
            current_sum = torch.zeros_like(h_seq[0])
            current_count = 0
            
            for i in range(L):
                current_sum += h_seq[i]
                current_count += 1
                
                if tok_seq[i] in punct_ids or i == L - 1:
                    centroids.append(current_sum / current_count)
                    
                    current_sum = torch.zeros_like(h_seq[0])
                    current_count = 0
            
            if len(centroids) < 2:
                centroids.append(centroids[-1])
                
            Z = torch.stack(centroids)
            delta_Z = Z[1:] - Z[:-1]
            
            seq_len = min(len(delta_Z), max_trajectory_len)
            padded_delta = torch.zeros((max_trajectory_len, D), device=device)
            
            padded_delta[:seq_len] = delta_Z[:seq_len]
            
            # Create valid mask for this sequence
            seq_mask = torch.zeros(max_trajectory_len, device=device)
            seq_mask[:seq_len] = 1.0
            
            all_delta_z.append(padded_delta)
            all_masks.append(seq_mask)
            
        batch_delta_z = torch.stack(all_delta_z)
        mask = torch.stack(all_masks)
        
    return batch_delta_z, mask

def train():
    parser = get_training_parser("Phase 4 Differential Planner Training (CUDA)")
    
    # Phase 4 Specific Data Pipeline Parameters
    parser.add_argument("--max_trajectory_len", type=int, default=128, help="Maximum number of semantic chunks per document")
    
    args = parser.parse_args()
    
    tokenizer = CharTokenizer()
    dataset = OmniDataset(
        omni_parquet_path="data/chunked_mixed_omni.parquet",
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        dynamic_padding=True
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.dynamic_collate)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    encoder = TinyCharEncoderCUDA(vocab_size=tokenizer.vocab_size).to(device)
    ckpt_path = "checkpoints/basencoder_step_115000/student.pt"
    if os.path.exists(ckpt_path):
        encoder.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        print("Loaded frozen TinyCharEncoder.")
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
        
    chunker = ConceptChunker(threshold=0.5).to(device)
    
    planner = DifferentialPlannerCUDA(d_model=args.z_dim).to(device)
    optimizer = optim.AdamW(planner.parameters(), lr=args.lr)
    
    checkpointer = Checkpointer(out_dir=args.out_dir, prefix=args.ckpt_prefix, keep_last_k=args.keep_last_k)
    checkpointer.register_model("planner", planner)
    checkpointer.register_optimizer("optimizer", optimizer)
    checkpointer.register_args(args)
    
    start_epoch = 0
    global_step = 0
    if args.resume_from:
        global_step = checkpointer.load(args.resume_from)
    elif args.auto_resume:
        global_step = checkpointer.load_latest()
        
    args = checkpointer._args
    
    if global_step > 0:
        start_epoch = global_step // len(dataloader)
        
    dataloader = checkpointer.wrap_dataloader(dataloader, start_epoch, global_step)
    
    print("Starting Phase 4 Differential Planner Training (CUDA)...")
    
    for epoch in range(start_epoch, args.epochs):
        if epoch > start_epoch:
            dataloader = checkpointer.wrap_dataloader(dataloader, epoch, checkpointer.current_step)
            
        planner.train()
        for batch_idx, (w_t, _) in enumerate(dataloader):
            w_t = w_t.to(device)
            
            dz, mask = prepare_differential_batch_cuda(encoder, chunker, w_t, max_trajectory_len=args.max_trajectory_len)
            
            optimizer.zero_grad()
            delta_loss = loss_fn_cuda(planner, dz, mask)
            
            delta_loss.backward()
            optimizer.step()
            
            current_step = checkpointer.step()
            
            if current_step % 50 == 0:
                print(f"Epoch {epoch+1} | Step {current_step} | Semantic Delta Loss: {delta_loss.item():.6f}")
                
if __name__ == "__main__":
    train()
