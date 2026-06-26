# =========================================================================
# [ARCHIVED] LEGACY SCRIPT: Z-Manifold Evaluator
#
# [Reason for Archival]:
# This script was used to calculate properties (mean, variance, distance) of the 
# SINGLE pooled Z vectors in the latent space. Since we are moving to Sequence 
# Flow Matching where the manifold consists of [L, D] sequence trajectories 
# instead of isolated points, this specific evaluation script is deprecated.
# =========================================================================

import torch
import torch.nn.functional as F
import argparse
import os
import math
from tqdm import tqdm

from training.core.dataloader import TextDocumentDataLoader
from training.core.char_tokenizer import CharTokenizer
# from training.train_phase1_cuda import E2EModel
from training.archive.train_phase1_cuda import E2EModel
from distilled_emb.model_cuda import TinyCharEncoderCUDA, load_mlx_safetensors_into_torch

def euler_ode_solve(flow_matcher, h_context, d_model=1024, num_steps=50, cfg_scale=1.0, device="cuda"):
    x_t = torch.randn(1, 1, d_model, device=device, dtype=torch.float32)
    dt = 1.0 / num_steps
    for i in range(num_steps):
        t_val = i / num_steps
        t_tensor = torch.full((1, 1, 1), t_val, device=device, dtype=torch.float32)
        with torch.cuda.amp.autocast():
            v_pred = flow_matcher.predict_with_cfg(x_t, t_tensor, h_context, cfg_scale=cfg_scale)
        x_t = x_t + v_pred.float() * dt
    return x_t

def main():
    parser = argparse.ArgumentParser(description="Evaluate Mamba Flow Matcher Latent Trajectory")
    parser.add_argument("--phase0_encoder_ckpt", type=str, default="checkpoints/distilled/tinybert_pt_v1_step_100000", help="Path to TinyEncoder PyTorch checkpoint folder")
    parser.add_argument("--phase1_ckpt", type=str, default="checkpoints/run/first_maflow_step_199000", help="Path to Phase 1 E2E checkpoint folder")
    parser.add_argument("--data_dir", type=str, default="./embs", help="Path to NPZ dataset")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of trajectories to evaluate")
    parser.add_argument("--seq_len", type=int, default=5, help="Length of chunks per trajectory")
    parser.add_argument("--ode_steps", type=int, default=50, help="Steps for Flow Matching ODE")
    parser.add_argument("--cfg_scale", type=float, default=1.0, help="CFG scale for Flow Matching")
    
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Initializing Manifold Evaluation on {device}...")

    # 1. Load Tokenizer & DataLoader (val split if possible, but here we just take random samples)
    tokenizer = CharTokenizer("training/core/char_vocab.json")
    dataloader = TextDocumentDataLoader(
        parquet_path="data/Basic_ZH/chunked_mixed_omni.parquet",
        tokenizer=tokenizer,
        batch_size=1, # One trajectory at a time
        max_seq_len=64, # Matches Phase 1
        max_episode_len=args.seq_len, backend="torch"
    )
    
    # 2. Load Models
    tiny_encoder = TinyCharEncoderCUDA(vocab_size=tokenizer.vocab_size, d_model=1024).to(device)
    encoder_pt = os.path.join(args.phase0_encoder_ckpt, "student.pt")
    tiny_encoder.load_state_dict(torch.load(encoder_pt, map_location=device, weights_only=True))
    tiny_encoder.eval()
    
    
    from model.mamba_planner_cuda import MambaPlanner
    # from model.flow_matcher_cuda import FlowMatcher
    from model.archive.flow_matcher_cuda import FlowMatcher
    # from training.train_phase1_cuda import E2EModel
    from training.archive.train_phase1_cuda import E2EModel
    
    mamba_planner = MambaPlanner(d_model=1024, d_state=16, d_conv=4, expand=2)
    flow_matcher = FlowMatcher(d_model=1024, hidden_dim=2048)
    e2e_model = E2EModel(mamba_planner, flow_matcher).to(device)

    e2e_pt = os.path.join(args.phase1_ckpt, "e2e_model.pt")
    e2e_model.load_state_dict(torch.load(e2e_pt, map_location=device, weights_only=True))
    e2e_model.eval()
    
    print("✅ Models loaded successfully. Starting evaluation...\n")
    
    total_sim = 0.0
    total_random_sim = 0.0
    total_naive_sim = 0.0 # Just copying the previous step
    
    valid_samples = 0
    
    # We manually fetch sequential chunks
    it = iter(dataloader)
    
    for i in range(args.num_samples):
        try:
            ids_t, att_t, sen_t = next(it)
            if ids_t.shape[1] < args.seq_len:
                print(f"Skip sample {i}, sequence too short.")
                continue
                
            ids_t = ids_t[:, :args.seq_len, :].to(device)
            att_t = att_t[:, :args.seq_len, :].to(device)
            sen_t = sen_t[:, :args.seq_len].to(device) # sen_t is (B, seq) usually
            
            # 3. Ground Truth Trajectory computation
            z_truths = []

            with torch.no_grad():
                for j in range(args.seq_len):
                    ids = ids_t[:, j, :]
                    att = att_t[:, j, :]
                    z_true = tiny_encoder(ids, attention_mask=att) # (1, 1024)

                    z_truths.append(z_true)
            
            z_truths = torch.stack(z_truths, dim=1) # (1, seq_len, 1024)
            
            print(f"\n--- 📍 Trajectory {i+1} / {args.num_samples} ---")
            
            # 4. Autoregressive Prediction
            z_stream = z_truths[:, 0:1, :] # Start with first chunk
            
            traj_sim = 0.0
            traj_random_sim = 0.0
            traj_naive_sim = 0.0
            
            with torch.no_grad():
                for step in range(1, args.seq_len):
                    z_target = z_truths[:, step:step+1, :]
                    
                    # Mamba Predicts Context
                    with torch.cuda.amp.autocast():
                        h_context = e2e_model.mamba(z_stream) # (1, S, 1024)
                        h_curr = h_context[:, -1:, :] # Take the last context vector
                    
                    # Flow Matcher Predicts Next Coordinate
                    z_pred = euler_ode_solve(e2e_model.flow, h_curr, d_model=1024, num_steps=args.ode_steps, cfg_scale=args.cfg_scale, device=device)
                    
                    # Metrics (Cosine Similarity)
                    sim = F.cosine_similarity(z_pred.squeeze(), z_target.squeeze(), dim=0).item()
                    
                    # Baselines
                    # Baseline 1: Random Vector (to see expected sim of random point in 1024D, should be ~0)
                    random_vec = torch.randn_like(z_target)
                    rand_sim = F.cosine_similarity(random_vec.squeeze(), z_target.squeeze(), dim=0).item()
                    
                    # Baseline 2: Naive Copy (is Mamba actually progressing the story, or just copying the last context?)
                    last_z = z_stream[:, -1:, :]
                    naive_sim = F.cosine_similarity(last_z.squeeze(), z_target.squeeze(), dim=0).item()
                    
                    print(f"  Step {step}->{step+1} | Mamba vs Truth: {sim:.4f} | Naive (Copy Last): {naive_sim:.4f} | Random: {rand_sim:.4f}")
                    
                    traj_sim += sim
                    traj_naive_sim += naive_sim
                    traj_random_sim += rand_sim
                    
                    # Append PREDICTED z for the next autoregressive step (Teacher Forcing = False)
                    # To be strict, we feed its own prediction back in
                    z_stream = torch.cat([z_stream, z_pred], dim=1)
                
            steps_evaluated = args.seq_len - 1
            avg_traj_sim = traj_sim / steps_evaluated
            total_sim += avg_traj_sim
            total_naive_sim += traj_naive_sim / steps_evaluated
            total_random_sim += traj_random_sim / steps_evaluated
            valid_samples += 1
            
            print(f"  --> Average Trajectory Score: {avg_traj_sim:.4f}")
            
        except StopIteration:
            break
            
    if valid_samples > 0:
        print(f"\n{'='*50}")
        print(f"🎯 FINAL MANIFOLD METRICS (Avg over {valid_samples} trajectories):")
        print(f"  1. Mamba+Flow (Ours) Cosine Sim : {total_sim / valid_samples:.4f}")
        print(f"  2. Naive Copy Baseline Cosine Sim: {total_naive_sim / valid_samples:.4f}")
        print(f"  3. Random Vector Cosine Sim    : {total_random_sim / valid_samples:.4f}")
        print(f"{'='*50}\n")
        print("💡 Interpretation:")
        print(" - High dimensional (>1000) random vectors are almost perfectly orthogonal (Sim ~ 0.0).")
        print(" - A sequence of text evolves continuously. Naive Copy will have a high score because adjacent sentences are similar.")
        print(" - If our model consistently beats the Naive Copy baseline, it means Mamba is ACCURATELY predicting the story's evolution direction!")
    
if __name__ == "__main__":
    main()
