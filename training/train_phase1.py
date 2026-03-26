import argparse
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from model.config import ModelConfig
from model.adapter import SensoryFuser
from model.god_encoder import GodEncoder
from model.mamba_planner import MambaPlanner
from model.mamba_mlx.mamba_mlx import MambaConfig
from training.core.dataloader import Phase1DataLoader
from training.core.checkpoint import Checkpointer
from training.core.loss import coverage_loss, momentum_continuity_loss
from training.core.args import get_training_parser

def main():
    # 1. Parse Args (Re-use the same arguments as Phase 0, add Phase 1 specifics)
    parser = get_training_parser("Phase 1 Mamba Training")
    parser.add_argument("--max_episode_len", type=int, default=None, help="If set, strictly bounds and pads sequences to this fixed length (without breaching documents).")
    parser.add_argument("--p0_ckpt", type=str, required=True, help="Path to the frozen Phase 0 checkpoint directory (e.g. checkpoints/run/p0_v1_step_160000)")
    parser.add_argument("--residual_mode", action="store_true", help="If True, Mamba predicts delta velocity instead of absolute coordinates")
    
    args = parser.parse_args()

    # 2. Config & Setup
    config = ModelConfig()
    config.z_dim = args.z_dim
    d_model = config.decoder_heads * 64

    # 3. Instantiate Phase 0 Models (FROZEN)
    fuser = SensoryFuser(config.emb_dims, d_model)
    god_encoder = GodEncoder(d_model, config.z_dim)
    
    print(f"Loading Frozen Phase 0 weights from {args.p0_ckpt}...")
    import os
    fuser_path = f"{args.p0_ckpt}/sense_fuser.safetensors"
    if not os.path.exists(fuser_path):
        fuser_path = f"{args.p0_ckpt}/sense_adapter.safetensors"
    fuser.load_weights(fuser_path)
    god_encoder.load_weights(f"{args.p0_ckpt}/god_encoder.safetensors")
    
    # Freeze them
    fuser.freeze()
    god_encoder.freeze()
    
    # 4. Instantiate Phase 1 Mamba Planner (TRAINABLE)
    mamba_cfg = MambaConfig(d_model=d_model, n_layers=2) # Default small mamba for testing
    mamba_planner = MambaPlanner(mamba_cfg, config.z_dim, residual_mode=args.residual_mode)
    mx.eval(mamba_planner.parameters())

    # 5. Dataloader for Trajectories
    emb_files = [
        "data/Basic_ZH/embs/hy-tmp/roberta_embeddings.npy",
        "data/Basic_ZH/embs/hy-tmp/gte_embeddings.npy",
        "data/Basic_ZH/embs/hy-tmp/bge_embeddings.npy",
        "data/Basic_ZH/embs/hy-tmp/text2vec_embeddings.npy"
    ]
    
    dataloader = Phase1DataLoader(
        parquet_path="data/Basic_ZH/chunked_mixed_wiki.parquet",
        emb_paths=emb_files,
        batch_size=args.batch_size,
        max_episode_len=args.max_episode_len
    )

    # 6. Checkpointer
    checkpointer = Checkpointer(args.out_dir, prefix=args.ckpt_prefix)
    checkpointer.register_model("mamba_planner", mamba_planner)
    checkpointer.register_dataloader("dataloader_p1", dataloader)
    checkpointer.register_args(args)
    
    start_step = 0
    if args.resume_from:
        start_step = checkpointer.load(args.resume_from)
    elif args.auto_resume:
        start_step = checkpointer.load_latest()

    # 7. Optimizer
    optimizer = optim.AdamW(learning_rate=args.lr)

    # 8. Loss Closure
    def loss_fn(model, f_t_input, z_target_truth, mask):
        # 8b. Mamba predicts the trajectory
        # Input to Mamba is the sensory stream f_t_input
        mu, logvar, _ = model(f_t_input) # mu, logvar shape: (B, L, z_dim)
        
        # 8c. Calculate Losses with dynamically padded sequence masks
        l_cov = coverage_loss(mu, logvar, z_target_truth, mask=mask)
        l_mom = momentum_continuity_loss(mu, mask=mask)
        
        # 8d. Total Loss Fusion
        # You can tune the momentum alpha later
        total_loss = l_cov + 0.1 * l_mom
        
        return total_loss, (l_cov, l_mom)

    step_fn = nn.value_and_grad(mamba_planner, loss_fn)

    # 9. Training Loop
    print(f"Starting Phase 1 Training. Epochs: {args.epochs}, Batch Size: {args.batch_size}, Dynamic Document Lengths (Mamba Masked)")
    global_step = start_step

    try:
        for epoch in range(dataloader.current_epoch, args.epochs):
            for batch_embs, masks in dataloader:
                global_step += 1
                
                # 8a. Generate frozen targets using Phase 0 (OUTSIDE the gradient tape)
                f_t = fuser(batch_embs, weights=None) # Centroid Mean for static truth
                z_target = god_encoder(f_t) # Shape: (B, L, z_dim)
                
                (total_loss, aux_losses), grads = step_fn(mamba_planner, f_t, z_target, masks)
                optimizer.update(mamba_planner, grads)
                mx.eval(mamba_planner.parameters(), optimizer.state)
                
                l_cov, l_mom = aux_losses
                
                if global_step % 10 == 0:
                    print(f"Epoch {epoch+1} | Step {global_step} | Total: {total_loss.item():.4f} | Cov: {l_cov.item():.4f} | Mom: {l_mom.item():.4f}")
                    
                if global_step % args.save_steps == 0:
                    checkpointer.save(global_step)
                    
    except KeyboardInterrupt:
        # Checkpointer handles the emergency trap
        checkpointer.save(global_step, is_emergency=True)

if __name__ == "__main__":
    main()
