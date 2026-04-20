import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import argparse
import time
import os
import sys
import json
from functools import partial

# Ensure workspace root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from training.core.dataloader import ChunkedNpzDataLoader
from training.core.char_tokenizer import CharTokenizer
from training.core.checkpoint import Checkpointer
from training.core.schedule import linear_warmup_schedule
from model.config import ModelConfig
from model.adapter import SensoryFuser
from model.god_encoder import GodEncoder
from model.decoder import WeakDecoder

def main():
    parser = argparse.ArgumentParser(description="Phase 0 V2: Unifying 5-Way Embeddings to God Space via WeakDecoder")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size (keep low for 10240d to avoid MLX Metal OOM)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3, help="Total epochs")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints/run", help="Checkpoint directory")
    parser.add_argument("--ckpt_prefix", type=str, default="p0_v2_5models", help="Prefix for checkpoints")
    parser.add_argument("--data_dir", type=str, default="./embs", help="Local cache directory for ModelScope downloads")
    parser.add_argument("--save_every", type=int, default=10000, help="Save checkpoint every N steps")
    parser.add_argument("--log_every", type=int, default=10, help="Log metrics every N steps")
    parser.add_argument("--config_file", type=str, default="dataset_config.json", help="Path to 5-model config")
    parser.add_argument("--resume_from", type=str, default=None, help="Specific checkpointer folder to resume from")
    parser.add_argument("--auto_resume", action=argparse.BooleanOptionalAction, default=True, help="Auto resume from latest checkpoint. Pass --no-auto_resume to disable.")
    
    # [BoW Restoration] Fine-grained loss weighting
    parser.add_argument("--ce_weight", type=float, default=1.0, help="Cross Entropy Baseline Weight")
    parser.add_argument("--bow_weight", type=float, default=0.5, help="Weight for Soft N-Gram BoW Reward")
    
    # [V1 Feature Restoration] Dynamic Warmup Schedulers
    parser.add_argument("--bow_max_n", type=int, default=5, help="Maximum length for N-Gram continuous reward matches")
    parser.add_argument("--warmup_steps", type=int, default=5000, help="Steps for LR to ramp up")
    parser.add_argument("--bow_warmup_steps", type=int, default=10000, help="Steps for BoW penalty to ramp up")
    
    args = parser.parse_args()

    print(f"\n{'='*50}\n🚀 Phase 0 V2: Ignition (Apple Silicon MLX)\n{'='*50}")

    # 1. Configuration & Dataloader
    with open(args.config_file, "r") as f:
        ds_config = json.load(f)
        
    models = ds_config["base_models"]
    emb_dims = ds_config["emb_dims"]
    chunk_patterns = ds_config["chunk_name_patterns"]
    parquet_path = ds_config["parquet_path"]
    ms_repo_id = ds_config["modelscope_repo_id"]

    tokenizer = CharTokenizer("training/core/char_vocab.json")
    
    print("⏳ Auto-Assembling Chunked Dataloader...")
    dataloader = ChunkedNpzDataLoader(
        parquet_path=parquet_path,
        models=models,
        chunk_patterns=chunk_patterns,
        tokenizer=tokenizer,
        ms_repo_id=ms_repo_id,
        chunk_size=500000,
        local_npz_dir=args.data_dir, 
        cache_dir=args.data_dir,
        batch_size=args.batch_size,
        max_seq_len=512,  # Keep the strong 512 context limit
        shuffle=True,     # Double shuffle enabled
        lazy_start=True,  # Prevent early fetch and unconsumed cleanup on Resume
        backend='mlx'
    )

    # 2. Model Initialization
    m_config = ModelConfig()
    d_model = m_config.d_model
    vocab_size = tokenizer.vocab_size
    
    print("🧠 Forging Phase 0 V2 Neural Core...")
    fuser = SensoryFuser(emb_dims, d_model)
    god_encoder = GodEncoder(d_model, m_config.z_dim)
    # We use a purely causal 4-layer decoder to translate Z back to text
    weak_decoder = WeakDecoder(m_config.z_dim, vocab_size, d_model=256, n_layers=4)
    
    class Phase0V2Composite(nn.Module):
        def __init__(self, fuser, god_enc, dec):
            super().__init__()
            self.fuser = fuser
            self.god_encoder = god_enc
            self.decoder = dec
            
        def __call__(self, texts, teacher_embs, active_w_bow, bow_max_n):
            z_fused = self.fuser(teacher_embs)
            z_god = self.god_encoder(z_fused)
            
            inputs = texts[:, :-1]
            targets = texts[:, 1:]
            
            logits = self.decoder(z_god, inputs)
            
            from training.losses.loss import decoder_reconstruction_loss
            ce_loss, bow_loss_val = decoder_reconstruction_loss(
                logits, 
                targets, 
                mask=None, 
                bow_weight=active_w_bow, 
                bow_max_n=bow_max_n
            )
            
            # Apply explicit CE scaling from CLI so ce_weight materially affects optimization.
            total_loss = args.ce_weight * ce_loss - bow_loss_val
            return total_loss, ce_loss, bow_loss_val

    model_composite = Phase0V2Composite(fuser, god_encoder, weak_decoder)
    mx.eval(model_composite.parameters())
    
    optimizer = optim.AdamW(learning_rate=args.lr)

    # 3. Checkpointer Setup
    os.makedirs(args.ckpt_dir, exist_ok=True)
    checkpointer = Checkpointer(out_dir=args.ckpt_dir, prefix=args.ckpt_prefix)
    checkpointer.register_model("sense_fuser", fuser)
    checkpointer.register_model("god_encoder", god_encoder)
    checkpointer.register_model("weak_decoder", weak_decoder)
    checkpointer.register_optimizer("optimizer", optimizer)
    checkpointer.register_dataloader("dataloader", dataloader)
    checkpointer.register_args(args)

    global_step = 0
    if args.resume_from:
        global_step = checkpointer.load(args.resume_from)
    elif args.auto_resume:
        # Auto-resume latest if exists
        global_step = checkpointer.load_latest()

    # 4. Training Core Logic
    state = [model_composite.state, optimizer.state]

    def loss_fn(model, texts, teacher_embs, active_w_bow, bow_max_n):
        total_loss, ce_loss, bow_loss = model(texts, teacher_embs, active_w_bow, bow_max_n)
        return total_loss, (total_loss, ce_loss, bow_loss)

    loss_and_grad = nn.value_and_grad(model_composite, loss_fn)

    @partial(mx.compile, inputs=state, outputs=state)
    def step(texts, teacher_embs, current_lr, active_w_bow, bow_max_n):
        optimizer.learning_rate = current_lr
        
        (_total_loss, (total_loss, ce_loss, bow_loss)), grads = loss_and_grad(
            model_composite, texts, teacher_embs, active_w_bow, bow_max_n
        )
        
        optimizer.update(model_composite, grads)
        return total_loss, ce_loss, bow_loss


    # 5. The Training Loop (Secured with Try-Except)
    print("\n🔥 Training sequence initialized. Brace for impact...")
    try:
        # Dataloader handles epoch tracking internally via StopIteration
        while dataloader.current_epoch < args.epochs:
            for texts, teacher_embs, masks in dataloader:
                t0 = time.perf_counter()
                
                # [V1 Feature Restoration] Dynamic Warmup Math
                current_lr = mx.array(linear_warmup_schedule(global_step, args.lr, args.warmup_steps))
                
                # V1 BoW Warmup: ramp up the BoW penalty factor
                current_w_bow = args.bow_weight
                if args.bow_warmup_steps > 0 and global_step < args.bow_warmup_steps:
                    current_w_bow = args.bow_weight * (global_step / args.bow_warmup_steps)
                current_w_bow = mx.array(current_w_bow)
                
                loss, ce_loss, bow_loss = step(texts, teacher_embs, current_lr, current_w_bow, args.bow_max_n)
                
                mx.eval(state)
                
                loss_val = loss.item()
                ce_val = ce_loss.item()
                bow_val = bow_loss.item()
                t1 = time.perf_counter()
                
                global_step += 1
                
                # Logging
                if global_step % args.log_every == 0:
                    throughput = texts.shape[0] / (t1 - t0)
                    print(f"Epoch {dataloader.current_epoch} | Step {global_step} | Total: {loss_val:.4f} (CE: {ce_val:.4f}, BoW: {bow_val:.4f}) | {throughput:.1f} seq/sec")
                
                # Checkpointing
                if global_step % args.save_every == 0:
                    checkpointer.save(global_step)
                    
        # Final save
        print("🎉 Training sequence complete!")
        checkpointer.save(global_step)
            
    except BaseException as e:
        # [EMERGENCY PROTOCOL] Catch ANY exception (OOM, Terminal Stop, Segfault, Watchdog)
        # Checkpointer already traps SIGINT, but this catches Python-level OOMs cleanly
        import traceback
        print(f"\n❌ [FATAL ALARM] Exception caught during main loop: {e}")
        traceback.print_exc()
        checkpointer.save(global_step, is_emergency=True)
        raise e

if __name__ == "__main__":
    main()
