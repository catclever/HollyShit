import os
import argparse
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from training.char_tokenizer import CharTokenizer
from model.config import ModelConfig
from model.adapter import SensoryFuser
from model.god_encoder import GodEncoder
from model.decoder import WeakDecoder
from training.core.dataloader import MultiEmbDataLoader
from training.losses.loss import decoder_reconstruction_loss
from training.core.checkpoint import Checkpointer
from training.core.schedule import linear_warmup_schedule
from training.core.args import get_training_parser

def main():
    parser = get_training_parser(description="Phase 0: GodEncoders + WeakDecoder Alignment Training")
    
    # Phase-0 Specific Arguments
    parser.add_argument("--bow_weight", type=float, default=0.5, help="Weight for the Soft N-Gram BoW Reward (0.0 to disable)")
    parser.add_argument("--bow_max_n", type=int, default=5, help="Maximum length for N-Gram continuous reward matches")
    parser.add_argument("--bow_warmup_steps", type=int, default=10000, help="Linearly scale BoW weight from 0 over X steps")
    
    args = parser.parse_args()

    # 1. Config & Tokenizer
    tokenizer = CharTokenizer()
    config = ModelConfig()
    config.z_dim = args.z_dim
    config.vocab_size = tokenizer.vocab_size

    # 2. Models Setup
    d_model = config.decoder_heads * 64
    fuser = SensoryFuser(config.emb_dims, d_model)
    god_encoder = GodEncoder(d_model, config.z_dim)
    decoder = WeakDecoder(config.z_dim, config.vocab_size, d_model=d_model, n_layers=config.decoder_layers)
    
    class Phase0Composite(nn.Module):
        def __init__(self, fuser, god_enc, dec):
            super().__init__()
            self.fuser = fuser
            self.god_encoder = god_enc
            self.decoder = dec
            
        def __call__(self, embs, tokens):
            # Phase 0 Specific Training Strategy: Stochastic Weighted Routing (0.7, 0.1...)
            # We enforce the specific weighting logic HERE in the training script
            if self.training:
                import random
                N = len(embs)
                alpha = args.fusion_alpha
                rem_weight = (1.0 - alpha) / (N - 1) if N > 1 else 0.0
                
                weights = [rem_weight] * N
                main_idx = random.randint(0, N - 1)
                weights[main_idx] = alpha
                f_t = self.fuser(embs, weights=weights)
            else:
                f_t = self.fuser(embs, weights=None)
                
            z_target = self.god_encoder(f_t)
            logits = self.decoder(z_target, tokens)
            return logits
            
    model_composite = Phase0Composite(fuser, god_encoder, decoder)
    mx.eval(model_composite.parameters())
    print("Model composite initialized.")

    # 3. Dataloader
    emb_files = [
        "data/Basic_ZH/embs/hy-tmp/roberta_embeddings.npy",
        "data/Basic_ZH/embs/hy-tmp/gte_embeddings.npy",
        "data/Basic_ZH/embs/hy-tmp/bge_embeddings.npy",
        "data/Basic_ZH/embs/hy-tmp/text2vec_embeddings.npy"
    ]
    
    dataloader = MultiEmbDataLoader(
        parquet_path="data/Basic_ZH/chunked_mixed_wiki.parquet",
        emb_paths=emb_files,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_seq_len=config.max_seq_len,
        shuffle=True
    )

    # 4. Checkpointer Abstraction (Replaces manual saving and handles Interrupts)
    checkpointer = Checkpointer(args.out_dir, prefix=args.ckpt_prefix)
    
    checkpointer.register_model("sense_fuser", fuser)
    checkpointer.register_model("god_encoder", god_encoder)
    checkpointer.register_model("decoder", decoder)
    checkpointer.register_dataloader("dataloader", dataloader)
    checkpointer.register_args(args)
    
    start_step = 0
    if args.resume_from:
        start_step = checkpointer.load(args.resume_from)
    elif args.auto_resume:
        start_step = checkpointer.load_latest()

    # 5. Optimizer
    optimizer = optim.AdamW(learning_rate=args.lr)

    # 6. Loss Function closure
    def loss_fn(model, embs, input_ids, target_ids, target_mask, active_bow_weight):
        logits = model(embs, input_ids)
        ce_l, bow_l = decoder_reconstruction_loss(
            logits, 
            target_ids, 
            mask=target_mask, 
            bow_weight=active_bow_weight,
            bow_max_n=args.bow_max_n
        )
        return ce_l - bow_l

    step_fn = nn.value_and_grad(model_composite, loss_fn)

    # 7. Training Loop 
    print(f"Starting Phase 0 Training. Epochs: {args.epochs}, Batch Size: {args.batch_size}")
    global_step = start_step

    try:
        for epoch in range(dataloader.current_epoch, args.epochs):
            for token_inputs, batch_embs, attention_mask in dataloader:
                
                # Dynamic BoW Warmup
                if args.bow_warmup_steps > 0 and global_step < args.bow_warmup_steps:
                    current_bow_weight = args.bow_weight * (global_step / args.bow_warmup_steps)
                else:
                    current_bow_weight = args.bow_weight
                    
                input_ids = token_inputs[:, :-1]
                target_ids = token_inputs[:, 1:]
                target_mask = attention_mask[:, 1:]
                
                # Sync LR natively with global step
                optimizer.learning_rate = linear_warmup_schedule(global_step, args.lr, args.warmup_steps)

                loss, grads = step_fn(model_composite, batch_embs, input_ids, target_ids, target_mask, current_bow_weight)
                
                optimizer.update(model_composite, grads)
                mx.eval(model_composite.parameters(), optimizer.state)
                
                global_step += 1
                
                if global_step % 10 == 0:
                    print(f"Epoch {epoch+1} | Step {global_step} | L_recon: {loss.item():.4f}")
                    
                if global_step % args.save_steps == 0:
                    checkpointer.save(global_step)
                    
    except KeyboardInterrupt:
        # The Checkpointer's universal signal trap raised KeyboardInterrupt
        checkpointer.save(global_step, is_emergency=True)

if __name__ == "__main__":
    main()
