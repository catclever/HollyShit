import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import time
import os
import sys

# Auto-resolve the parent workspace root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir in sys.path and os.path.exists(os.path.join(script_dir, "model.py")):
    sys.path = [p for p in sys.path if p != script_dir] + [script_dir]
    
from training.core.char_tokenizer import CharTokenizer

from model.phase1_charencoder_mlx import TinyCharEncoderMLX
from model.concept_chunker_mlx import ConceptChunkerMLX
from model.crossattn_decoder import StrongConceptDecoder
from training.core.args import get_training_parser
from training.core.checkpoint import Checkpointer
from tools.weight_utils import load_torch_weights_into_mlx

# Removed pad_chunks as it is no longer needed.


def train():
    import wandb
    wandb.init(project="HollyShit", name="phase2_2_convdecoder")
    parser = get_training_parser("Phase 2.2 Parallel Convolutional Decoder Training (MLX)")
    parser.add_argument('--encoder_ckpt', type=str, default="checkpoints/basencoder_step_115000/student.pt", help="Path to Phase 1 encoder")
    
    from model.conv_decoder_mlx import ConvDecoderConfig
    from training.core.config_utils import build_parser_from_config, extract_config_from_args, save_config, load_config
    
    parser = build_parser_from_config(ConvDecoderConfig, parser)
    args = parser.parse_args()
    
    # 1. Initialize Checkpointer and intercept args before model instantiation
    from training.core.checkpoint import Checkpointer
    checkpointer = Checkpointer(out_dir=args.out_dir, prefix=args.ckpt_prefix, keep_last_k=args.keep_last_k)
    
    resume_ckpt_path = None
    if args.auto_resume:
        resume_ckpt_path = checkpointer.get_latest_checkpoint_path()
    elif args.resume_from:
        resume_ckpt_path = args.resume_from
        
    config = None
    if resume_ckpt_path:
        args = checkpointer.load_args(resume_ckpt_path, args)
        config = load_config(resume_ckpt_path, ConvDecoderConfig, filename="decoder_config.json")
        if config:
            print(f"Loaded ConvDecoderConfig from {resume_ckpt_path}/decoder_config.json")
    
    if config is None:
        config = extract_config_from_args(args, ConvDecoderConfig)
        
    tokenizer = CharTokenizer()
    config.vocab_size = tokenizer.vocab_size
    from data.phase1_5_omni_dataset import Phase1_5_OmniDataset
    
    dataset = Phase1_5_OmniDataset(
        omni_parquet_path="data/Basic_ZH/chunked_mixed_omni.parquet",
        tokenizer=tokenizer,
        seq_len=args.seq_len
    )
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    from model.phase1_charencoder_mlx import TinyCharEncoderMLX
    
    encoder = TinyCharEncoderMLX(vocab_size=tokenizer.vocab_size)
    if os.path.exists(args.encoder_ckpt):
        load_torch_weights_into_mlx(encoder, args.encoder_ckpt)
        print(f"Loaded encoder from {args.encoder_ckpt}")
        
    encoder.freeze()
    
    @mx.compile
    def encode_step(w):
        h = encoder.get_normalized_embeddings(w)
        for layer in encoder.layers:
            h = layer(h, None)
        return encoder.final_ln(h)
    
    from model.conv_decoder_mlx import ParallelConvDecoder
    decoder = ParallelConvDecoder(config)
    
    # Train from scratch since it's a completely new architecture
    mx.eval(decoder.parameters())
    
    def lr_schedule(step):
        return mx.where(step < 2000, args.lr * (step / 2000), args.lr)
        
    optimizer = optim.AdamW(learning_rate=lr_schedule)
    
    loss_and_grad_fn = nn.value_and_grad(decoder, loss_fn)
    
    @mx.compile
    def train_step(model_state, opt_state, c, t, pad_id, noise_min, noise_max):
        B = c.shape[0]
        noise_scale = mx.random.uniform(low=noise_min, high=noise_max, shape=(B, 1, 1))
        noise = mx.random.normal(shape=c.shape) * noise_scale
        c = c + noise
        
        decoder.update(model_state)
        optimizer.state = opt_state
        
        loss, grads = loss_and_grad_fn(decoder, c, t, pad_id)
        clipped_grads, _ = optim.clip_grad_norm(grads, 1.0)
        optimizer.update(decoder, clipped_grads)
        
        return loss, decoder.parameters(), optimizer.state

    checkpointer.register_model("decoder", decoder)
    checkpointer.register_optimizer("decoder_opt", optimizer)
    checkpointer.register_args(args)
    
    global_step = 0
    if resume_ckpt_path:
        global_step = checkpointer.load(resume_ckpt_path)
        
    start_epoch = global_step // len(dataloader) if len(dataloader) > 0 else 0
    import time
    step_start_time = time.time()
    
    punctuations_set = set('。，、；：！？\n ()（）《》「」"\'')
    try:
        for epoch in range(start_epoch, args.epochs):
            steps_per_epoch = len(dataloader)
            epoch_dataloader = checkpointer.wrap_dataloader(dataloader, epoch, global_step)
            
            for step, (w_t_pt, _) in enumerate(epoch_dataloader):
                if step >= steps_per_epoch:
                    break
                    
                w_t_np = w_t_pt.numpy()
                B, L = w_t_np.shape
                
                # Prepare target tokens
                for b in range(B):
                    pad_indices = np.where(w_t_np[b] == tokenizer.pad_token_id)[0]
                    if len(pad_indices) > 0:
                        w_t_np[b, pad_indices[0]] = tokenizer.eos_token_id
                
                # For parallel NAT, target is just the tokens, no BOS needed
                mx_t = mx.array(w_t_np)
                
                # 1. Forward pass through Phase 1 Encoder to get Contextual Hidden States
                mx_c = encode_step(mx_t)
                
                loss, new_model_state, new_opt_state = train_step(
                    decoder.parameters(),
                    optimizer.state,
                    mx_c, mx_t, tokenizer.pad_token_id,
                    args.noise_min, args.noise_max
                )
                
                decoder.update(new_model_state)
                optimizer.state = new_opt_state
                mx.eval(decoder.parameters(), optimizer.state, loss)
                
                global_step += 1
                wandb.log({"loss": loss.item()}, step=global_step)
                
                if global_step % 100 == 0:
                    elapsed = time.time() - step_start_time
                    print(f"Epoch {epoch+1} | Step {global_step} | Loss: {loss.item():.4f} | Time/100steps: {elapsed:.2f}s")
                    step_start_time = time.time()
                    
                if global_step % 1000 == 0:
                    sample_c = mx_c[0:1] # [1, L, D]
                    sample_t = mx_t[0]   # [L+1]
                    sample_anchors_h = mx_anchors_h[0:1]
                    sample_anchors_t = mx_anchors_t[0:1]
                    
                    # Generate text in parallel!
                    gen_ids = decoder.generate(
                        sample_c, 
                        temperature=0.0
                    )
                    mx.eval(gen_ids)
                    
                    orig_tokens = [i.item() for i in np.array(sample_t)]
                    gen_tokens = [i.item() for i in np.array(gen_ids[0])]
                    
                    orig_text = tokenizer.decode(orig_tokens, skip_special_tokens=True)
                    gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
                    
                    print(f"\n--- EVAL STEP {global_step} ---")
                    print(f"Target: {orig_text}")
                    print(f"Generated: {gen_text}")
                    print("-" * 25 + "\n")
                    
                if hasattr(args, 'save_steps') and args.save_steps > 0 and global_step % args.save_steps == 0:
                    checkpointer.save(global_step)
                    print(f"Saved intermediate checkpoint at global step {global_step}.")
                    
    except KeyboardInterrupt:
        print("Training interrupted manually.")
        checkpointer.save(global_step)
        
    checkpointer.save(global_step)
    print("Training finished.")

if __name__ == '__main__':
    train()
