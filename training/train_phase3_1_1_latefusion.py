import argparse
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import time
import os
import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from training.core.checkpoint import Checkpointer
from training.core.args import get_training_parser
from tools.weight_utils import load_torch_weights_into_mlx

from model.phase1_charencoder_mlx import TinyCharEncoderMLX
from model.latedecoder import LateFusionDecoder

def step(decoder, input_ids, encoder_out, noise_min, noise_max):
    # Add dynamic noise to encoder_out for latent robustness
    if noise_max > 0:
        # Sample random noise scale per sequence in the batch
        B = encoder_out.shape[0]
        noise_scale = mx.random.uniform(low=noise_min, high=noise_max, shape=(B, 1, 1))
        noise = mx.random.normal(encoder_out.shape) * noise_scale
        encoder_out_noised = encoder_out + noise
    else:
        encoder_out_noised = encoder_out
        
    # target_z is the continuous target for the NEXT token, so we shift by 1
    target_z = encoder_out_noised[:, 1:, :]
    
    logits = decoder(
        target_tokens=input_ids,
        target_z=target_z
    )
    
    # Shift tokens for CE Loss
    targets = input_ids[:, 1:]
    loss = mx.mean(nn.losses.cross_entropy(logits, targets))
    return loss, logits

def worker_init_fn(worker_id):
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def train():
    parser = get_training_parser("Phase 3.1 Late Fusion Alignment")
    parser.add_argument("--encoder_ckpt", type=str, default="checkpoints/basencoder_step_115000/student.pt")
    parser.add_argument("--decoder_ckpt", type=str, default="checkpoints/phase2_1_decoder/partial_window_step_45000/decoder.safetensors")
    
    from model.latedecoder import LateFusionConfig
    from training.core.config_utils import build_parser_from_config, extract_config_from_args, load_config
    
    parser = build_parser_from_config(LateFusionConfig, parser)
    
    args = parser.parse_args()
    if args.seed is not None:
        mx.random.seed(args.seed)

    # 1. Initialize Checkpointer
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
        config = load_config(resume_ckpt_path, LateFusionConfig, filename="decoder_config.json")
        if config:
            print(f"Loaded LateFusionConfig from {resume_ckpt_path}/decoder_config.json")
            
    if config is None:
        config = extract_config_from_args(args, LateFusionConfig)

    # Load dataset
    from training.core.char_tokenizer import CharTokenizer
    tokenizer = CharTokenizer()
    config.vocab_size = tokenizer.vocab_size
    from data.omni_dataset import OmniDataset
    from torch.utils.data import DataLoader
    
    dataset = OmniDataset(
        omni_parquet_path="data/Basic_ZH/chunked_mixed_omni.parquet",
        tokenizer=tokenizer,
        seq_len=args.seq_len
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, prefetch_factor=2, persistent_workers=True, worker_init_fn=worker_init_fn)

    # Init Models
    encoder = TinyCharEncoderMLX(
        vocab_size=tokenizer.vocab_size,
        d_model=args.z_dim,
        n_layers=6,
        n_heads=8,
        z_dim=args.z_dim
    )
    
    decoder = LateFusionDecoder(config=config)
    
    # Load frozen encoder weights
    load_torch_weights_into_mlx(encoder, args.encoder_ckpt)
    
    # Disable gradient tracking on encoder entirely
    encoder.freeze()
    
    optimizer = optim.AdamW(learning_rate=args.lr)
    
    # Load Phase 2.1 Decoder weights BEFORE checkpointer load
    if args.decoder_ckpt:
        print(f"Loading pretrained Phase 2.1 Decoder from {args.decoder_ckpt}...")
        decoder.load_weights(args.decoder_ckpt)
        # Optional: We usually freeze the text generation parts to only train the fusion_mlp
        # If Kaelina wants it to fine-tune, she can comment this out.
        # for block in decoder.blocks:
        #     block.freeze()
        # decoder.char_embed.freeze()

    # Checkpointing
    os.makedirs(args.out_dir, exist_ok=True)
    checkpointer = Checkpointer(args.out_dir, prefix=args.ckpt_prefix, keep_last_k=args.keep_last_k)
    checkpointer.register_model("decoder", decoder)
    checkpointer.register_optimizer("optimizer", optimizer)
    checkpointer.register_args(args)
    
    @mx.compile
    def encode_step(w):
        h = encoder.get_normalized_embeddings(w)
        for layer in encoder.layers:
            h = layer(h, None)
        return encoder.final_ln(h)

    step_fn = nn.value_and_grad(decoder, step)
    
    @mx.compile
    def compiled_step(model_state, opt_state, batch_input_ids, encoder_out, noise_min, noise_max):
        decoder.update(model_state)
        optimizer.state = opt_state
        (loss, logits), grads = step_fn(decoder, batch_input_ids, encoder_out, noise_min, noise_max)
        optimizer.update(decoder, grads)
        return loss, decoder.state, optimizer.state
    
    
    global_step = 0
    resume_ckpt_path = None
    if args.auto_resume:
        resume_ckpt_path = checkpointer.get_latest_checkpoint_path()
    elif args.resume_from:
        resume_ckpt_path = args.resume_from
        
    if resume_ckpt_path:
        args = checkpointer.load_args(resume_ckpt_path, args)
        global_step = checkpointer.load(resume_ckpt_path)
    print(f"Starting Phase 1.6 Continuous Autoencoder (Late Fusion) from step {global_step}...")
    t0 = time.time()
    try:
        steps_per_epoch = len(dataloader)
        start_epoch = global_step // steps_per_epoch
        for epoch in range(start_epoch, args.epochs):
            checkpointer.setup_epoch_seed(epoch, args.seed)
            epoch_dataloader = checkpointer.wrap_dataloader(dataloader, epoch, global_step)
            for batch in epoch_dataloader:
                import numpy as np
                # batch is a tuple: (input_ids, target_ids)
                # input_ids is [B, SeqLen] without BOS
                raw_input_ids = batch[0].numpy()
                B, SeqLen = raw_input_ids.shape
                
                # We need to prepend BOS so the decoder has something to start with
                bos_col = np.full((B, 1), tokenizer.bos_token_id, dtype=np.int32)
                # Prepend BOS and truncate the last token to keep sequence length equal to args.seq_len
                input_ids_with_bos = np.concatenate([bos_col, raw_input_ids[:, :-1]], axis=1) # [B, SeqLen]
                input_ids = mx.array(input_ids_with_bos)
                
                # We also need the encoder to process the raw input without BOS (or it doesn't matter much)
                # But to align z_target precisely: target_z for the NEXT token
                # Actually, encoder_out should just match input_ids length.
                # Let's pass the BOS-prepended sequence to the encoder too, so the dimensions match perfectly.
                encoder_out = encode_step(input_ids)
                
                # Decoder forward + backward
                loss, decoder_state, opt_state = compiled_step(decoder.state, optimizer.state, input_ids, encoder_out, args.noise_min, args.noise_max)
                
                decoder.update(decoder_state)
                optimizer.state = opt_state
                mx.eval(decoder.state, optimizer.state, loss)
                
                global_step += 1
                
                if global_step % 100 == 0:
                    step_time = time.time() - t0
                    print(f"Epoch {epoch + 1} | Step {global_step} | Loss: {loss.item():.4f} | Time/100steps: {step_time:.2f}s")
                    t0 = time.time()
                    
                if global_step % 1000 == 0:
                    import numpy as np
                    sample_input = input_ids[0:1]
                    sample_encoder_out = encode_step(sample_input)
                    
                    gen_ids = decoder.generate(
                        encoder_out=sample_encoder_out,
                        bos_token_id=tokenizer.bos_token_id,
                        temperature=0.0
                    )
                    mx.eval(gen_ids)
                    
                    orig_tokens = [i.item() for i in np.array(sample_input[0])]
                    gen_tokens = [i.item() for i in np.array(gen_ids[0])]
                    
                    orig_text = tokenizer.decode(orig_tokens, skip_special_tokens=True)
                    gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
                    
                    print(f"\n--- EVAL STEP {global_step} ---")
                    print(f"Target: {orig_text}")
                    print(f"Generated: {gen_text}")
                    print("-" * 25 + "\n")
                    
                if global_step % args.save_steps == 0:
                    checkpointer.save(global_step)
    except KeyboardInterrupt:
        print("Training interrupted manually.")
        checkpointer.save(global_step)
        
    checkpointer.save(global_step)
    print("Training finished.")

if __name__ == "__main__":
    train()
