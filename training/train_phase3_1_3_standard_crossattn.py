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


def compute_resonance_and_diffs(mx_c, window=32, res_thresh=0.5):
    import mlx.core as mx
    B, L, D = mx_c.shape
    c_norm = mx_c / mx.maximum(mx.linalg.norm(mx_c, axis=-1, keepdims=True), 1e-8)
    S = mx.matmul(c_norm, c_norm.transpose(0, 2, 1))
    indices = mx.arange(L)
    dist = mx.abs(indices.reshape(-1, 1) - indices.reshape(1, -1))
    window_mask = dist <= window
    S = mx.where(window_mask, mx.zeros_like(S), S)
    token_scores = mx.sum(S > res_thresh, axis=-1)
    diffs = mx.sum(mx.abs(mx_c[:, 1:] - mx_c[:, :-1]), axis=-1)
    return token_scores, diffs

def extract_anchors_batch(h_np, token_scores_np, diffs_np, tokens_np, tokenizer, num_concept_blocks=10):
    B, L, D = h_np.shape
    stop_words = set(['的', '了', '在', '是', '和', '与', '、', '，', '。', '（', '）', ' ', '于', '有', '为', '\n'])
    
    batch_anchors_h = []
    batch_anchors_t = []
    max_k = 0
    
    for b in range(B):
        token_scores = token_scores_np[b]
        diffs = diffs_np[b]
        boundaries = [0] + (np.where(diffs > 1e-4)[0] + 1).tolist() + [L]
        
        scored_blocks = []
        for i in range(len(boundaries) - 1):
            start_idx = boundaries[i]
            end_idx = boundaries[i+1]
            if end_idx <= start_idx:
                continue
                
            is_valid = False
            for j in range(start_idx, end_idx):
                char = tokenizer.decode([tokens_np[b, j]])
                if char not in stop_words and char.strip():
                    is_valid = True
                    break
                    
            if is_valid:
                block_score = token_scores[start_idx]
                scored_blocks.append((block_score, start_idx, end_idx))
                
        scored_blocks.sort(key=lambda x: x[0], reverse=True)
        chosen_blocks = scored_blocks[:num_concept_blocks]
        
        b_h = []
        b_t = []
        for block_score, start_idx, end_idx in chosen_blocks:
            for i in range(start_idx, end_idx):
                b_h.append(h_np[b, i])
                b_t.append(tokens_np[b, i])
                
        batch_anchors_h.append(b_h)
        batch_anchors_t.append(b_t)
        if len(b_t) > max_k:
            max_k = len(b_t)
            
    if max_k == 0:
        max_k = 1
        
    padded_anchors_h = np.zeros((B, max_k, D), dtype=np.float32)
    padded_anchors_t = np.full((B, max_k), -1, dtype=np.int32)
    
    for b in range(B):
        length = len(batch_anchors_t[b])
        if length > 0:
            padded_anchors_h[b, :length, :] = batch_anchors_h[b]
            padded_anchors_t[b, :length] = batch_anchors_t[b]
            
    import mlx.core as mx
    return mx.array(padded_anchors_h), mx.array(padded_anchors_t)

def loss_fn(model, x, targets, pad_id, text_noise_prob=0.0):
    noisy_targets = targets
    if text_noise_prob > 0:
        rand = mx.random.uniform(shape=targets.shape)
        should_mask = rand < text_noise_prob
        col_mask = mx.arange(targets.shape[1]) > 0
        should_mask = should_mask & col_mask & (targets != pad_id)
        noisy_targets = mx.where(should_mask, pad_id, targets)
        
    logits = model(x, noisy_targets) # [B, L-1, V]
    labels = targets[:, 1:]
    
    logits_flat = logits.reshape(-1, logits.shape[-1])
    labels_flat = labels.reshape(-1)
    
    loss = nn.losses.cross_entropy(logits_flat, labels_flat)
    mask = (labels_flat != pad_id).astype(logits.dtype)
    loss = (loss * mask).sum() / mask.sum()
    return loss


import math

def precompute_anchors_cpu(h_np, c_zoh_np, tokens_np, tokenizer, num_concept_blocks, window=32, res_thresh=0.5):
    # Pure CPU version for a chunk of documents
    B, L, D = h_np.shape
    stop_words = set(['的', '了', '在', '是', '和', '与', '、', '，', '。', '（', '）', ' ', '于', '有', '为', '\n'])
    
    # Normalize
    norms = np.linalg.norm(c_zoh_np, axis=-1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    c_norm = c_zoh_np / norms
    
    batch_anchors_h = []
    batch_anchors_t = []
    max_k = 0
    
    for b in range(B):
        S = np.dot(c_norm[b], c_norm[b].T)
        for i in range(L):
            start = max(0, i - window)
            end = min(L, i + window + 1)
            S[i, start:end] = 0.0
            
        token_scores = (S > res_thresh).sum(axis=-1)
        diffs = np.sum(np.abs(c_zoh_np[b, 1:] - c_zoh_np[b, :-1]), axis=-1)
        boundaries = [0] + (np.where(diffs > 1e-4)[0] + 1).tolist() + [L]
        
        scored_blocks = []
        for i in range(len(boundaries) - 1):
            start_idx = boundaries[i]
            end_idx = boundaries[i+1]
            if end_idx <= start_idx: continue
            
            is_valid = False
            for j in range(start_idx, end_idx):
                char = tokenizer.decode([tokens_np[b, j]])
                if char not in stop_words and char.strip():
                    is_valid = True
                    break
            
            if is_valid:
                scored_blocks.append((token_scores[start_idx], start_idx, end_idx))
                
        scored_blocks.sort(key=lambda x: x[0], reverse=True)
        chosen_blocks = scored_blocks[:num_concept_blocks]
        
        b_h = []
        b_t = []
        for block_score, start_idx, end_idx in chosen_blocks:
            for i in range(start_idx, end_idx):
                b_h.append(h_np[b, i])
                b_t.append(tokens_np[b, i])
                
        batch_anchors_h.append(b_h)
        batch_anchors_t.append(b_t)
        if len(b_t) > max_k: max_k = len(b_t)
        
    if max_k == 0: max_k = 1
    
    padded_anchors_h = np.zeros((B, max_k, D), dtype=np.float32)
    padded_anchors_t = np.full((B, max_k), -1, dtype=np.int32)
    
    for b in range(B):
        length = len(batch_anchors_t[b])
        if length > 0:
            padded_anchors_h[b, :length, :] = batch_anchors_h[b]
            padded_anchors_t[b, :length] = batch_anchors_t[b]
            
    return padded_anchors_h, padded_anchors_t

def super_batch_generator(dataloader, base_encoder, chunker, tokenizer, args):
    buffer_w = []
    buffer_is_punct = []
    
    punctuations_set = set('。，、；：！？\n ()（）《》「」"\'')
    punct_ids = [tokenizer.encode(p)[0] for p in punctuations_set if len(tokenizer.encode(p)) > 0]
    punct_ids_np = np.array(punct_ids, dtype=np.int32)
    
    for w_t, _ in dataloader:
        w_t_np = w_t.numpy()
        w_is_punct = np.isin(w_t_np, punct_ids_np)
        
        buffer_w.append(w_t_np)
        buffer_is_punct.append(w_is_punct)
        
        current_docs = sum(x.shape[0] for x in buffer_w)
        
        if current_docs >= args.cpu_batch_size:
            w_super = np.concatenate(buffer_w, axis=0)[:args.cpu_batch_size]
            punct_super = np.concatenate(buffer_is_punct, axis=0)[:args.cpu_batch_size]
            
            # Keep the remainder
            rem_w = np.concatenate(buffer_w, axis=0)[args.cpu_batch_size:]
            rem_punct = np.concatenate(buffer_is_punct, axis=0)[args.cpu_batch_size:]
            buffer_w = [rem_w] if rem_w.shape[0] > 0 else []
            buffer_is_punct = [rem_punct] if rem_punct.shape[0] > 0 else []
            
            # 1. Process super-batch on GPU to get h and c_zoh
            # We must process it in chunks on GPU to avoid OOM, e.g. chunks of args.batch_size
            h_list = []
            c_list = []
            
            num_chunks = math.ceil(w_super.shape[0] / args.batch_size)
            for i in range(num_chunks):
                chunk_w = w_super[i*args.batch_size : (i+1)*args.batch_size]
                chunk_punct = punct_super[i*args.batch_size : (i+1)*args.batch_size]
                
                mx_w = mx.array(chunk_w)
                mx_punct = mx.array(chunk_punct)
                
                h = base_encoder(mx_w)
                c = chunker(h, mx_punct)
                
                h_list.append(np.array(h))
                c_list.append(np.array(c))
                
            h_np = np.concatenate(h_list, axis=0)
            c_np = np.concatenate(c_list, axis=0)
            
            # 2. Extract anchors on CPU
            anchors_h, anchors_t = precompute_anchors_cpu(h_np, c_np, w_super, tokenizer, args.num_concept_blocks)
            
            # 3. Yield mini-batches ready for Decoder training
            indices = np.arange(w_super.shape[0])
            np.random.shuffle(indices) # Shuffle mini-batches
            
            for i in range(num_chunks):
                batch_idx = indices[i*args.batch_size : (i+1)*args.batch_size]
                
                b_c = c_np[batch_idx]
                b_w = w_super[batch_idx]
                b_ah = anchors_h[batch_idx]
                b_at = anchors_t[batch_idx]
                
                yield mx.array(b_c), b_w, mx.array(b_ah), mx.array(b_at)


def train():
    import wandb
    wandb.init(project="HollyShit", name="phase3_1_3_standard_crossattn")
    parser = get_training_parser("Phase 3.1.3 Standard Cross Attention Decoder Training (MLX)")
    # No decoder_ckpt because this architecture doesn't match Phase 2.1
    parser.add_argument('--encoder_ckpt', type=str, default="checkpoints/basencoder_step_115000/student.pt", help="Path to Phase 1 encoder")
    
    from model.standard_crossattn_decoder_mlx import StandardDecoderConfig
    from training.core.config_utils import build_parser_from_config, extract_config_from_args, load_config
    
    parser = build_parser_from_config(StandardDecoderConfig, parser)
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
        config = load_config(resume_ckpt_path, StandardDecoderConfig, filename="decoder_config.json")
        if config:
            print(f"Loaded StandardDecoderConfig from {resume_ckpt_path}/decoder_config.json")
            
    if config is None:
        config = extract_config_from_args(args, StandardDecoderConfig)
        
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
    encoder.freeze()
    
    # Concept Chunker with higher threshold (e.g. 0.8) to extract precise phrases
    chunker = ConceptChunkerMLX(threshold=0.8)
    
    @mx.compile
    def encode_step(w):
        h = encoder.get_normalized_embeddings(w)
        for layer in encoder.layers:
            h = layer(h, None)
        return encoder.final_ln(h)
    
    from model.standard_crossattn_decoder_mlx import StandardCrossAttnDecoder
    decoder = StandardCrossAttnDecoder(config=config)
    
    # We can't easily load Phase 2.1 weights here because StandardCrossAttnDecoder has Cross-Attention layers
    # that did not exist in Phase 2.1. So it trains from scratch.
    mx.eval(decoder.parameters())
    
    def lr_schedule(step):
        return mx.where(step < 2000, args.lr * (step / 2000), args.lr)
        
    optimizer = optim.AdamW(learning_rate=lr_schedule)
    
    loss_and_grad_fn = nn.value_and_grad(decoder, loss_fn)
    
    @mx.compile
    def train_step(model_state, opt_state, c, t, pad_id, noise_min, noise_max, text_noise_prob=0.15):
        B = c.shape[0]
        noise_scale = mx.random.uniform(low=noise_min, high=noise_max, shape=(B, 1, 1))
        noise = mx.random.normal(shape=c.shape) * noise_scale
        c = c + noise
        
        decoder.update(model_state)
        optimizer.state = opt_state
        
        loss, grads = loss_and_grad_fn(decoder, c, t, pad_id, text_noise_prob)
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
            super_gen = super_batch_generator(epoch_dataloader, encode_step, chunker, tokenizer, args)
            
            for step, (mx_c, w_t_np, mx_anchors_h, mx_anchors_t) in enumerate(super_gen):
                if step >= steps_per_epoch:
                    break
                    
                B, L = w_t_np.shape
                
                # Prepare target tokens
                for b in range(B):
                    pad_indices = np.where(w_t_np[b] == tokenizer.pad_token_id)[0]
                    if len(pad_indices) > 0:
                        w_t_np[b, pad_indices[0]] = tokenizer.eos_token_id
                
                bos_col = np.full((B, 1), tokenizer.bos_token_id, dtype=np.int32)
                target_tokens_np = np.concatenate([bos_col, w_t_np], axis=1)
                mx_t = mx.array(target_tokens_np)
                
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
                    
                    # Generate text
                    gen_ids = decoder.generate(
                        sample_c, 
                        tokenizer.bos_token_id, 
                        max_len=args.seq_len, 
                        temperature=0.0, 
                        repetition_penalty=1.2
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
