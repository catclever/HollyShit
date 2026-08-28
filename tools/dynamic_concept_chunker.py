import argparse
import pandas as pd
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from tqdm import tqdm
import sys
import os

# Ensure we can import from model
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from training.core.char_tokenizer import CharTokenizer
from model.phase1_charencoder_mlx import TinyCharEncoderMLX
from tools.weight_utils import load_torch_weights_into_mlx

def physical_chunking(text, max_len=512):
    """Fallback physical chunking based on punctuation and length."""
    import re
    if not isinstance(text, str) or not text.strip(): return []
    pattern = re.compile(r'([^。！？!?\n]+)([。！？!?\n]*)')
    chunks = []
    current_chunk = ""
    for match in pattern.finditer(text):
        content = match.group(1).strip()
        delims = match.group(2)
        if not content and not delims: continue
        piece = content + delims
        if len(current_chunk) + len(piece) > max_len and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = piece
        else:
            current_chunk += piece
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def extract_concept_boundaries(h_np, diffs_np, tokens_np, tokenizer):
    B, L, D = h_np.shape
    batch_anchors_t = []
    
    for b in range(B):
        diffs = diffs_np[b]
        # Find boundaries where diff > 1e-4
        boundaries = [0] + (np.where(diffs > 1e-4)[0] + 1).tolist() + [L]
        
        b_t = []
        for i in range(len(boundaries) - 1):
            start_idx = boundaries[i]
            end_idx = boundaries[i+1]
            if end_idx <= start_idx: continue
            
            # Reconstruct the string for this concept block
            block_tokens = tokens_np[b, start_idx:end_idx]
            block_text = tokenizer.decode(block_tokens.tolist())
            b_t.append(block_text)
            
        batch_anchors_t.append(b_t)
    return batch_anchors_t

def compute_resonance_and_diffs(mx_c, window=32, res_thresh=0.5):
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/Basic_ZH/cleaned_mixed_wiki.parquet")
    parser.add_argument("--output", type=str, default="data/Basic_ZH/concept_chunked.parquet")
    parser.add_argument("--compute_resonance", action="store_true", help="If set, computes resonance using encoder")
    parser.add_argument("--ckpt", type=str, default="checkpoints/basencoder_step_115000/student.pt")
    parser.add_argument("--vocab", type=str, default="data/Basic_ZH/char_vocab.json")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=512)
    args = parser.parse_args()

    print(f"Loading data from {args.input}...")
    df = pd.read_parquet(args.input)
    # df = df.head(100) # Fast test mode if needed
    
    if not args.compute_resonance:
        print("Only performing physical punctuation chunking...")
        tqdm.pandas(desc="Chunking")
        df['concept_chunks'] = df['text'].progress_apply(lambda x: physical_chunking(x, args.seq_len))
    else:
        print(f"Loading Tokenizer from {args.vocab}...")
        tokenizer = CharTokenizer(args.vocab)
        
        print(f"Loading Encoder from {args.ckpt}...")
        encoder = TinyCharEncoderMLX(vocab_size=tokenizer.vocab_size)
        load_torch_weights_into_mlx(encoder, args.ckpt)
        encoder.eval()
        
        @mx.compile
        def encode_batch(x):
            h = encoder.get_normalized_embeddings(x)
            for layer in encoder.layers:
                h = layer(h, None)
            h = encoder.final_ln(h)
            _, diffs = compute_resonance_and_diffs(h)
            return h, diffs
            
        print("Performing physical chunking first...")
        df['chunks'] = df['text'].apply(lambda x: physical_chunking(x, args.seq_len))
        
        # Flatten chunks to process them through the model
        flat_chunks = []
        for row_idx, chunks in enumerate(df['chunks']):
            for c in chunks:
                flat_chunks.append((row_idx, c))
                
        print(f"Total physical chunks to process: {len(flat_chunks)}")
        
        all_concept_blocks = {}
        
        # Process in batches
        for i in tqdm(range(0, len(flat_chunks), args.batch_size), desc="Computing Resonance"):
            batch_items = flat_chunks[i:i+args.batch_size]
            batch_texts = [item[1] for item in batch_items]
            
            # Tokenize and pad
            batch_tokens = []
            max_len = 0
            for t in batch_texts:
                tokens = tokenizer.encode(t, add_special_tokens=False)[:args.seq_len]
                batch_tokens.append(tokens)
                if len(tokens) > max_len: max_len = len(tokens)
                
            if max_len == 0: continue
            
            # Pad
            padded_tokens = np.full((len(batch_texts), max_len), tokenizer.pad_token_id, dtype=np.int32)
            for j, tk in enumerate(batch_tokens):
                padded_tokens[j, :len(tk)] = tk
                
            mx_x = mx.array(padded_tokens)
            h, diffs = encode_batch(mx_x)
            
            # Extract concept boundaries
            h_np = np.array(h)
            diffs_np = np.array(diffs)
            
            batch_concept_texts = extract_concept_boundaries(h_np, diffs_np, padded_tokens, tokenizer)
            
            for j, (row_idx, _) in enumerate(batch_items):
                if row_idx not in all_concept_blocks:
                    all_concept_blocks[row_idx] = []
                all_concept_blocks[row_idx].append(batch_concept_texts[j])
                
        # Assign back to dataframe
        df['concept_chunks'] = df.index.map(lambda idx: all_concept_blocks.get(idx, []))
        df = df.drop(columns=['chunks'])
        
    df['chunk_count'] = df['concept_chunks'].apply(lambda x: len(x) if isinstance(x, list) else 0)
    valid_df = df[df['chunk_count'] > 0].copy()
    
    print(f"\nFinished processing. Valid rows: {len(valid_df)}")
    print(f"Saving to {args.output}...")
    valid_df.to_parquet(args.output, engine='pyarrow', index=False)
    print("Done!")

if __name__ == '__main__':
    main()
