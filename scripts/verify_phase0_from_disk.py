import mlx.core as mx
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
import os

from model.config import ModelConfig
from model.adapter import SensoryFuser
from model.god_encoder import GodEncoder
from model.decoder import WeakDecoder

import argparse
import random

def verify():
    parser = argparse.ArgumentParser(description="Phase 0 Auto-regressive Dream Verifier")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of random sentences to pull and dream-reconstruct")
    parser.add_argument("--emb_idx", type=int, default=-1, help="-1: fully fused. 0:roberta, 1:gte, 2:bge, 3:text2vec")
    parser.add_argument("--ckpt", type=str, default="checkpoints/run/p0_v1_step_150000", help="Path to checkpoint directory")
    args = parser.parse_args()

    print("1. Loading physical architecture...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
    
    config = ModelConfig()
    config.vocab_size = 151643 # Hardcoded to match the exact saved checkpoint parameter shape
    d_model = config.decoder_heads * 64
    
    fuser = SensoryFuser(config.emb_dims, d_model)
    god_encoder = GodEncoder(d_model, config.z_dim)
    decoder = WeakDecoder(config.z_dim, config.vocab_size, d_model=d_model, n_layers=config.decoder_layers)
    
    # Note: Using args.ckpt
    ckpt_path = args.ckpt
    print(f"2. Loading weights from {ckpt_path}...")
    
    fuser_path = f"{ckpt_path}/sense_fuser.safetensors"
    if not os.path.exists(fuser_path):
        fuser_path = f"{ckpt_path}/sense_adapter.safetensors"
    fuser.load_weights(fuser_path)
    god_encoder.load_weights(f"{ckpt_path}/god_encoder.safetensors")
    decoder.load_weights(f"{ckpt_path}/decoder.safetensors")
    
    print(f"3. Pulling {args.num_samples} random sentence(s) from Parquet (Source of truth)...")
    df = pd.read_parquet("data/Basic_ZH/chunked_mixed_wiki.parquet")
    text_chunks = df['chunks'].explode().dropna().tolist()
    total_chunks = len(text_chunks)
    
    del df # Free memory
    sampled_indices = random.sample(range(total_chunks), args.num_samples)
    
    print("4. Mmapping all .npy disk arrays...")
    emb_files = [
        "data/Basic_ZH/embs/hy-tmp/roberta_embeddings.npy",
        "data/Basic_ZH/embs/hy-tmp/gte_embeddings.npy",
        "data/Basic_ZH/embs/hy-tmp/bge_embeddings.npy",
        "data/Basic_ZH/embs/hy-tmp/text2vec_embeddings.npy"
    ]
    embs_mmap = [np.load(path, mmap_mode='r') for path in emb_files]
    
    for i, idx in enumerate(sampled_indices):
        target_text = text_chunks[idx]
        print(f"\n======================================")
        print(f"       >>> SAMPLE {i+1} (Row #{idx}) <<<")
        print(f"======================================")
        
        # We slice exactly one row to simulate a batch size of 1
        embs = [mx.array(arr[idx:idx+1]) for arr in embs_mmap]
            
        print("5. Forwarding through SensoryFuser and GodEncoder -> z_target...")
        
        weights = None
        if args.emb_idx != -1:
            weights = [0.0] * 4
            weights[args.emb_idx] = 1.0
            print(f"   (Using EXCLUSIVE external embedding: {emb_files[args.emb_idx].split('/')[-1]})")
        else:
            print("   (Using pure Centroid Fusion of all 4 inputs)")
            
        f_t = fuser(embs, weights=weights)
        z_target = god_encoder(f_t)
        
        print("6. Autoregressive Decoding from z_target (The Topological Dream)...")
        bos_token = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (tokenizer.eos_token_id or 0)
        eos_token = tokenizer.eos_token_id
        
        # We call the newly implemented generate method on Decoder
        generated_ids = decoder.generate(z_target, start_token=bos_token, eos_token=eos_token, max_tokens=100, temperature=0.7)
        decoded_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        print(f"\n=================【终极拓扑解码对比】=================")
        print(f"[真实世界原句 (Original)]:")
        print(f" > {target_text}\n")
        print(f"[降维压缩后梦境重建 (Dream Reconstruction)]:")
        print(f" > {decoded_text}")
        print(f"======================================================\n")

if __name__ == "__main__":
    verify()
