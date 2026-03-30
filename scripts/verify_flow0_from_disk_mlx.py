"""
【脚本功能】：基于 Flow Matching (流匹配) 拓扑解码的端到端梦境重建器
【使用场景】：Phase 0 连续空间验证阶段。模拟验证大语言模型的文字序列被 GodEncoder 深度降维压榨后，能否通过 ODE 解码器（Continuous Flow）顺滑地解析出它应有的自然语言形态。
【用法示例】：`python scripts/verify_flow0_from_disk.py --ckpt checkpoints/run/p0_flow_v1_step_50000 --num_samples 3`
"""
import mlx.core as mx
import numpy as np
import pandas as pd
import os

from training.char_tokenizer import CharTokenizer
from model.config import ModelConfig
from model.adapter import SensoryFuser
from model.god_encoder import GodEncoder
from model.flow_decoder import FlowDecoder

import argparse
import random

def verify_flow():
    parser = argparse.ArgumentParser(description="Phase 0 Continuous Flow Matching Dream Verifier")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of random sentences to pull and dream-reconstruct")
    parser.add_argument("--emb_idx", type=int, default=-1, help="-1: fully fused. 0:roberta, 1:gte, 2:bge, 3:text2vec")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint directory (e.g. checkpoints/run/p0_flow_v1_step_50000)")
    args = parser.parse_args()

    print("1. Loading physical architecture...")
    tokenizer = CharTokenizer()
    
    config = ModelConfig()
    d_model = config.decoder_heads * 64
    
    fuser = SensoryFuser(config.emb_dims, d_model)
    god_encoder = GodEncoder(d_model, config.z_dim)
    # Using the new FlowDecoder
    decoder = FlowDecoder(config.z_dim, d_model, config.vocab_size, n_layers=config.decoder_layers)
    
    ckpt_path = args.ckpt
    print(f"2. Loading weights from {ckpt_path}...")
    
    fuser_path = f"{ckpt_path}/sense_fuser.safetensors"
    if not os.path.exists(fuser_path):
        fuser_path = f"{ckpt_path}/sense_adapter.safetensors"
    fuser.load_weights(fuser_path)
    god_encoder.load_weights(f"{ckpt_path}/god_encoder.safetensors")
    decoder.load_weights(f"{ckpt_path}/flow_decoder.safetensors")
    
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
        
        print("6. Continuous ODE Decoding from z_target (The Topological Dream)...")
        # --- THE CHEAT ---
        # We extract the actual token sequence length to bypass padding and dynamic length prediction.
        tokenized_ids = tokenizer.encode(target_text)
        cheat_length = len(tokenized_ids)
        print(f"   [Cheat Mode: Feeding exact sentence length ({cheat_length}) to ODE Solver]")
        
        # Call the continuous Euler solver!
        generated_ids = decoder.generate_euler(z_target, target_length=cheat_length, steps=20)
        
        # Convert generated_ids to python list and decode
        generated_list = generated_ids[0].tolist()
        decoded_text = tokenizer.decode(generated_list, skip_special_tokens=True)
        
        print(f"\n=================【终极连续流体力学解码对比】=================")
        print(f"[真实世界原句 (Original)]:")
        print(f" > {target_text}\n")
        print(f"[高斯爆震生成梦境 (Euler Dream Reconstruction)]:")
        print(f" > {decoded_text}")
        print(f"==============================================================\n")

if __name__ == "__main__":
    verify_flow()
