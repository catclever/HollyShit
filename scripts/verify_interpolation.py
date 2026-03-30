"""
【脚本功能】：高维空间球面平滑线性插值 (SLERP) 的概念漫游器
【使用场景】：Phase 0 验证阶段。探究大语言模型的离散文字空间在被降维锚定后，是否形成了连续的拓扑结构。从 A 概念到 B 概念的每半步迁移，所解码出的句子是否过渡依然物理丝滑且合理。
【用法示例】：`python scripts/verify_interpolation.py --ckpt checkpoints/run/p0_v1_step_150000 --idx_a 1000 --idx_b 2000000 --steps 5`
"""
import mlx.core as mx
import numpy as np
import pandas as pd
import os
import argparse

from training.char_tokenizer import CharTokenizer
from model.config import ModelConfig
from model.adapter import SensoryFuser
from model.god_encoder import GodEncoder
from model.decoder import WeakDecoder

def slerp(val, low, high):
    """
    Spherical Linear Interpolation (SLERP) for high-dimensional vectors.
    Produces a much smoother transition through the semantic space than straight LERP.
    """
    # Normalize vectors
    low_norm = np.linalg.norm(low)
    high_norm = np.linalg.norm(high)
    
    # Avoid div/0
    low_norm = low_norm if low_norm > 0 else 1.0
    high_norm = high_norm if high_norm > 0 else 1.0
    
    omega = np.arccos(np.dot(low/low_norm, high/high_norm))
    if np.isnan(omega) or omega == 0:
        return (1 - val) * low + val * high
        
    so = np.sin(omega)
    if so == 0:
        return (1 - val) * low + val * high
        
    return np.sin((1.0 - val) * omega) / so * low + np.sin(val * omega) / so * high

def verify_smoothness():
    parser = argparse.ArgumentParser(description="GodEncoder Latent Space Interpolation")
    parser.add_argument("--idx_a", type=int, default=1000, help="Row index for Sentence A")
    parser.add_argument("--idx_b", type=int, default=2000000, help="Row index for Sentence B")
    parser.add_argument("--steps", type=int, default=5, help="Number of interpolation steps between A and B")
    parser.add_argument("--emb_indices", type=int, nargs="+", default=[-1], help="-1: fully fused. Else, a space-separated list of indices 0-3 (e.g., 0 2)")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint directory")
    args = parser.parse_args()

    print("1. Loading architecture...")
    tokenizer = CharTokenizer()
    config = ModelConfig()
    d_model = config.decoder_heads * 64
    
    fuser = SensoryFuser(config.emb_dims, d_model)
    god_encoder = GodEncoder(d_model, config.z_dim)
    decoder = WeakDecoder(config.z_dim, config.vocab_size, d_model=d_model, n_layers=config.decoder_layers)
    
    ckpt_path = args.ckpt
    print(f"2. Loading weights from {ckpt_path}...")
    
    fuser_path = f"{ckpt_path}/sense_fuser.safetensors"
    if not os.path.exists(fuser_path):
        fuser_path = f"{ckpt_path}/sense_adapter.safetensors"
    fuser.load_weights(fuser_path)
    god_encoder.load_weights(f"{ckpt_path}/god_encoder.safetensors")
    decoder.load_weights(f"{ckpt_path}/decoder.safetensors")
    
    print("3. Mmapping source texts and arrays...")
    df = pd.read_parquet("data/Basic_ZH/chunked_mixed_wiki.parquet")
    text_chunks = df['chunks'].explode().dropna().tolist()
    
    emb_files = [
        "data/Basic_ZH/embs/hy-tmp/roberta_embeddings.npy",
        "data/Basic_ZH/embs/hy-tmp/gte_embeddings.npy",
        "data/Basic_ZH/embs/hy-tmp/bge_embeddings.npy",
        "data/Basic_ZH/embs/hy-tmp/text2vec_embeddings.npy"
    ]
    embs_mmap = [np.load(path, mmap_mode='r') for path in emb_files]
    
    # ----------------------------------------------------
    # EXTRACT Z_A
    text_a = text_chunks[args.idx_a]
    embs_a = [mx.array(arr[args.idx_a:args.idx_a+1]) for arr in embs_mmap]
    
    if -1 in args.emb_indices:
        weights = None  # Centroid of all 4
    else:
        weights = [0.0] * 4
        for idx in args.emb_indices:
            weights[idx] = 1.0 / len(args.emb_indices)

    f_a = fuser(embs_a, weights=weights)
    z_a = god_encoder(f_a) # Shape: [1, z_dim]
    
    # EXTRACT Z_B
    text_b = text_chunks[args.idx_b]
    embs_b = [mx.array(arr[args.idx_b:args.idx_b+1]) for arr in embs_mmap]
    f_b = fuser(embs_b, weights=weights)
    z_b = god_encoder(f_b) # Shape: [1, z_dim]
    # ----------------------------------------------------

    print(f"\n======================================")
    print(f" [A] 始发站 ({args.idx_a}): {text_a}")
    print(f" [B] 终点站 ({args.idx_b}): {text_b}")
    print(f"======================================\n")
    
    z_a_np = np.array(z_a[0].tolist())
    z_b_np = np.array(z_b[0].tolist())
    
    # ----------------------------------------------------
    # Traverse the Latent Space Space
    print("🚀 启动高维空间球面平滑过渡 (Spherical Interpolation)... \n")
    
    start_token = tokenizer.bos_token_id
    eos_token = tokenizer.eos_token_id
    
    for i in range(args.steps + 1):
        alpha = float(i) / args.steps
        
        # 1. 混合向量 (SLERP)
        z_mixed_np = slerp(alpha, z_a_np, z_b_np)
        z_mixed = mx.array(z_mixed_np).reshape(1, config.z_dim)
        
        # 2. 降维解码
        generated_ids = decoder.generate(z_mixed, start_token=start_token, eos_token=eos_token, max_tokens=100, temperature=0.2)
        decoded_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # 3. 打印路标
        print(f" [α = {alpha:.2f}] ({'░'*i}{'▒'}{'░'*(args.steps-i)}) : {decoded_text}")
        
    print(f"\n======================================================\n")

if __name__ == "__main__":
    verify_smoothness()
