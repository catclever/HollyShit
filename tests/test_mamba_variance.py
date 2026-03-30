"""
【脚本功能】：Mamba 预测方差 (LogVar) 物理塌缩侦测器
【使用场景】：核心组件测试。我们设计 Mamba 不仅要预测坐标 `mu`，还要预测方差 `logvar`。本脚本测试在网络训练初期，方差是否能保持稳定，而不至于引发负无穷或 NaN 的算力塌缩（Posterior Collapse）。
【用法示例】：`python tests/test_mamba_variance.py`
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import mlx.core as mx

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.config import ModelConfig
from model.adapter import SensoryFuser
from model.mamba_mlx.mamba_mlx import MambaConfig
from model.mamba_planner import MambaPlanner
from training.core.dataloader import Phase1DataLoader

def load_models_for_variance(p0_ckpt, p1_ckpt):
    config = ModelConfig()
    d_model = config.decoder_heads * 64
    
    print(f"Loading Adapter/Fuser from Phase 0: {p0_ckpt}")
    fuser = SensoryFuser(config.emb_dims, d_model)
    fuser_path = f"{p0_ckpt}/sense_fuser.safetensors"
    if not os.path.exists(fuser_path):
        fuser_path = f"{p0_ckpt}/sense_adapter.safetensors"
    fuser.load_weights(fuser_path)
    
    print(f"Loading Mamba Planner from Phase 1: {p1_ckpt}")
    mamba_cfg = MambaConfig(d_model=d_model, n_layers=2)
    mamba = MambaPlanner(mamba_cfg, config.z_dim, residual_mode=False) 
    mamba.load_weights(f"{p1_ckpt}/mamba_planner.safetensors")
    
    fuser.eval()
    mamba.eval()
    return fuser, mamba

def run_variance_test():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p0_ckpt", required=True, help="Frozen Phase 0 weights")
    parser.add_argument("--p1_ckpt", required=True, help="Trained Phase 1 weights")
    args = parser.parse_args()
    
    fuser, mamba = load_models_for_variance(args.p0_ckpt, args.p1_ckpt)
    
    # We will manually load the first document from Parquet to print the actual Chinese context
    parquet_path = "data/Basic_ZH/chunked_mixed_wiki.parquet"
    print(f"\nScanning human-readable text from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    
    doc_index = 0
    text_chunks = df.iloc[doc_index]['chunks'] # list of strings
    
    emb_files = [
        "data/Basic_ZH/embs/hy-tmp/roberta_embeddings.npy",
        "data/Basic_ZH/embs/hy-tmp/gte_embeddings.npy",
        "data/Basic_ZH/embs/hy-tmp/bge_embeddings.npy",
        "data/Basic_ZH/embs/hy-tmp/text2vec_embeddings.npy"
    ]
    
    # We use Phase1DataLoader to seamlessly fetch just the exact same 1 aligned batch
    dataloader = Phase1DataLoader(parquet_path, emb_files, batch_size=1, max_episode_len=len(text_chunks))
    batch_embs, masks = next(iter(dataloader))
    
    print(f"Document Length: {len(text_chunks)} sentences.")
    print("Feeding Semantic Flow into Mamba to capture Aleatoric Uncertainty...\n")
    
    # 1. Fuse offline vectors
    f_t = fuser(batch_embs, weights=None)
    
    # 2. Extract trajectory variance
    _, logvar, _ = mamba(f_t)
    var = mx.exp(logvar) # (1, L, z_dim)
    
    # 3. Compute structural uncertainty (Sum of variance across all spatial dimensions)
    total_variance_per_step = var[0].sum(axis=-1).tolist() # Extract the 1st batch item
    
    print("==================================================")
    print("ALEATORIC UNCERTAINTY MAP (MAMBA CONFIDENCE CHART)")
    print("==================================================")
    print("Longer Bar = High Uncertainty (Large Variance / Throwing a wide net)")
    print("Shorter Bar = Extreme Confidence (Focused Sniper Coordinate)\n")
    
    max_var = max(total_variance_per_step)
    
    for t in range(len(text_chunks)):
        chunk_text = text_chunks[t][:30].replace('\n', ' ')
        if len(text_chunks[t]) > 30:
            chunk_text += "..."
            
        current_var = total_variance_per_step[t]
        
        # Scale to 40 max visual blocks
        bar_len = int((current_var / max_var) * 40)
        bar = "█" * bar_len
        
        print(f"T={t:02d} | Var: {current_var:7.2f} | {bar:<40} | Text: {chunk_text}")
        
    print("\n✅ Plot rendering complete! Analyze how reading context dynamically alters Mamba's statistical net.")

if __name__ == "__main__":
    run_variance_test()
