"""
【脚本功能】：基于 Flow Matching (流匹配) 拓扑解码的端到端梦境重建器
【使用场景】：Phase 0 连续空间验证阶段。从 ModelScope 拉取真实 embedding 数据，
             通过 SensoryFuser → GodEncoder → FlowDecoder ODE 解码验证重建质量。
【用法示例】：
  python scripts/verify_flow0_from_disk_mlx.py --ckpt checkpoints/run/p0_flow_v0_step_160000 --num_samples 3
"""
import mlx.core as mx
import numpy as np
import pandas as pd
import os, sys, json, re

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from training.core.char_tokenizer import CharTokenizer
from model.config import ModelConfig
from model.adapter import SensoryFuser
from model.god_encoder import GodEncoder
from model.flow_decoder import FlowDecoder

import argparse
import random

def sniff_emb_dims_from_weights(ckpt_path: str):
    """从 checkpoint 权重反推每个 adapter 的输入维度"""
    weights = mx.load(f"{ckpt_path}/sense_fuser.safetensors")
    dims = {}
    for k, v in weights.items():
        # adapters.0.net.layers.0.weight -> shape (d_model, input_dim)
        match = re.match(r'adapters\.(\d+)\.net\.layers\.0\.weight', k)
        if match:
            adapter_idx = int(match.group(1))
            dims[adapter_idx] = v.shape[1]  # input_dim is second dim
    return [dims[i] for i in range(len(dims))]

def verify_flow():
    parser = argparse.ArgumentParser(description="Phase 0 Continuous Flow Matching Dream Verifier")
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--ode_steps", type=int, default=20, help="ODE Euler solver steps")
    parser.add_argument("--config_file", type=str, default="dataset_config.json")
    parser.add_argument("--data_dir", type=str, default="./embs", help="Local cache directory for ModelScope downloads")
    parser.add_argument("--chunk_start", type=int, default=None, help="Explicitly use a specific chunk_start instead of randomly picking")
    parser.add_argument("--offsets", type=str, default=None, help="Comma separated list of chunk offsets (e.g. '125570,240913')")
    args = parser.parse_args()

    # 0. 加载数据集配置
    with open(args.config_file, "r") as f:
        ds_config = json.load(f)
    base_models = ds_config.get("base_models", [])
    chunk_patterns = ds_config.get("chunk_name_patterns", {})
    ms_repo_id = ds_config.get("modelscope_repo_id", "catclever/emb_npy")
    parquet_path = ds_config.get("parquet_path", "data/Basic_ZH/chunked_mixed_omni.parquet")

    # 1. 从 checkpoint 权重反推模型架构
    print("1. Sniffing architecture from checkpoint weights...")
    emb_dims = sniff_emb_dims_from_weights(args.ckpt)
    print(f"   Detected emb_dims: {emb_dims} ({len(emb_dims)} models: {base_models})")
    
    config = ModelConfig()
    d_model = config.decoder_heads * 64
    
    fuser = SensoryFuser(emb_dims, d_model)
    god_encoder = GodEncoder(d_model, config.z_dim)
    decoder = FlowDecoder(config.z_dim, d_model, config.vocab_size, n_layers=config.decoder_layers)
    
    print(f"2. Loading weights from {args.ckpt}...")
    fuser_path = f"{args.ckpt}/sense_fuser.safetensors"
    if not os.path.exists(fuser_path):
        fuser_path = f"{args.ckpt}/sense_adapter.safetensors"
    fuser.load_weights(fuser_path)
    god_encoder.load_weights(f"{args.ckpt}/god_encoder.safetensors")
    decoder.load_weights(f"{args.ckpt}/flow_decoder.safetensors")
    mx.eval(fuser.parameters(), god_encoder.parameters(), decoder.parameters())
    
    # 3. 加载文本
    print(f"3. Loading text pool from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    text_chunks = df['chunks'].explode().dropna().tolist()
    total_chunks = len(text_chunks)
    del df
    print(f"   Total text chunks: {total_chunks}")
    
    # 4. 从 ModelScope 探测可用分块并拉取一个
    print("4. Probing ModelScope for a valid chunk...")
    from modelscope.hub.api import HubApi
    from modelscope.hub.file_download import dataset_file_download
    
    api = HubApi()
    res = api.get_dataset_files(ms_repo_id, recursive=True)
    chunk_files = [f.get("Path", f.get("Name", "")) for f in res]
    
    # 按模型归类 start → end
    per_model_bounds = {}
    for model_name in base_models:
        pattern = chunk_patterns.get(model_name, "")
        prefix = pattern.split("/")[0] if "/" in pattern else model_name
        start_to_end = {}
        for f in chunk_files:
            if f.endswith(".npz") and f.startswith(prefix + "/"):
                match = re.search(r'chunk_(\d+)_(\d+)\.npz', os.path.basename(f))
                if match:
                    start_to_end[int(match.group(1))] = int(match.group(2))
        per_model_bounds[model_name] = start_to_end
    
    common_starts = sorted(set.intersection(*[set(d.keys()) for d in per_model_bounds.values()]))
    print(f"   Found {len(common_starts)} common chunks")
    
    # 随机或指定选一个块
    if args.chunk_start is not None:
        chosen_start = args.chunk_start
        min_end = min(per_model_bounds[m][chosen_start] for m in base_models)
        chunk_size = min_end - chosen_start
    else:
        chosen_start = random.choice(common_starts)
        min_end = min(per_model_bounds[m][chosen_start] for m in base_models)
        chunk_size = min_end - chosen_start
    print(f"   Using chunk [{chosen_start}:{min_end}] (size: {chunk_size})")
    
    # 下载该块所有模型的数据
    chunk_embs = []
    for model_name in base_models:
        pattern = chunk_patterns.get(model_name, "")
        model_end = per_model_bounds[model_name][chosen_start]
        cand_name = pattern.format(start=chosen_start, end=model_end)
        print(f"   Downloading {cand_name}...")
        path = dataset_file_download(ms_repo_id, cand_name, cache_dir=args.data_dir)
        arr = np.load(path)['features']
        chunk_embs.append(arr)
    
    # 5. 随机采样验证或指定 offsets
    tokenizer = CharTokenizer()
    if args.offsets is not None:
        sampled_offsets = [int(o) for o in args.offsets.split(",")]
    else:
        sampled_offsets = random.sample(range(chunk_size), min(args.num_samples, chunk_size))
    
    for i, offset in enumerate(sampled_offsets):
        global_idx = chosen_start + offset
        target_text = text_chunks[global_idx]
        
        print(f"\n{'='*60}")
        print(f"       >>> SAMPLE {i+1} (Global Row #{global_idx}) <<<")
        print(f"{'='*60}")
        
        # 取出该行的 embedding (1, dim)
        embs = [mx.array(arr[offset:offset+1]) for arr in chunk_embs]
        
        print("5. SensoryFuser → GodEncoder → z_target...")
        f_t = fuser(embs, weights=None)  # Centroid fusion
        z_target = god_encoder(f_t)
        mx.eval(z_target)
        
        print("6. Continuous ODE Decoding (Euler solver)...")
        tokenized_ids = tokenizer.encode(target_text, add_special_tokens=True)
        cheat_length = len(tokenized_ids)
        print(f"   [Cheat Mode: target length = {cheat_length}]")
        
        generated_ids = decoder.generate_euler(z_target, target_length=cheat_length, steps=args.ode_steps)
        mx.eval(generated_ids)
        
        generated_list = generated_ids[0].tolist()
        decoded_text = tokenizer.decode(generated_list, skip_special_tokens=True)
        
        # 计算字符级准确率
        min_len = min(len(target_text), len(decoded_text))
        correct = sum(1 for a, b in zip(target_text[:min_len], decoded_text[:min_len]) if a == b)
        accuracy = correct / max(len(target_text), 1) * 100
        
        print(f"\n{'='*20}【连续流体力学解码对比】{'='*20}")
        print(f"[原句 Original]:  {target_text[:200]}{'...' if len(target_text) > 200 else ''}")
        print(f"[ODE 重建 Dream]: {decoded_text[:200]}{'...' if len(decoded_text) > 200 else ''}")
        print(f"[字符准确率]:      {accuracy:.1f}%")
        print(f"{'='*60}\n")

if __name__ == "__main__":
    verify_flow()
