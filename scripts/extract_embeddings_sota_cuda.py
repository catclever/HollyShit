"""
【脚本功能】：高维连续特征 (GodEncoder Anchor) 的零拷贝实体化榨汁机
【使用场景】：Phase 0.1 云端高算力压榨阶段。运行于云端带有至少 24GB 显存的显卡上。负责动用 BFloat16/FlashAttention2 加速模型，将百万级自然语言文本提炼成极致浓缩的纯数学高维特征，并以 `.npy` memmap 直接刷入硬盘防止 OOM。
【用法示例】：
    - 榨取 BGE 对比特征：`python scripts/extract_embeddings_sota_cuda.py --model bge --batch_size 256`
    - 榨取 Qwen 因果特征：`python scripts/extract_embeddings_sota_cuda.py --model qwen --batch_size 16`
"""

import sys
import os
import argparse
import importlib.util
import json
from typing import TypedDict
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

class StringDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]

def get_last_token_pooling(last_hidden_states, attention_mask):
    """Pools the last non-padding token for causal LLMs (like Qwen)."""
    # attention_mask is 1 for real tokens, 0 for padding.
    # The last real token is at index: attention_mask.sum(dim=1) - 1
    seq_lengths = attention_mask.sum(dim=1) - 1
    # Gather the specific token for each item in the batch
    batch_size = last_hidden_states.size(0)
    return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), seq_lengths, :]

def find_first_zero_row(arr):
    lo, hi = 0, arr.shape[0]
    while lo < hi:
        mid = (lo + hi) // 2
        if np.all(arr[mid] == 0):
            hi = mid
        else:
            lo = mid + 1
    return lo

def save_progress(progress_path, payload):
    tmp_path = progress_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    os.replace(tmp_path, progress_path)

def ensure_losskwargs_compat():
    import transformers.utils as tf_utils
    if hasattr(tf_utils, "LossKwargs"):
        return
    class LossKwargs(TypedDict, total=False):
        pass
    tf_utils.LossKwargs = LossKwargs

def ensure_rope_default_compat():
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    if "default" in ROPE_INIT_FUNCTIONS:
        return
    def _compute_default_rope_parameters(config, device=None, seq_len=None):
        dim = int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads))
        base = float(getattr(config, "rope_theta", 10000.0))
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
        attention_factor = 1.0
        return inv_freq, attention_factor
    ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters

def ensure_pruned_heads_compat():
    from transformers.configuration_utils import PretrainedConfig
    if not hasattr(PretrainedConfig, "pruned_heads"):
        PretrainedConfig.pruned_heads = {}

def validate_local_weight_shards(model_dir):
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if not os.path.isfile(index_path):
        return
    with open(index_path, "r", encoding="utf-8") as f:
        index_json = json.load(f)
    shard_files = sorted(set(index_json.get("weight_map", {}).values()))
    missing = [name for name in shard_files if not os.path.isfile(os.path.join(model_dir, name))]
    if missing:
        raise SystemExit(f"本地模型分片不完整，缺失 {len(missing)} 个文件（示例: {missing[:3]}）。请先完整下载模型后再运行。")

class ActiveChunkManager:
    def __init__(self, out_dir, base_name, chunk_size, hidden_dim, repo_id=None, token=None, repo_type="auto", repo_subdir="", delete_chunks=False):
        self.chunk_size = chunk_size
        self.hidden_dim = hidden_dim
        self.out_dir = out_dir
        self.base_name = base_name.replace('.npy', '')
        self.delete_chunks = delete_chunks
        
        self.current_chunk_idx = -1
        self.buffer_start_global = -1
        self.buffer = None
        self.uploader = None
        self.repo_subdir = repo_subdir
        
        if repo_id:
            sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
            try:
                from tools.chunk_and_upload import ChunkUploader
                self.uploader = ChunkUploader(ms_repo_id=repo_id, ms_token=token, repo_type=repo_type)
            except Exception as e:
                print(f"[FATAL] 无法初始化 ChunkUploader: {e}")
                exit(1)
                
    def get_chunk_file_path(self, chunk_idx):
        start = chunk_idx * self.chunk_size
        end = start + self.chunk_size
        return os.path.join(self.out_dir, f"{self.base_name}_chunk_{start:07d}_{end:07d}.npz")

    def flush_chunk(self, final_cut=None):
        if self.buffer is None or self.current_chunk_idx < 0:
            return
        chunk_path = self.get_chunk_file_path(self.current_chunk_idx)
        print(f"\n[ChunkManager] 触发落盘切块: {chunk_path}")
        
        # 如果提供了 final_cut，代表是最后一个不足 chunk_size 的残块，精准切除尾声全是 0 的预留内存
        data_to_save = self.buffer[:final_cut] if final_cut is not None else self.buffer
        np.savez_compressed(chunk_path, features=data_to_save)
        
        if self.uploader:
            print(f"[ChunkManager] 自动推送至 ModelScope...")
            success = self.uploader.push(chunk_path, delete_source=self.delete_chunks, repo_subdir=self.repo_subdir)
            if not success:
               raise RuntimeError(f"Chunk 上传失败，立刻停止以避免数据断层丢失: {chunk_path}")
        self.buffer = None

    def write(self, global_start, data):
        N = data.shape[0]
        written = 0
        while written < N:
            global_curr = global_start + written
            target_chunk_idx = global_curr // self.chunk_size
            
            if target_chunk_idx != self.current_chunk_idx:
                self.flush_chunk()
                self.current_chunk_idx = target_chunk_idx
                self.buffer_start_global = target_chunk_idx * self.chunk_size
                self.buffer = np.zeros((self.chunk_size, self.hidden_dim), dtype=np.float16)
            
            local_start = global_curr - self.buffer_start_global
            space_in_chunk = self.chunk_size - local_start
            write_len = min(space_in_chunk, N - written)
            
            self.buffer[local_start : local_start + write_len] = data[written : written + write_len]
            written += write_len

def main():
    parser = argparse.ArgumentParser(description="GodEncoder Feature Extractor (OOM-Safe Memmap)")
    parser.add_argument("--model", type=str, required=True, choices=["bge", "qwen", "xiaobu", "youtu", "conan_v1", "conan"], help="Which architecture to extract.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference. Small for Qwen, Large for BGE.")
    parser.add_argument("--max_seq_len", type=int, default=1024, help="Context length truncation.")
    parser.add_argument("--max_items", type=int, default=0, help="Limit number of text chunks to process. 0 means all.")
    parser.add_argument("--parquet_path", type=str, default="./datas.parquet")
    parser.add_argument("--cache_dir", type=str, default="./models", help="Model weights cache directory")
    parser.add_argument("--out_dir", type=str, default="/hy-tmp/embs/", help="Output directory for massive .npy matrices")
    parser.add_argument("--resume", "--continue", dest="resume", action="store_true", help="Resume from existing output .npy if possible")
    parser.add_argument("--resume_safety_rows", type=int, default=8, help="Recompute a few rows before detected resume point")
    
    # --- 新增的流式切块参数 ---
    parser.add_argument("--chunk_size", type=int, default=0, help="如果 >0，开启分块截断模式，摒弃巨星单文件")
    parser.add_argument("--ms_repo_id", type=str, default=None, help="魔搭目标仓库 (如 kael/phase0-embeddings)，填了即启动自动上传")
    parser.add_argument("--ms_token", type=str, default=os.environ.get("MODELSCOPE_API_TOKEN", ""), help="默认读取环境 MODELSCOPE_API_TOKEN")
    parser.add_argument("--ms_repo_type", type=str, choices=["auto", "model", "dataset"], default="auto", help="ModelScope 仓库类型")
    parser.add_argument("--ms_repo_subdir", type=str, default="", help="上传到仓库中的子目录，如 embs/qwen")
    parser.add_argument("--delete_chunks", action="store_true", help="【零空间开销】传完立刻抹除本地块")
    
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resume_safety_rows = max(max(0, args.resume_safety_rows), max(1, args.batch_size))

    model_key = "conan_v1" if args.model == "conan" else args.model

    # 1. Setup Model Specific Configurations
    if model_key == "bge":
        model_id = "BAAI/bge-m3"
        hidden_dim = 1024
        dtype = torch.float16
        use_flash_attn = False
        out_name = "bge_embeddings.npy"
        engine = "auto"
    elif model_key == "qwen":
        model_id = "iic/gte_Qwen2-7B-instruct"
        hidden_dim = 3584
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        use_flash_attn = True
        out_name = "gte_qwen2_embeddings.npy"
        engine = "auto"
    elif model_key == "xiaobu":
        model_id = "lier007/xiaobu-embedding-v2"
        hidden_dim = None
        dtype = torch.float16
        use_flash_attn = False
        out_name = "xiaobu_embeddings.npy"
        engine = "st"
    elif model_key == "youtu":
        model_id = "tencent/Youtu-Embedding"
        hidden_dim = None
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
        use_flash_attn = False
        out_name = "youtu_embeddings.npy"
        engine = "st"
    elif model_key == "conan_v1":
        model_id = "TencentBAC/Conan-embedding-v1"
        hidden_dim = None
        dtype = torch.float16
        use_flash_attn = False
        out_name = "conan_v1_embeddings.npy"
        engine = "st"
    else:
        raise RuntimeError("Unsupported model type")

    out_path = os.path.join(args.out_dir, out_name)
    progress_path = out_path + ".progress.json"
    model_local_path = os.path.join(args.cache_dir, model_id)
    if model_key in {"xiaobu", "youtu", "conan_v1"} and not os.path.isdir(model_local_path):
        raise SystemExit(f"本地模型目录不存在: {model_local_path}。请先下载到本地后再跑。")
    if model_key == "youtu":
        validate_local_weight_shards(model_local_path)
    model_source = model_local_path if os.path.isdir(model_local_path) else model_id

    print(f"=== SOTA GodEncoder Pipeline: {model_id} ===")
    print(f"Loading Text Targets from: {args.parquet_path}")

    # 2. Extract and Flatten all Chunks
    df = pd.read_parquet(args.parquet_path, columns=['chunks'])
    all_chunks = df['chunks'].explode().dropna().tolist()
    if args.max_items > 0:
        all_chunks = all_chunks[:args.max_items]
    total_elements = len(all_chunks)
    print(f"Exploded into {total_elements} physical chunks.")
    del df # Free RAM immediately

    tokenizer = None
    if engine == "st":
        st_kwargs = {"trust_remote_code": True}
        if str(device).startswith("cuda"):
            st_kwargs["model_kwargs"] = {"torch_dtype": dtype}
        ensure_losskwargs_compat()
        ensure_rope_default_compat()
        ensure_pruned_heads_compat()
        try:
            model = SentenceTransformer(model_source, device=str(device), **st_kwargs)
        except ImportError as e:
            if "LossKwargs" not in str(e):
                raise
            ensure_losskwargs_compat()
            ensure_rope_default_compat()
            ensure_pruned_heads_compat()
            model = SentenceTransformer(model_source, device=str(device), **st_kwargs)
        hidden_dim = model.get_sentence_embedding_dimension()
    else:
        print(f"Loading Tokenizer from {model_source}...")
        tokenizer = AutoTokenizer.from_pretrained(model_source, cache_dir=args.cache_dir, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print(f"Booting Weights into VRAM ({dtype})...")
        model_kwargs = {
            "cache_dir": args.cache_dir,
            "torch_dtype": dtype,
            "trust_remote_code": True,
            "device_map": "auto"
        }

        flash_attn_available = importlib.util.find_spec("flash_attn") is not None
        if use_flash_attn and torch.cuda.is_bf16_supported() and flash_attn_available:
            print("Engaging FlashAttention-2 High-Speed Kernel...")
            model_kwargs["attn_implementation"] = "flash_attention_2"

        model = AutoModel.from_pretrained(model_source, **model_kwargs)
        model.eval()

    # 3. Create OOM-Proof Numpy Memmap (Direct-to-Disk array) or Chunk Manager
    start_idx = 0
    mmap_arr = None
    chunk_mgr = None
    
    # 高级断点重算：先基于 JSON 判断大概进度
    if args.resume and os.path.exists(progress_path):
        try:
            with open(progress_path, "r", encoding="utf-8") as f:
                progress_rows = int(json.load(f).get("written_rows", 0))
                start_idx = max(0, progress_rows - resume_safety_rows)
                print(f"[RESUME] 成功读取 JSON 进度，游标从安全记录点 {start_idx} 恢复")
        except Exception:
            pass

    if args.chunk_size > 0:
        print(f"=============================")
        print(f"🚀 已启用【自动切片流水线】！Chunk Size: {args.chunk_size}")
        if args.ms_repo_id:
            print(f"📡 【边打边传模式】已激活！目标挂载库：{args.ms_repo_id}")
        if args.delete_chunks:
            print(f"⚔️ 【零本地存储模式】极速挥发，空间占用趋于0")
        print(f"=============================")
        chunk_mgr = ActiveChunkManager(
            args.out_dir, out_name, args.chunk_size, hidden_dim, 
            repo_id=args.ms_repo_id, token=args.ms_token, repo_type=args.ms_repo_type, repo_subdir=args.ms_repo_subdir, delete_chunks=args.delete_chunks
        )
    else:
        # 传统单体巨兽的 Memmap 强行容错恢复
        if args.resume and os.path.exists(out_path):
            mmap_arr = np.load(out_path, mmap_mode="r+")
            if mmap_arr.shape != (total_elements, hidden_dim):
                raise RuntimeError(f"Existing output shape {mmap_arr.shape} mismatches {total_elements} x {hidden_dim}")
            
            # 物理验证覆盖 json
            first_zero = find_first_zero_row(mmap_arr)
            start_idx = min(start_idx, max(0, first_zero - resume_safety_rows)) if start_idx > 0 else first_zero
            print(f"Resuming existing memmap at {out_path} row: {start_idx}")
        else:
            print(f"Allocating {total_elements} x {hidden_dim} Memmap at {out_path}... Prepare your local SSD!")
            mmap_arr = np.lib.format.open_memmap(out_path, mode='w+', shape=(total_elements, hidden_dim), dtype=np.float16)

    # Boot Progress json info immediately
    if start_idx == 0:
        save_progress(progress_path, {
            "model": args.model,
            "out_path": out_path,
            "written_rows": 0
        })

    # 4. Dataloader execution
    dataset = StringDataset(all_chunks[start_idx:])
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    print(f"\nCommencing High-Throughput CUDA Extraction! Batch Size: {args.batch_size}")
    
    with torch.no_grad():
        for batch_texts in tqdm(dataloader, desc=f"Extracting {args.model.upper()}"):
            if engine == "auto":
                inputs = tokenizer(
                    list(batch_texts), 
                    padding=True, 
                    truncation=True, 
                    max_length=args.max_seq_len, 
                    return_tensors="pt"
                ).to(device)
                outputs = model(**inputs)
                last_hidden = outputs.last_hidden_state
                if model_key == "bge":
                    embs = last_hidden[:, 0, :]
                    embs = torch.nn.functional.normalize(embs, p=2, dim=1)
                elif model_key == "qwen":
                    embs = get_last_token_pooling(last_hidden, inputs.attention_mask)
                    embs = torch.nn.functional.normalize(embs, p=2, dim=1)
                else:
                    raise RuntimeError("Unsupported model type")
                numpy_embs = embs.cpu().to(torch.float16).numpy()
            else:
                numpy_embs = model.encode(
                    list(batch_texts),
                    batch_size=args.batch_size,
                    convert_to_numpy=True,
                    normalize_embeddings=False,
                    show_progress_bar=False
                ).astype(np.float16)
            if not np.isfinite(numpy_embs).all():
                bad_ratio = float((~np.isfinite(numpy_embs)).mean())
                raise RuntimeError(f"检测到非有限 embedding 值（NaN/Inf），bad_ratio={bad_ratio:.6f}，当前写入起点={start_idx}。建议优先使用 youtu 的 bfloat16/float32 推理。")

            batch_len = numpy_embs.shape[0]
            end_idx = start_idx + batch_len

            if chunk_mgr:
                chunk_mgr.write(start_idx, numpy_embs)
            else:
                mmap_arr[start_idx:end_idx] = numpy_embs

            start_idx = end_idx
            
            # Periodically flush every 10,000 items to guarantee zero data loss
            if start_idx % 10000 < args.batch_size:
                if mmap_arr is not None:
                    mmap_arr.flush()
                save_progress(progress_path, {
                    "model": args.model,
                    "model_id": model_id,
                    "out_path": out_path,
                    "shape": [total_elements, hidden_dim],
                    "dtype": "float16",
                    "written_rows": int(start_idx)
                })
                
    # Final flush
    if chunk_mgr:
        # 如果最后一个有效快塞不满导致没被保存，计算余数强制切出上传
        final_valid_rows = total_elements % args.chunk_size
        if final_valid_rows == 0 and total_elements > 0:
            final_valid_rows = args.chunk_size  # 刚刚好填满
        chunk_mgr.flush_chunk(final_cut=final_valid_rows)
    else:
        mmap_arr.flush()
        
    save_progress(progress_path, {
        "model": args.model,
        "model_id": model_id,
        "out_path": out_path,
        "shape": [total_elements, hidden_dim],
        "dtype": "float16",
        "written_rows": int(start_idx),
        "completed": bool(start_idx == total_elements)
    })
    print(f"\n[SUCCESS] Extracted Matrix fully solidified at: {out_path}")
    print(f"Matrix Shape: ({total_elements}, {hidden_dim})")

if __name__ == "__main__":
    main()
