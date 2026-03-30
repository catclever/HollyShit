"""
【脚本功能】：高维连续特征 (GodEncoder Anchor) 的零拷贝实体化榨汁机
【使用场景】：Phase 0.1 云端高算力压榨阶段。运行于云端带有至少 24GB 显存的显卡上。负责动用 BFloat16/FlashAttention2 加速模型，将百万级自然语言文本提炼成极致浓缩的纯数学高维特征，并以 `.npy` memmap 直接刷入硬盘防止 OOM。
【用法示例】：
    - 榨取 BGE 对比特征：`python scripts/extract_embeddings_sota_cuda.py --model bge --batch_size 256`
    - 榨取 Qwen 因果特征：`python scripts/extract_embeddings_sota_cuda.py --model qwen --batch_size 16`
"""
import os
import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel

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

def main():
    parser = argparse.ArgumentParser(description="GodEncoder Feature Extractor (OOM-Safe Memmap)")
    parser.add_argument("--model", type=str, required=True, choices=["bge", "qwen"], help="Which architecture to extract.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference. Small for Qwen, Large for BGE.")
    parser.add_argument("--max_seq_len", type=int, default=1024, help="Context length truncation.")
    parser.add_argument("--parquet_path", type=str, default="data/Basic_ZH/chunked_mixed_omni.parquet")
    parser.add_argument("--cache_dir", type=str, default="./models", help="Model weights cache directory")
    parser.add_argument("--out_dir", type=str, default="data/Basic_ZH/embs", help="Output directory for massive .npy matrices")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Setup Model Specific Configurations
    if args.model == "bge":
        model_id = "BAAI/bge-m3"
        hidden_dim = 1024
        dtype = torch.float16
        use_flash_attn = False
        out_name = "bge_embeddings.npy"
    else:
        # qwen
        model_id = "iic/gte_Qwen2-7B-instruct"
        # Can fallback to Alibaba-NLP/gte-Qwen2-7B-instruct if needed
        hidden_dim = 3584
        dtype = torch.bfloat16
        use_flash_attn = True # Requires Ampere/Ada cards (3090, 4090, A100)
        out_name = "gte_qwen2_embeddings.npy"

    out_path = os.path.join(args.out_dir, out_name)

    print(f"=== SOTA GodEncoder Pipeline: {model_id} ===")
    print(f"Loading Text Targets from: {args.parquet_path}")

    # 2. Extract and Flatten all Chunks
    df = pd.read_parquet(args.parquet_path, columns=['chunks'])
    all_chunks = df['chunks'].explode().dropna().tolist()
    total_elements = len(all_chunks)
    print(f"Exploded into {total_elements} physical chunks.")
    del df # Free RAM immediately

    # 3. Create OOM-Proof Numpy Memmap (Direct-to-Disk array)
    print(f"Allocating {total_elements} x {hidden_dim} Memmap at {out_path}...")
    # Mode 'w+' creates or overwrites the file. Data type is float16 to save space (half the 50GB size).
    mmap_arr = np.lib.format.open_memmap(out_path, mode='w+', shape=(total_elements, hidden_dim), dtype=np.float16)

    # 4. Initialize Transformers
    print(f"Loading Tokenizer from {args.cache_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=args.cache_dir, trust_remote_code=True)
    
    # Qwen/Llama tokenizers often lack a pad token by default, assign EOS if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Booting Weights into VRAM ({dtype})...")
    model_kwargs = {
        "cache_dir": args.cache_dir,
        "torch_dtype": dtype,
        "trust_remote_code": True,
        "device_map": "auto" # Safely spans across GPUs if needed
    }
    
    if use_flash_attn and torch.cuda.is_bf16_supported():
        print("Engaging FlashAttention-2 High-Speed Kernel...")
        model_kwargs["attn_implementation"] = "flash_attention_2"
        
    model = AutoModel.from_pretrained(model_id, **model_kwargs)
    model.eval()

    # 5. Dataloader execution
    dataset = StringDataset(all_chunks)
    # Using multiple workers is not strictly necessary for pure string lists
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    print(f"\nCommencing High-Throughput CUDA Extraction! Batch Size: {args.batch_size}")
    
    start_idx = 0
    with torch.no_grad():
        for batch_texts in tqdm(dataloader, desc=f"Extracting {args.model.upper()}"):
            
            # Tokenize dynamically
            inputs = tokenizer(
                list(batch_texts), 
                padding=True, 
                truncation=True, 
                max_length=args.max_seq_len, 
                return_tensors="pt"
            ).to(device)
            
            # Forward Pass (AutoModel yields BaseModelOutput)
            outputs = model(**inputs)
            last_hidden = outputs.last_hidden_state
            
            # Pooling Strategy
            if args.model == "bge":
                # BGE-M3 standard is to use the [CLS] token (Index 0)
                embs = last_hidden[:, 0, :]
                # BGE usually normalizes
                embs = torch.nn.functional.normalize(embs, p=2, dim=1)
            else:
                # Qwen-7B causal models pool via the Last Token
                embs = get_last_token_pooling(last_hidden, inputs.attention_mask)
                # Normalization is strongly recommended for GTE-Qwen
                embs = torch.nn.functional.normalize(embs, p=2, dim=1)
                
            # Direct Write-to-Disk (Float16 precision)
            batch_len = embs.size(0)
            end_idx = start_idx + batch_len
            
            # Safely move off GPU and cast to numpy float16
            mmap_arr[start_idx:end_idx] = embs.cpu().to(torch.float16).numpy()
            start_idx = end_idx
            
            # Periodically flush every 10,000 items to guarantee zero data loss
            if start_idx % 10000 < args.batch_size:
                mmap_arr.flush()
                
    # Final flush
    mmap_arr.flush()
    print(f"\n[SUCCESS] Extracted Matrix fully solidified at: {out_path}")
    print(f"Matrix Shape: {mmap_arr.shape}")

if __name__ == "__main__":
    main()
