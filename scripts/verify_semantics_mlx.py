"""
【脚本功能】：多态降维闭环自动审判庭 (Semantic Assessor)
【使用场景】：Phase 0 验证阶段。这是最严格的“黑箱语义保鲜”自动化测试。系统会将生成的杂牌“幻觉”文本与原意在独立的高精度 Encoder（独立评委）里过一遍余弦相似度（Cosine Similarity）。一旦余弦断崖，代表系统发生语义坍缩。
【用法示例】：
    - 随机抽测：`python scripts/verify_semantics.py --ckpt checkpoints/run/p0_v1_step_150000 --mode random`
    - 实时对话冷测：`python scripts/verify_semantics.py --ckpt ... --mode interactive`
"""
import mlx.core as mx
import numpy as np
import pandas as pd
import os
import argparse
import random
import torch
import torch.nn.functional as F

from training.char_tokenizer import CharTokenizer
from model.config import ModelConfig
from model.adapter import SensoryFuser
from model.god_encoder import GodEncoder
from model.decoder import WeakDecoder

# Using SentenceTransformers for actual semantic metrics
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Please install sentence-transformers: pip install sentence-transformers")
    import sys
    sys.exit(1)

def compute_cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    # emb1, emb2 are purely numpy arrays of shape (D,)
    e1 = torch.tensor(emb1).unsqueeze(0)
    e2 = torch.tensor(emb2).unsqueeze(0)
    similarity = F.cosine_similarity(e1, e2).item()
    return similarity

def verify_semantics():
    parser = argparse.ArgumentParser(description="Closed-Loop Semantic Verification")
    parser.add_argument("--num_samples", type=int, default=3, help="Number of random sentences to evaluate (also caps file mode)")
    parser.add_argument("--emb_idx", type=int, default=-1, help="-1: fully fused. 0:roberta, 1:gte, 2:bge, 3:text2vec")
    parser.add_argument("--mode", type=str, choices=["random", "file", "interactive"], default="random", help="Test mode")
    parser.add_argument("--file_path", type=str, default="", help="Path to text file containing test sentences (required for 'file' mode)")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint directory")
    args = parser.parse_args()

    print("1. Loading physical architecture...")
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
    
    # ---------------------------------------------------------
    # The Independent Semantic Judge
    MODEL_MAPPING = {
        0: 'hfl/chinese-roberta-wwm-ext',
        1: 'thenlper/gte-large-zh',
        2: 'BAAI/bge-large-zh-v1.5',
        3: 'shibing624/text2vec-base-chinese'
    }
    
    judge_idx = args.emb_idx if args.emb_idx != -1 else 2 # Default to BGE if fully fused
    model_name = MODEL_MAPPING[judge_idx]
    
    print(f"Loading Semantic Judge ({model_name})... this might take a moment")
    judge_model = SentenceTransformer(model_name)
    print("Judge Online. Ready to evaluate semantic preservation.")
    
    weights = None
    if args.emb_idx != -1:
        weights = [0.0] * 4
        weights[args.emb_idx] = 1.0

    LIVE_MODELS = {judge_idx: judge_model}
    
    def get_live_embs(text, weights):
        """Lazy loads SentenceTransformers to encode unseen text on the fly."""
        embs = []
        dims = config.emb_dims
        for i in range(4):
            if weights is None or weights[i] > 0.0:
                if i not in LIVE_MODELS:
                    print(f"\n[Lazy Load] Activating external model {MODEL_MAPPING[i]} for live encoding...")
                    LIVE_MODELS[i] = SentenceTransformer(MODEL_MAPPING[i])
                vec = LIVE_MODELS[i].encode(text)
                embs.append(mx.array(vec).reshape(1, -1))
            else:
                embs.append(mx.zeros((1, dims[i])))
        return embs

    def process_sentence(target_text, embs, sample_idx):
        print(f"\n======================================")
        if args.mode == "random":
            print(f"       >>> Parquet SAMPLE (Row #{sample_idx}) <<<")
        else:
            print(f"       >>> Live SAMPLE {sample_idx} <<<")
        print(f"======================================")
        
        f_t = fuser(embs, weights=weights)
        z_target = god_encoder(f_t)
        
        start_token = tokenizer.bos_token_id
        eos_token = tokenizer.eos_token_id
        
        # Lowered temperature to 0.2 as requested for less hallucination
        generated_ids = decoder.generate(z_target, start_token=start_token, eos_token=eos_token, max_tokens=100, temperature=0.2)
        decoded_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        print(f"[1] Original Text : {target_text}")
        print(f"[2] Dreamed Text  : {decoded_text}\n")
        
        if not decoded_text.strip():
            print("\n   [!] Model hallucinated complete silence. Similarity: 0.000")
            return
            
        # Original Embedding Retrieval (From MMap or Live Encoding)
        if args.mode == 'random':
            judge_orig_emb = embs[judge_idx][0].tolist() 
        else:
            judge_orig_emb = LIVE_MODELS[judge_idx].encode(target_text).tolist()
            
        judge_dream_emb = judge_model.encode(decoded_text).tolist()
        
        similarity = compute_cosine_similarity(np.array(judge_orig_emb), np.array(judge_dream_emb))
        print(f"   >>> [Semantic Cosine Similarity]: {similarity:.4f} / 1.000 <<<")
        
        if similarity > 0.8:
            print("   (Verdict: Outstanding Concept Preservation!)")
        elif similarity > 0.6:
            print("   (Verdict: Core Concepts Recognized but Syntax Tangled)")
        else:
            print("   (Verdict: Semantic Collapse)")

    # ========================== MODE EXECUTION ==========================
    if args.mode == "random":
        print(f"Pulling {args.num_samples} random sentence(s) from Parquet (Source of truth)...")
        df = pd.read_parquet("data/Basic_ZH/chunked_mixed_wiki.parquet")
        text_chunks = df['chunks'].explode().dropna().tolist()
        total_chunks = len(text_chunks)
        del df 

        print("Mmapping all .npy disk arrays...")
        emb_files = [
            "data/Basic_ZH/embs/hy-tmp/roberta_embeddings.npy",
            "data/Basic_ZH/embs/hy-tmp/gte_embeddings.npy",
            "data/Basic_ZH/embs/hy-tmp/bge_embeddings.npy",
            "data/Basic_ZH/embs/hy-tmp/text2vec_embeddings.npy"
        ]
        embs_mmap = [np.load(path, mmap_mode='r') for path in emb_files]
        sampled_indices = random.sample(range(total_chunks), args.num_samples)
        
        for i, idx in enumerate(sampled_indices):
            target_text = text_chunks[idx]
            embs = [mx.array(arr[idx:idx+1]) for arr in embs_mmap]
            process_sentence(target_text, embs, idx)
            
    elif args.mode == "file":
        if not args.file_path:
            print("[ERROR] --file_path must be provided when using --mode file")
            return
        
        with open(args.file_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
            
        print(f"Processing up to {args.num_samples} lines from {args.file_path}...")
        for i, target_text in enumerate(lines[:args.num_samples]):
            embs = get_live_embs(target_text, weights)
            process_sentence(target_text, embs, i+1)
            
    elif args.mode == "interactive":
        print("\n进入交互式验证模式 (Interactive Mode)")
        print("提示：输入 'quit' 或 'exit' 退出。首次执行长句将冷启动加载 Embedding 模型。")
        i = 1
        while True:
            target_text = input(f"\n[You] 请输入一句测试文本 (Sample {i}): ")
            if target_text.strip().lower() in ["quit", "exit", "q"]:
                print("Exit signal received. Goodbye!")
                break
            if not target_text.strip():
                continue
                
            embs = get_live_embs(target_text, weights)
            process_sentence(target_text, embs, i)
            i += 1

if __name__ == "__main__":
    verify_semantics()
