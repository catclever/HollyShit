"""
[Agent Generated] 
历史使命：发现了长难句在潜空间中的两大定律：【模长收缩效应】和【Dispersion (离散度) 减小效应】（即词汇越多，概念云抱得越紧）。
对理解高维潜空间的几何特性极有帮助。
用法：uv run python dev/test_sentence_complexity.py --ckpt checkpoints/.../student.pt
"""
import torch
import torch.nn.functional as F
import argparse
import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.phase1_charencoder_cuda import TinyCharEncoderCUDA
from training.core.char_tokenizer import CharTokenizer
import json

def get_contextual_outputs(encoder, x):
    h = encoder.get_normalized_embeddings(x)
    for layer in encoder.layers:
        h = layer(h, None)
    return encoder.final_ln(h)

def analyze_sentence(encoder, tokenizer, sentence, device, threshold=0.5):
    ids = tokenizer.encode(sentence, add_special_tokens=True)
    chars = [tokenizer.decode([idx]) for idx in ids]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    
    with torch.no_grad():
        h = get_contextual_outputs(encoder, x).squeeze(0) # (L, D)
        
        # 1. z_macro
        z_macro = h.mean(dim=0)
        z_macro_norm = torch.norm(z_macro).item()
        
        # 2. Dispersion (Average distance of tokens to z_macro)
        distances = [torch.norm(h[i] - z_macro).item() for i in range(len(h)) if chars[i] not in ["<BOS>", "<EOS>"]]
        dispersion = np.mean(distances) if distances else 0.0
        
        # 3. Dynamic Chunking (Concept Clouds)
        chunks = []
        current_chunk = [0]
        for i in range(len(h) - 1):
            if chars[i] in ["<BOS>", "<EOS>"]:
                continue
            sim = F.cosine_similarity(h[i].unsqueeze(0), h[i+1].unsqueeze(0)).item()
            if sim < threshold:
                chunks.append(current_chunk)
                current_chunk = [i+1]
            else:
                current_chunk.append(i+1)
        if current_chunk:
            chunks.append(current_chunk)
            
        # Clean chunks and calculate similarity to z_macro
        valid_chunks = 0
        sim_to_macro_list = []
        for indices in chunks:
            chunk_chars = [chars[idx].replace("\n", "").replace("<BOS>", "").replace("<EOS>", "") for idx in indices]
            chunk_text = "".join(chunk_chars)
            if not chunk_text.strip():
                continue
            valid_chunks += 1
            mu_k = h[indices].mean(dim=0)
            sim = F.cosine_similarity(mu_k.unsqueeze(0), z_macro.unsqueeze(0)).item()
            sim_to_macro_list.append(sim)
            
        avg_cloud_sim = np.mean(sim_to_macro_list) if sim_to_macro_list else 0.0
        
        return {
            "sentence": sentence,
            "len": len(chars) - 2, # exluding BOS/EOS
            "z_norm": z_macro_norm,
            "dispersion": dispersion,
            "num_clouds": valid_chunks,
            "avg_cloud_sim": avg_cloud_sim
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to student.pt")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = CharTokenizer()
    
    config_path = os.path.join(os.path.dirname(args.ckpt), "training_args.json")
    with open(config_path, "r") as f:
        cfg = json.load(f)
        
    encoder = TinyCharEncoderCUDA(
        vocab_size=tokenizer.vocab_size,
        d_model=cfg.get("student_d_model", 512),
        n_layers=cfg.get("student_n_layers", 6),
        n_heads=cfg.get("student_n_heads", 8),
        max_seq_len=cfg.get("max_seq_len", 128)
    ).to(device)
    
    encoder.load_state_dict(torch.load(args.ckpt, map_location=device))
    encoder.eval()
    
    sentences = [
        # Very Short
        "猫。",
        "你好。",
        "快跑！",
        # Short
        "我在吃饭。",
        "天黑了。",
        "今天天气真好！",
        "他昨天去了超市。",
        # Medium
        "人工智能正在飞速发展。",
        "我喜欢在下雨天喝着热茶看书。",
        "这本小说的情节跌宕起伏，让人爱不释手。",
        # Long / Rich
        "一只毛茸茸的橘猫轻巧地跳上了破旧的阳台，在夕阳的余晖中舒服地伸了一个懒腰。",
        "虽然昨天的暴风雨摧毁了沿海的许多古老建筑，但是当地居民依然保持着乐观的心态，并在清晨开始了艰难的重建工作。",
        "在这个充满不确定性的时代，掌握一门能够跨越文化壁垒的语言，不仅是生存的技能，更是理解世界复杂性的一把金钥匙。",
        "量子计算的突破性进展，预示着传统的加密算法可能在未来十年内彻底失效，这将迫使全球金融系统进行一次史无前例的底层架构升级。",
        "随着宇宙微波背景辐射数据的最新测绘，天体物理学家们终于拼凑出了早期星系形成的完整图景，这或许能解释暗物质在引力坍缩中扮演的隐秘角色。"
    ]
    
    print("\n=== Testing Sentence Complexity in Latent Space ===")
    print(f"{'Type':<6} | {'Len':<4} | {'Clouds':<6} | {'z_macro Norm':<12} | {'Dispersion':<10} | {'Avg Cloud Sim to z_macro':<25} | Sentence")
    print("-" * 130)
    
    for s in sentences:
        res = analyze_sentence(encoder, tokenizer, s, device)
        stype = "Rich" if res['len'] > 15 else "Sparse"
        print(f"{stype:<6} | {res['len']:<4} | {res['num_clouds']:<6} | {res['z_norm']:<12.4f} | {res['dispersion']:<10.4f} | {res['avg_cloud_sim']:<25.4f} | {res['sentence']}")

if __name__ == "__main__":
    main()
