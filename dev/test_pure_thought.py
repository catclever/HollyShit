"""
[Agent Generated] 
历史使命：证明了原来的 z_macro（Mean Pooling）根本不懂“主动 vs 被动”的纯逻辑，它只是个愚蠢的词袋模型（猫追老鼠 vs 老鼠被猫追 相似度极低）。
它成功完成了它的历史使命，证明 Mean Pooling 会发生致命碰撞，把我们逼向了 DiT 的无序点云方案。
用法：uv run python dev/test_pure_thought.py --ckpt checkpoints/.../student.pt
"""
import torch
import torch.nn.functional as F
import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.phase1_charencoder_cuda import TinyCharEncoderCUDA
from training.core.char_tokenizer import CharTokenizer
import json

def get_z_macro(encoder, tokenizer, sentence, device):
    ids = tokenizer.encode(sentence, add_special_tokens=True)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        h = encoder.get_normalized_embeddings(x)
        for layer in encoder.layers:
            h = layer(h, None)
        h = encoder.final_ln(h).squeeze(0)
        return h.mean(dim=0)

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
    
    s_active = "猫追老鼠。"
    s_passive = "老鼠被猫追。"
    s_reverse = "老鼠追猫。"
    
    z_a = get_z_macro(encoder, tokenizer, s_active, device)
    z_p = get_z_macro(encoder, tokenizer, s_passive, device)
    z_r = get_z_macro(encoder, tokenizer, s_reverse, device)
    
    print("\n=== Testing Pure Thought Hypothesis ===")
    sim_ap = F.cosine_similarity(z_a.unsqueeze(0), z_p.unsqueeze(0)).item()
    sim_ar = F.cosine_similarity(z_a.unsqueeze(0), z_r.unsqueeze(0)).item()
    
    print(f"Active vs Passive (Different order, SAME meaning): {sim_ap:.4f} | {s_active} vs {s_passive}")
    print(f"Active vs Reverse (Similar order, DIFF meaning): {sim_ar:.4f} | {s_active} vs {s_reverse}")

if __name__ == "__main__":
    main()
