"""
[Agent Generated] 
历史使命：证明了“字向量”本质上是潜空间里的【方向移动向量（Directional Shifts）】，而不是概念本身。
这是证明“纯思想不需要字”的关键物理学定律测试，非常有纪念和参考价值。
用法：uv run python dev/test_character_directions.py --ckpt checkpoints/.../student.pt
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

def get_concept_vector(encoder, tokenizer, sentence, target_word, device):
    ids = tokenizer.encode(sentence, add_special_tokens=True)
    chars = [tokenizer.decode([idx]) for idx in ids]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    
    with torch.no_grad():
        h = encoder.get_normalized_embeddings(x)
        for layer in encoder.layers:
            h = layer(h, None)
        h = encoder.final_ln(h).squeeze(0) # (L, D)
        
    # Find target chunk
    target_len = len(target_word)
    target_indices = []
    for j in range(len(chars) - target_len + 1):
        match = True
        for k in range(target_len):
            if chars[j+k] != target_word[k]:
                match = False
                break
        if match:
            target_indices = list(range(j, j+target_len))
            break
            
    if target_indices:
        # Return the concept cloud center (Mean Pooling)
        return h[target_indices].mean(dim=0)
    else:
        raise ValueError(f"Target '{target_word}' not found in '{sentence}'")

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
        max_seq_len=cfg.get("max_seq_len", 64)
    ).to(device)
    
    encoder.load_state_dict(torch.load(args.ckpt, map_location=device))
    encoder.eval()
    
    template = "我看到了一只{}在路边。"
    
    # Base concepts
    v_cat = get_concept_vector(encoder, tokenizer, template.format("猫"), "猫", device)
    v_dog = get_concept_vector(encoder, tokenizer, template.format("狗"), "狗", device)
    v_bird = get_concept_vector(encoder, tokenizer, template.format("鸟"), "鸟", device)
    
    # Modified concepts
    v_orange_cat = get_concept_vector(encoder, tokenizer, template.format("橘猫"), "橘猫", device)
    v_orange_dog = get_concept_vector(encoder, tokenizer, template.format("橘狗"), "橘狗", device)
    v_orange_bird = get_concept_vector(encoder, tokenizer, template.format("橘鸟"), "橘鸟", device)
    
    v_black_cat = get_concept_vector(encoder, tokenizer, template.format("黑猫"), "黑猫", device)
    v_black_dog = get_concept_vector(encoder, tokenizer, template.format("黑狗"), "黑狗", device)
    
    print("\n=== Hypothesis: '橘' (Orange) is a direction of change ===")
    
    # Calculate shift vectors (Directions of change)
    delta_orange_cat = v_orange_cat - v_cat
    delta_orange_dog = v_orange_dog - v_dog
    delta_orange_bird = v_orange_bird - v_bird
    
    print(f"Shift '橘' on 猫 (Orange Cat - Cat) vs Shift '橘' on 狗 (Orange Dog - Dog): "
          f"{F.cosine_similarity(delta_orange_cat.unsqueeze(0), delta_orange_dog.unsqueeze(0)).item():.4f}")
    
    print(f"Shift '橘' on 猫 (Orange Cat - Cat) vs Shift '橘' on 鸟 (Orange Bird - Bird): "
          f"{F.cosine_similarity(delta_orange_cat.unsqueeze(0), delta_orange_bird.unsqueeze(0)).item():.4f}")
          
    print("\n=== Contrast Check: '黑' (Black) vs '橘' (Orange) ===")
    delta_black_cat = v_black_cat - v_cat
    delta_black_dog = v_black_dog - v_dog
    
    print(f"Shift '黑' on 猫 vs Shift '黑' on 狗: "
          f"{F.cosine_similarity(delta_black_cat.unsqueeze(0), delta_black_dog.unsqueeze(0)).item():.4f}")
          
    print(f"Shift '黑' (Black) vs Shift '橘' (Orange) on 猫: "
          f"{F.cosine_similarity(delta_black_cat.unsqueeze(0), delta_orange_cat.unsqueeze(0)).item():.4f}")

if __name__ == "__main__":
    main()
