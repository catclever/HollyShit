"""
[Agent Generated] 
历史使命：终极形态验证。结合了 0.5 阈值切分法，证明了最终提纯出来的【概念云块（Concept Clouds）】依然完美保留了物理拓扑坐标！
这是支持我们在 Phase 2 使用 DiT + 最优传输(OT) Loss 生成无序点云的最强理论基石代码。
用法：uv run python dev/test_concept_cloud_roles.py --ckpt checkpoints/.../student.pt
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

def get_concept_clouds(encoder, tokenizer, sentence, device, threshold=0.5):
    ids = tokenizer.encode(sentence, add_special_tokens=True)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        h = encoder.get_normalized_embeddings(x)
        for layer in encoder.layers:
            h = layer(h, None)
        h = encoder.final_ln(h).squeeze(0) # [L, 512]
    
    tokens = [tokenizer.id_to_char.get(i, "[UNK]") for i in ids]
    
    # Greedy chunking based on cosine similarity > threshold
    clouds = []
    current_cloud_vecs = []
    current_cloud_tokens = []
    
    for i in range(len(h)):
        if not current_cloud_vecs:
            current_cloud_vecs.append(h[i])
            current_cloud_tokens.append(tokens[i])
        else:
            # Check similarity with the mean of the current cloud
            current_mean = torch.stack(current_cloud_vecs).mean(dim=0)
            sim = F.cosine_similarity(h[i].unsqueeze(0), current_mean.unsqueeze(0)).item()
            
            if sim > threshold:
                current_cloud_vecs.append(h[i])
                current_cloud_tokens.append(tokens[i])
            else:
                clouds.append({
                    "mean_vec": current_mean,
                    "tokens": current_cloud_tokens
                })
                current_cloud_vecs = [h[i]]
                current_cloud_tokens = [tokens[i]]
                
    if current_cloud_vecs:
        current_mean = torch.stack(current_cloud_vecs).mean(dim=0)
        clouds.append({
            "mean_vec": current_mean,
            "tokens": current_cloud_tokens
        })
        
    return clouds

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
        "猫追老鼠",
        "老鼠被猫追",
        "老鼠追猫"
    ]
    
    results = {}
    for s in sentences:
        print(f"\n--- Sentence: {s} ---")
        clouds = get_concept_clouds(encoder, tokenizer, s, device, threshold=0.5)
        results[s] = clouds
        for idx, c in enumerate(clouds):
            print(f"Cloud {idx}: {''.join(c['tokens'])}")
            
    # Find the cloud containing "猫" for each sentence
    def get_cat_cloud(clouds):
        for c in clouds:
            if "猫" in c['tokens']:
                return c['mean_vec']
        return None

    cat_a = get_cat_cloud(results["猫追老鼠"])
    cat_b = get_cat_cloud(results["老鼠被猫追"])
    cat_c = get_cat_cloud(results["老鼠追猫"])
    
    print("\n=== Concept Cloud Role Test [猫所在的云] ===")
    if cat_a is not None and cat_b is not None:
        print(f"A(Agent) vs B(Agent, Passive): {F.cosine_similarity(cat_a.unsqueeze(0), cat_b.unsqueeze(0)).item():.4f}")
    if cat_a is not None and cat_c is not None:
        print(f"A(Agent) vs C(Patient):        {F.cosine_similarity(cat_a.unsqueeze(0), cat_c.unsqueeze(0)).item():.4f}")

if __name__ == "__main__":
    main()
