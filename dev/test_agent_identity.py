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
        h = encoder.final_ln(h).squeeze(0)
    
    tokens = [tokenizer.id_to_char.get(i, "[UNK]") for i in ids]
    
    clouds = []
    current_cloud_vecs = []
    current_cloud_tokens = []
    
    for i in range(len(h)):
        if not current_cloud_vecs:
            current_cloud_vecs.append(h[i])
            current_cloud_tokens.append(tokens[i])
        else:
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
    
    s_cat_agent = "猫追老鼠"
    s_mouse_agent = "老鼠追猫"
    
    clouds_cat_agent = get_concept_clouds(encoder, tokenizer, s_cat_agent, device)
    clouds_mouse_agent = get_concept_clouds(encoder, tokenizer, s_mouse_agent, device)
    
    # 找到 句子1 里的 "猫" (Agent)
    cat_agent_vec = None
    for c in clouds_cat_agent:
        if "猫" in c['tokens'] and "老鼠" not in c['tokens']:
            cat_agent_vec = c['mean_vec']
            
    # 找到 句子2 里的 "老鼠" (Agent)
    mouse_agent_vec = None
    for c in clouds_mouse_agent:
        if "鼠" in c['tokens'] and "猫" not in c['tokens']:
            mouse_agent_vec = c['mean_vec']
            
    # 顺便找一下 句子2 里的 "猫" (Patient)
    cat_patient_vec = None
    for c in clouds_mouse_agent:
        if "猫" in c['tokens'] and "老鼠" not in c['tokens']:
            cat_patient_vec = c['mean_vec']
            
    print("=== Agent Identity vs Agent Role ===")
    if cat_agent_vec is not None and mouse_agent_vec is not None:
        sim_agents = F.cosine_similarity(cat_agent_vec.unsqueeze(0), mouse_agent_vec.unsqueeze(0)).item()
        print(f"Agent(Cat) vs Agent(Mouse): {sim_agents:.4f}")
        
    if cat_agent_vec is not None and cat_patient_vec is not None:
        sim_cat = F.cosine_similarity(cat_agent_vec.unsqueeze(0), cat_patient_vec.unsqueeze(0)).item()
        print(f"Agent(Cat) vs Patient(Cat): {sim_cat:.4f}")
        
    # Baseline: base char embeddings for cat and mouse
    cat_id = tokenizer.encode("猫", add_special_tokens=False)[0]
    mouse_id = tokenizer.encode("鼠", add_special_tokens=False)[0]
    with torch.no_grad():
        w = F.normalize(encoder.tok_emb.weight, p=2, dim=-1)
        base_cat = w[cat_id]
        base_mouse = w[mouse_id]
        sim_base = F.cosine_similarity(base_cat.unsqueeze(0), base_mouse.unsqueeze(0)).item()
        print(f"Base Embedding (Cat) vs (Mouse): {sim_base:.4f}")

if __name__ == "__main__":
    main()
