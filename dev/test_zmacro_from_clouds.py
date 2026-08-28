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
    
    for i in range(len(h)):
        if not current_cloud_vecs:
            current_cloud_vecs.append(h[i])
        else:
            current_mean = torch.stack(current_cloud_vecs).mean(dim=0)
            sim = F.cosine_similarity(h[i].unsqueeze(0), current_mean.unsqueeze(0)).item()
            
            if sim > threshold:
                current_cloud_vecs.append(h[i])
            else:
                clouds.append(current_mean)
                current_cloud_vecs = [h[i]]
                
    if current_cloud_vecs:
        current_mean = torch.stack(current_cloud_vecs).mean(dim=0)
        clouds.append(current_mean)
        
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
    
    sentences = ["猫追老鼠", "老鼠被猫追", "老鼠追猫"]
    
    z_v2 = {}
    for s in sentences:
        clouds = get_concept_clouds(encoder, tokenizer, s, device, threshold=0.5)
        # z_macro_v2 is the Mean of the Concept Clouds!
        z = torch.stack(clouds).mean(dim=0)
        z_v2[s] = z
        
    print("\n=== z_macro_v2 (Mean of Clouds) Collision Test ===")
    sim_ap = F.cosine_similarity(z_v2["猫追老鼠"].unsqueeze(0), z_v2["老鼠被猫追"].unsqueeze(0)).item()
    sim_ar = F.cosine_similarity(z_v2["猫追老鼠"].unsqueeze(0), z_v2["老鼠追猫"].unsqueeze(0)).item()
    
    print(f"Active vs Passive (SAME meaning): {sim_ap:.4f}")
    print(f"Active vs Reverse (DIFF meaning): {sim_ar:.4f}")

if __name__ == "__main__":
    main()
