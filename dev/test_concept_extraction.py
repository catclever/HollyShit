import torch
import torch.nn.functional as F
import os
import sys

from model.phase1_charencoder_cuda import TinyCharEncoderCUDA as TinyCharEncoder
from training.core.char_tokenizer import CharTokenizer

def get_concept_clouds_with_norms(encoder, tokenizer, sentence, device, threshold=0.5):
    ids = tokenizer.encode(sentence, add_special_tokens=True)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        h = encoder.get_normalized_embeddings(x)
        for layer in encoder.layers:
            h = layer(h, None)
            
        h_before_ln = h.squeeze(0) # [L, 512]
        h_after_ln = encoder.final_ln(h).squeeze(0) # [L, 512]
    
    tokens = [tokenizer.id_to_char.get(i, "[UNK]") for i in ids]
    
    # We use h_after_ln to do the chunking (cosine similarity is usually done after LN)
    # But we can calculate the norm of the chunk before LN to measure "Energy" / "Importance"
    
    clouds = []
    current_cloud_vecs = []
    current_cloud_vecs_before_ln = []
    current_cloud_tokens = []
    
    for i in range(len(h_after_ln)):
        if not current_cloud_vecs:
            current_cloud_vecs.append(h_after_ln[i])
            current_cloud_vecs_before_ln.append(h_before_ln[i])
            current_cloud_tokens.append(tokens[i])
        else:
            current_mean = torch.stack(current_cloud_vecs).mean(dim=0)
            sim = F.cosine_similarity(h_after_ln[i].unsqueeze(0), current_mean.unsqueeze(0)).item()
            
            if sim > threshold:
                current_cloud_vecs.append(h_after_ln[i])
                current_cloud_vecs_before_ln.append(h_before_ln[i])
                current_cloud_tokens.append(tokens[i])
            else:
                mean_vec_before_ln = torch.stack(current_cloud_vecs_before_ln).mean(dim=0)
                norm = torch.norm(mean_vec_before_ln, p=2).item()
                clouds.append({
                    "tokens": "".join(current_cloud_tokens),
                    "norm": norm
                })
                
                current_cloud_vecs = [h_after_ln[i]]
                current_cloud_vecs_before_ln = [h_before_ln[i]]
                current_cloud_tokens = [tokens[i]]
                
    if current_cloud_vecs:
        mean_vec_before_ln = torch.stack(current_cloud_vecs_before_ln).mean(dim=0)
        norm = torch.norm(mean_vec_before_ln, p=2).item()
        clouds.append({
            "tokens": "".join(current_cloud_tokens),
            "norm": norm
        })
        
    return clouds

def test_long_sentence_splitting():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = CharTokenizer()
    encoder = TinyCharEncoder(vocab_size=tokenizer.vocab_size).to(device)
    
    checkpoint = torch.load('checkpoints/basencoder_step_115000/student.pt', map_location=device)
    encoder.load_state_dict(checkpoint)
    encoder.eval()
    
    text = "老鼠今天饿极了，偷偷跑出洞口想找点奶酪吃。不料一只黑猫正潜伏在沙发后面。黑猫突然猛扑过去，老鼠吓得拼命钻回了洞里。"
    
    clouds = get_concept_clouds_with_norms(encoder, tokenizer, text, device, threshold=0.5)
    
    print(f"{'Cloud':<30} | {'Norm (Energy / Importance)':<15}")
    print("-" * 50)
    
    # Sort by norm to see the most important concepts
    sorted_clouds = sorted(clouds, key=lambda x: x["norm"], reverse=True)
    
    for cloud in sorted_clouds:
        print(f"{cloud['tokens']:<30} | {cloud['norm']:<15.4f}")

if __name__ == "__main__":
    test_long_sentence_splitting()
