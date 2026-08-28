import torch
import torch.nn.functional as F
import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.phase1_charencoder_cuda import TinyCharEncoderCUDA
from training.core.char_tokenizer import CharTokenizer
import json

def get_char_vecs(encoder, tokenizer, sentence, device):
    ids = tokenizer.encode(sentence, add_special_tokens=True)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        h = encoder.get_normalized_embeddings(x)
        for layer in encoder.layers:
            h = layer(h, None)
        h = encoder.final_ln(h).squeeze(0)
    
    tokens = [tokenizer.id_to_char.get(i, "[UNK]") for i in ids]
    
    # Store vectors in a dict keyed by token string for easy access
    vecs = {}
    for t, vec in zip(tokens, h):
        if t not in ["<BOS>", "<EOS>", "被"]:
            vecs[t] = vec
    return vecs, h.mean(dim=0)

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
    
    s1 = "猫追老鼠"
    s2 = "老鼠追猫"
    
    vecs1, mean1 = get_char_vecs(encoder, tokenizer, s1, device)
    vecs2, mean2 = get_char_vecs(encoder, tokenizer, s2, device)
    
    # Calculate Char-only Mean Pooling (without BOS/EOS)
    chars1 = torch.stack([vecs1['猫'], vecs1['追'], vecs1['老'], vecs1['鼠']])
    chars2 = torch.stack([vecs2['猫'], vecs2['追'], vecs2['老'], vecs2['鼠']])
    mean1_chars = chars1.mean(dim=0)
    mean2_chars = chars2.mean(dim=0)
    
    print("=== 0. Mean Pooling Collision Test ===")
    print(f"Mean(s1) vs Mean(s2) similarity (WITH BOS/EOS): {F.cosine_similarity(mean1.unsqueeze(0), mean2.unsqueeze(0)).item():.4f}")
    print(f"Mean(s1) vs Mean(s2) similarity (WITHOUT BOS/EOS): {F.cosine_similarity(mean1_chars.unsqueeze(0), mean2_chars.unsqueeze(0)).item():.4f}")
    print()
    
    print("=== 1. The Paradox ===")
    print(f"Cat(s1) vs Cat(s2) similarity: {F.cosine_similarity(vecs1['猫'].unsqueeze(0), vecs2['猫'].unsqueeze(0)).item():.4f}")
    print(f"Mean(s1) vs Mean(s2) similarity: {F.cosine_similarity(mean1.unsqueeze(0), mean2.unsqueeze(0)).item():.4f}")
    
    print("\n=== 2. The Explanation: Vector Decomposition ===")
    # Shift from Patient to Agent
    shift_cat = vecs1['猫'] - vecs2['猫']      # (Cat as Agent) - (Cat as Patient)
    shift_mouse = vecs2['鼠'] - vecs1['鼠']  # (Mouse as Agent) - (Mouse as Patient)
    
    print(f"Role Shift Direction for Cat (Agent-Patient): norm={torch.norm(shift_cat).item():.4f}")
    print(f"Role Shift Direction for Mouse (Agent-Patient): norm={torch.norm(shift_mouse).item():.4f}")
    
    sim_shift = F.cosine_similarity(shift_cat.unsqueeze(0), shift_mouse.unsqueeze(0)).item()
    print(f"Similarity of the Role Shifts: {sim_shift:.4f}")
    
    print("\n=== 3. Vector Addition Cancellation ===")
    # sum of s1
    sum1 = vecs1['猫'] + vecs1['追'] + vecs1['老'] + vecs1['鼠']
    # sum of s2
    sum2 = vecs2['猫'] + vecs2['追'] + vecs2['老'] + vecs2['鼠']
    
    print(f"Sum of s1 vs Sum of s2: {F.cosine_similarity(sum1.unsqueeze(0), sum2.unsqueeze(0)).item():.4f}")
    
    # Difference between sums
    diff_sum = sum1 - sum2
    print(f"Norm of the difference between sums: {torch.norm(diff_sum).item():.4f}")
    
    # Prove that the difference between sums is just the sum of shifts, which cancel out!
    # diff_sum = (vecs1['猫'] - vecs2['猫']) + (vecs1['鼠'] - vecs2['鼠']) + ...
    # diff_sum = shift_cat - shift_mouse + ...
    
    shift_diff = shift_cat - shift_mouse
    print(f"Norm of (ShiftCat - ShiftMouse): {torch.norm(shift_diff).item():.4f}")
    

if __name__ == "__main__":
    main()
