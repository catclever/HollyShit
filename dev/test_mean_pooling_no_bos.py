import torch
import torch.nn.functional as F
import os
import sys

from model.phase1_charencoder_cuda import TinyCharEncoderCUDA as TinyCharEncoder
from training.core.char_tokenizer import CharTokenizer

def test_pooling():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = CharTokenizer()
    encoder = TinyCharEncoder(vocab_size=tokenizer.vocab_size).to(device)
    
    checkpoint = torch.load('checkpoints/basencoder_step_115000/student.pt', map_location=device)
    encoder.load_state_dict(checkpoint)
    encoder.eval()
    
    s1 = "猫追老鼠"
    s2 = "老鼠被猫追"
    s3 = "老鼠追猫"
    
    with torch.no_grad():
        ids1 = torch.tensor([tokenizer.encode(s1, add_special_tokens=True)]).to(device)
        ids2 = torch.tensor([tokenizer.encode(s2, add_special_tokens=True)]).to(device)
        ids3 = torch.tensor([tokenizer.encode(s3, add_special_tokens=True)]).to(device)
        
        out1 = encoder.get_normalized_embeddings(ids1)
        for layer in encoder.layers:
            out1 = layer(out1, None)
        out1 = encoder.final_ln(out1).squeeze(0) # (L, D)
        
        out2 = encoder.get_normalized_embeddings(ids2)
        for layer in encoder.layers:
            out2 = layer(out2, None)
        out2 = encoder.final_ln(out2).squeeze(0) # (L, D)
        
        out3 = encoder.get_normalized_embeddings(ids3)
        for layer in encoder.layers:
            out3 = layer(out3, None)
        out3 = encoder.final_ln(out3).squeeze(0) # (L, D)
        
        # Mean pooling WITH BOS/EOS
        mean1_all = out1.mean(dim=0).unsqueeze(0)
        mean2_all = out2.mean(dim=0).unsqueeze(0)
        mean3_all = out3.mean(dim=0).unsqueeze(0)
        
        sim_1_2_all = F.cosine_similarity(mean1_all, mean2_all).item()
        sim_1_3_all = F.cosine_similarity(mean1_all, mean3_all).item()
        sim_2_3_all = F.cosine_similarity(mean2_all, mean3_all).item()
        
        # Mean pooling WITHOUT BOS/EOS
        mean1_chars = out1[1:-1].mean(dim=0).unsqueeze(0)
        mean2_chars = out2[1:-1].mean(dim=0).unsqueeze(0)
        mean3_chars = out3[1:-1].mean(dim=0).unsqueeze(0)
        
        sim_1_2_chars = F.cosine_similarity(mean1_chars, mean2_chars).item()
        sim_1_3_chars = F.cosine_similarity(mean1_chars, mean3_chars).item()
        sim_2_3_chars = F.cosine_similarity(mean2_chars, mean3_chars).item()
        
        print("=== WITH BOS & EOS (Full Sentence Average) ===")
        print(f"Similarity: [猫追老鼠] vs [老鼠被猫追] (Same semantics): {sim_1_2_all:.4f}")
        print(f"Similarity: [猫追老鼠] vs [老鼠追猫]   (Opposite semantics): {sim_1_3_all:.4f}")
        print(f"Similarity: [老鼠被猫追] vs [老鼠追猫] (Opposite semantics, same noun order): {sim_2_3_all:.4f}")
        print("\n=== WITHOUT BOS & EOS (Char Only Average) ===")
        print(f"Similarity: [猫追老鼠] vs [老鼠被猫追] (Same semantics): {sim_1_2_chars:.4f}")
        print(f"Similarity: [猫追老鼠] vs [老鼠追猫]   (Opposite semantics): {sim_1_3_chars:.4f}")
        print(f"Similarity: [老鼠被猫追] vs [老鼠追猫] (Opposite semantics, same noun order): {sim_2_3_chars:.4f}")

if __name__ == "__main__":
    test_pooling()
