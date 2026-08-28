"""
[Agent Generated] 
历史使命：最初验证了公主的伟大直觉——通过 0.5 相似度阈值，可以将字向量切分成连续的 Concept Clouds（概念云）。
这是未来 Phase 2 DataLoader 提取 Ground Truth 的核心算法原形 (MLX 版)。
用法：uv run python dev/test_tokenizer_by_similarity_mlx.py --ckpt checkpoints/.../student.safetensors
"""
import mlx.core as mx
import argparse
import sys
import os

# Append project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.phase1_charencoder import TinyCharEncoder
from training.core.char_tokenizer import CharTokenizer
import json

def cosine_similarity(v1, v2):
    # MLX cosine similarity
    dot = mx.sum(v1 * v2)
    norm1 = mx.linalg.norm(v1)
    norm2 = mx.linalg.norm(v2)
    return dot / (norm1 * norm2 + 1e-8)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to student.safetensors")
    parser.add_argument("--prompt", type=str, default="一只毛茸茸的橘猫轻巧地跳上了破旧的阳台", help="Sentence to test")
    parser.add_argument("--threshold", type=float, default=0.7, help="Similarity threshold for word boundary")
    args = parser.parse_args()
    
    tokenizer = CharTokenizer()
    
    # Load config from companion file
    config_path = os.path.join(os.path.dirname(args.ckpt), "training_args.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            cfg = json.load(f)
            d_model = cfg.get("student_d_model", 512)
            n_layers = cfg.get("student_n_layers", 6)
            n_heads = cfg.get("student_n_heads", 8)
            max_seq_len = cfg.get("max_seq_len", 64)
    else:
        d_model, n_layers, n_heads, max_seq_len = 512, 6, 8, 64
        
    encoder = TinyCharEncoder(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        max_seq_len=max_seq_len
    )
    
    print(f"Loading MLX TinyCharEncoder from {args.ckpt}")
    encoder.load_weights(args.ckpt)
    encoder.eval()
    
    # Tokenize
    ids = tokenizer.encode(args.prompt, add_special_tokens=True)
    chars = [tokenizer.decode([i]) for i in ids]
    
    x = mx.array([ids], dtype=mx.int32)
    
    # Get contextualized vectors (B, L, D) using return_seq=True
    h = encoder(x, return_seq=True)
    h = h[0] # (L, D)
    
    # Calculate pairwise cosine similarities
    print("\n--- Unsupervised Tokenization via Contextual Similarity (MLX) ---")
    print(f"Sentence: {args.prompt}")
    print("-" * 50)
    
    similarities = []
    for i in range(len(h) - 1):
        v1 = h[i]
        v2 = h[i+1]
        sim = cosine_similarity(v1, v2).item()
        similarities.append(sim)
        
        char1 = chars[i].replace("\n", "\\n")
        char2 = chars[i+1].replace("\n", "\\n")
        # Highlight strong connections
        indicator = "====>" if sim > args.threshold else "  |  "
        print(f"{char1} {indicator} {char2} : Similarity = {sim:.4f}")
        
    # Detect boundaries (simple threshold logic)
    print(f"\n--- Proposed Word Boundaries (Threshold < {args.threshold}) ---")
    words = []
    current_word = chars[0]
    
    for i, sim in enumerate(similarities):
        # If similarity drops below threshold, we slice it
        if sim < args.threshold:
            words.append(current_word)
            current_word = chars[i+1]
        else:
            current_word += chars[i+1]
    words.append(current_word)
    
    # Clean up special tokens for display
    clean_words = [w.replace("<BOS>", "").replace("<EOS>", "") for w in words]
    clean_words = [w for w in clean_words if w.strip()]
    print(" | ".join(clean_words))
    print("=" * 50)
        
if __name__ == "__main__":
    main()
