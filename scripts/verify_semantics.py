import mlx.core as mx
import numpy as np
import pandas as pd
import os
import argparse
import random
import torch
import torch.nn.functional as F

from training.char_tokenizer import CharTokenizer
from model.config import ModelConfig
from model.adapter import SensoryFuser
from model.god_encoder import GodEncoder
from model.decoder import WeakDecoder

# Using SentenceTransformers for actual semantic metrics
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Please install sentence-transformers: pip install sentence-transformers")
    import sys
    sys.exit(1)

def compute_cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    # emb1, emb2 are purely numpy arrays of shape (D,)
    e1 = torch.tensor(emb1).unsqueeze(0)
    e2 = torch.tensor(emb2).unsqueeze(0)
    similarity = F.cosine_similarity(e1, e2).item()
    return similarity

def verify_semantics():
    parser = argparse.ArgumentParser(description="Closed-Loop Semantic Verification")
    parser.add_argument("--num_samples", type=int, default=3, help="Number of random sentences to evaluate")
    parser.add_argument("--emb_idx", type=int, default=-1, help="-1: fully fused. 0:roberta, 1:gte, 2:bge, 3:text2vec")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint directory")
    args = parser.parse_args()

    print("1. Loading physical architecture...")
    tokenizer = CharTokenizer()
    config = ModelConfig()
    d_model = config.decoder_heads * 64
    
    fuser = SensoryFuser(config.emb_dims, d_model)
    god_encoder = GodEncoder(d_model, config.z_dim)
    decoder = WeakDecoder(config.z_dim, config.vocab_size, d_model=d_model, n_layers=config.decoder_layers)
    
    ckpt_path = args.ckpt
    print(f"2. Loading weights from {ckpt_path}...")
    
    fuser_path = f"{ckpt_path}/sense_fuser.safetensors"
    if not os.path.exists(fuser_path):
        fuser_path = f"{ckpt_path}/sense_adapter.safetensors"
    fuser.load_weights(fuser_path)
    god_encoder.load_weights(f"{ckpt_path}/god_encoder.safetensors")
    decoder.load_weights(f"{ckpt_path}/decoder.safetensors")
    
    # ---------------------------------------------------------
    # The Independent Semantic Judge
    # We dynamically load the exact HuggingFace model corresponding to the feature space being tested
    MODEL_MAPPING = {
        0: 'hfl/chinese-roberta-wwm-ext',
        1: 'thenlper/gte-large-zh',
        2: 'BAAI/bge-large-zh-v1.5',
        3: 'shibing624/text2vec-base-chinese'
    }
    
    judge_idx = args.emb_idx if args.emb_idx != -1 else 2 # Default to BGE if fully fused
    model_name = MODEL_MAPPING[judge_idx]
    
    print(f"Loading Semantic Judge ({model_name})... this might take a moment")
    judge_model = SentenceTransformer(model_name)
    print(f"{model_name} Judge Online. Ready to evaluate semantic preservation against Physical Disk Arrays.")
    # ---------------------------------------------------------

    print(f"3. Pulling {args.num_samples} random sentence(s) from Parquet (Source of truth)...")
    df = pd.read_parquet("data/Basic_ZH/chunked_mixed_wiki.parquet")
    text_chunks = df['chunks'].explode().dropna().tolist()
    total_chunks = len(text_chunks)
    
    del df # Free memory
    sampled_indices = random.sample(range(total_chunks), args.num_samples)
    
    print("4. Mmapping all .npy disk arrays...")
    emb_files = [
        "data/Basic_ZH/embs/hy-tmp/roberta_embeddings.npy",
        "data/Basic_ZH/embs/hy-tmp/gte_embeddings.npy",
        "data/Basic_ZH/embs/hy-tmp/bge_embeddings.npy",
        "data/Basic_ZH/embs/hy-tmp/text2vec_embeddings.npy"
    ]
    embs_mmap = [np.load(path, mmap_mode='r') for path in emb_files]
    
    for i, idx in enumerate(sampled_indices):
        target_text = text_chunks[idx]
        print(f"\n======================================")
        print(f"       >>> SAMPLE {i+1} (Row #{idx}) <<<")
        print(f"======================================")
        
        embs = [mx.array(arr[idx:idx+1]) for arr in embs_mmap]
            
        weights = None
        if args.emb_idx != -1:
            weights = [0.0] * 4
            weights[args.emb_idx] = 1.0
        
        f_t = fuser(embs, weights=weights)
        z_target = god_encoder(f_t)
        
        # THE CAUSAL COLLAPSE FIX (REVISED):
        # We discovered that the training script DOES natively inject <BOS> (id 258) at the start of every sequence!
        # The true bug was simply that the verification script was previously asking the model 
        # to generate from <PAD> (id 256) instead of <BOS>.
        # Now we use the completely native trained <BOS> token to kickstart the unconditional generation!
        start_token = tokenizer.bos_token_id
        eos_token = tokenizer.eos_token_id
        
        # Lowered temperature to 0.2 as requested for less hallucination
        generated_ids = decoder.generate(z_target, start_token=start_token, eos_token=eos_token, max_tokens=100, temperature=0.2)
        decoded_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        print(f"[1] Original Text : {target_text}")
        print(f"[2] Dreamed Text  : {decoded_text}")
        
        # Now we pass BOTH through the independent BGE Judge to grade the "Concept Fidelity"
        if not decoded_text.strip():
            print("\n   [!] Model hallucinated complete silence. Similarity: 0.000")
            continue
            
        # USER'S ARCHITECTURAL GENIUS: 
        # Do NOT re-embed the original text. The original text's EXACT physical representation 
        # that the GodEncoder saw is locked in the disk array!
        # We must pull it directly from the npy file.
        judge_orig_emb = embs[judge_idx][0].tolist()
        
        # We only embed the DREAM
        judge_dream_emb = judge_model.encode([decoded_text])[0]
        
        similarity = compute_cosine_similarity(np.array(judge_orig_emb), judge_dream_emb)
        
        print(f"\n   >>> [Semantic Cosine Similarity]: {similarity:.4f} / 1.000 <<<")
        if similarity > 0.8:
            print("   (Verdict: Outstanding Concept Preservation!)")
        elif similarity > 0.6:
            print("   (Verdict: Core Concepts Recognized but Syntax Tangled)")
        else:
            print("   (Verdict: Semantic Collapse)")
            
    print(f"\n======================================================\n")

if __name__ == "__main__":
    verify_semantics()
