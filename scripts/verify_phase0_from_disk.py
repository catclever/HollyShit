import mlx.core as mx
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
import os

from model.config import ModelConfig
from model.adapter import SensoryFuser
from model.god_encoder import GodEncoder
from model.decoder import WeakDecoder

def verify():
    print("1. Loading physical architecture...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
    
    config = ModelConfig()
    config.vocab_size = 151643 # Hardcoded to match the exact saved checkpoint parameter shape
    d_model = config.decoder_heads * 64
    
    fuser = SensoryFuser(config.emb_dims, d_model)
    god_encoder = GodEncoder(d_model, config.z_dim)
    decoder = WeakDecoder(config.z_dim, config.vocab_size, d_model=d_model, n_layers=config.decoder_layers)
    
    # Note: Using step_150000 for verification as it's the safest finalized checkpoint format available.
    ckpt_path = "checkpoints/run/p0_v1_step_150000"
    print(f"2. Loading weights from {ckpt_path}...")
    
    fuser_path = f"{ckpt_path}/sense_fuser.safetensors"
    if not os.path.exists(fuser_path):
        fuser_path = f"{ckpt_path}/sense_adapter.safetensors"
    fuser.load_weights(fuser_path)
    god_encoder.load_weights(f"{ckpt_path}/god_encoder.safetensors")
    decoder.load_weights(f"{ckpt_path}/decoder.safetensors")
    
    print("3. Pulling random sentence #100,000 from Parquet (Source of truth)...")
    df = pd.read_parquet("data/Basic_ZH/chunked_mixed_wiki.parquet")
    text_chunks = df['chunks'].explode().dropna().tolist()
    target_text = text_chunks[100000]
    print(f"\n======================================")
    print(f"[ORIGINAL TEXT (Ground Truth)]:\n{target_text}")
    print(f"======================================\n")
    
    del df # Free memory
    
    print("4. Fetching corresponding Embeddings purely from .npy disk...")
    emb_files = [
        "data/Basic_ZH/embs/hy-tmp/roberta_embeddings.npy",
        "data/Basic_ZH/embs/hy-tmp/gte_embeddings.npy",
        "data/Basic_ZH/embs/hy-tmp/bge_embeddings.npy",
        "data/Basic_ZH/embs/hy-tmp/text2vec_embeddings.npy"
    ]
    
    embs = []
    for path in emb_files:
        arr = np.load(path, mmap_mode='r')
        # We slice exactly one row to simulate a batch size of 1
        embs.append(mx.array(arr[100000:100001])) 
        
    print("5. Forwarding through SensoryFuser and GodEncoder -> z_target...")
    # Using centroid mean or default alpha fusion
    f_t = fuser(embs, weights=None)
    z_target = god_encoder(f_t)
    
    print(f"   => z_target synthesized. Shape: {z_target.shape}")
    
    print("6. Autoregressive Decoding from z_target (The Topological Dream)...")
    
    # Auto-regressive generation requires a start token.
    # Qwen uses specific IDs, but if pad_token_id is none, eos is 151643
    bos_token = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (tokenizer.eos_token_id or 0)
    
    # We call the newly implemented generate method on Decoder
    generated_ids = decoder.generate(z_target, start_token=bos_token, max_tokens=100, temperature=0.7)
    
    decoded_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    print(f"\n======================================")
    print(f"[DREAM RECONSTRUCTION (WeakDecoder)]:\n{decoded_text}")
    print(f"======================================\n")

if __name__ == "__main__":
    verify()
