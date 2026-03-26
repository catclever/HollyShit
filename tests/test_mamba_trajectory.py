import os
import sys
import argparse
import numpy as np
import mlx.core as mx

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.config import ModelConfig
from model.adapter import SensoryFuser
from model.decoder import WeakDecoder
from model.mamba_mlx.mamba_mlx import MambaConfig
from model.mamba_planner import MambaPlanner
from training.char_tokenizer import CharTokenizer
from training.core.dataloader import Phase1DataLoader
import json

def load_models(p0_ckpt, p1_ckpt):
    config = ModelConfig()
    d_model = config.decoder_heads * 64
    
    # 1. Load Fuser and Decoder (From Phase 0)
    print(f"Loading Decoder & Fuser strictly from Phase 0: {p0_ckpt}...")
    fuser = SensoryFuser(config.emb_dims, d_model)
    decoder = WeakDecoder(config.z_dim, config.vocab_size, d_model=d_model, n_layers=config.decoder_layers)
    
    fuser_path = f"{p0_ckpt}/sense_fuser.safetensors"
    if not os.path.exists(fuser_path):
        fuser_path = f"{p0_ckpt}/sense_adapter.safetensors"
    fuser.load_weights(fuser_path)
    decoder.load_weights(f"{p0_ckpt}/decoder.safetensors")
    
    # 2. Load Trajectory Planner (From Phase 1)
    print(f"Loading Mamba Planner strictly from Phase 1: {p1_ckpt}...")
    mamba_cfg = MambaConfig(d_model=d_model, n_layers=2)
    mamba = MambaPlanner(mamba_cfg, config.z_dim, residual_mode=False) 
    mamba.load_weights(f"{p1_ckpt}/mamba_planner.safetensors")
    
    fuser.eval()
    decoder.eval()
    mamba.eval()
    
    return fuser, mamba, decoder, config

def decode_coordinate(decoder, mu_coord, tokenizer, max_len=64):
    """Autoregressively decodes a single spatial coordinate into text."""
    # mu_coord shape: (1, z_dim)
    BOS_TOKEN = 1 # Assuming 1 is BOS in our CharTokenizer
    tokens = [BOS_TOKEN]
    
    for _ in range(max_len):
        tok_arr = mx.array([tokens], dtype=mx.int32)
        # Decoder expects (z, tokens)
        logits = decoder(mu_coord, tok_arr)
        next_tok = mx.argmax(logits[:, -1, :], axis=-1).item()
        
        if next_tok == 0 or next_tok == 2: # PAD or EOS
            break
        tokens.append(next_tok)
        
    return tokenizer.decode(tokens[1:])

def run_trajectory_test():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p0_ckpt", required=True, help="Frozen Phase 0 weights (Fuser & Decoder)")
    parser.add_argument("--p1_ckpt", required=True, help="Trained Phase 1 weights (Mamba)")
    args = parser.parse_args()
    
    tokenizer = CharTokenizer()
    fuser, mamba, decoder, config = load_models(args.p0_ckpt, args.p1_ckpt)
    
    # Grab 1 batch from the dataset to get real continuous documents
    print("\nFetching a continuous document from the Parquet Dataset...")
    emb_files = [
        "data/Basic_ZH/embs/hy-tmp/roberta_embeddings.npy",
        "data/Basic_ZH/embs/hy-tmp/gte_embeddings.npy",
        "data/Basic_ZH/embs/hy-tmp/bge_embeddings.npy",
        "data/Basic_ZH/embs/hy-tmp/text2vec_embeddings.npy"
    ]
    dataloader = Phase1DataLoader("data/Basic_ZH/chunked_mixed_wiki.parquet", emb_files, batch_size=1, max_episode_len=256)
    
    # Get just the first document sequence
    batch_embs, masks = next(iter(dataloader))
    
    # 1. Forward pass thru Fuser to get semantic flow
    f_t = fuser(batch_embs, weights=None) # (1, L, d_model)
    
    # 2. Forward pass thru Mamba to map the physical trajectory
    mu, logvar, _ = mamba(f_t) # mu: (1, L, z_dim)
    
    valid_len = int(masks[0].sum().item())
    
    print("\n==================================================")
    print("TASK 1: SEMANTIC TRAJECTORY TRANSLATION")
    print("==================================================")
    print(f"Document Length: {valid_len} chunks.\nTranslating Mamba's spatial coordinates back to text...\n")
    
    for t in range(min(5, valid_len)):
        coord = mu[:, t, :] # (1, z_dim)
        text_recovery = decode_coordinate(decoder, coord, tokenizer)
        print(f"[Step {t+1}] Mamba Coordinate decoded as: {text_recovery}")
        
    print("\n==================================================")
    print("TASK 2: NEXT-STEP MOMENTUM PREDICTION")
    print("==================================================")
    print(f"Truncating document at Step 5. Forcing Mamba to predict Step 6 purely from momentum...")
    
    # Cut off sensory input at t=5
    f_t_cutoff = f_t[:, :5, :]
    
    # Feed 5 steps to establish hidden state momentum
    _, _, states = mamba(f_t_cutoff)
    last_state = [s for s in states] # Capture the RNN hidden state
    
    # Step 6: Total Sensory Deprivation (Zero Vector input)
    # Since it's blind, it MUST rely on the state momentum `last_state` to hallucinate a continuation
    blind_input = mx.zeros((1, 1, f_t.shape[-1]))
    mu_blind, _, _ = mamba.forward_step(blind_input, last_state)
    
    hallucinated_text = decode_coordinate(decoder, mu_blind[:, 0, :], tokenizer)
    print(f"[Blind Step 6] Mamba blindly extrapolated trajectory decoded as: {hallucinated_text}")

if __name__ == "__main__":
    run_trajectory_test()
