"""
【脚本功能】：Mamba 核心推演拓扑连贯性单元测试 (Trajectory Verification)
【使用场景】：核心组件测试。用于测试 `MambaPlanner` 的时序预测结构是否能够跑通 100 步的极限顺滑外推，并验证其坐标预测的物理方差属性。
【用法示例】：`python tests/test_mamba_trajectory.py`
"""
import os
import sys
import argparse
import numpy as np
import mlx.core as mx

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.config import ModelConfig
from model.adapter import SensoryFuser
from model.decoder import WeakDecoder
from model.god_encoder import GodEncoder
from model.mamba_mlx.mamba_mlx import MambaConfig
from model.mamba_planner import MambaPlanner
from training.core.char_tokenizer import CharTokenizer
from training.core.dataloader import Phase1DataLoader
import json

def load_models(p0_ckpt, p1_ckpt):
    config = ModelConfig()
    d_model = config.decoder_heads * 64
    
    # 1. Load Fuser and Decoder (From Phase 0)
    print(f"Loading Decoder, Fuser & GodEncoder strictly from Phase 0: {p0_ckpt}...")
    fuser = SensoryFuser(config.emb_dims, d_model)
    god_encoder = GodEncoder(d_model, config.z_dim)
    decoder = WeakDecoder(config.z_dim, config.vocab_size, d_model=d_model, n_layers=config.decoder_layers)
    
    fuser_path = f"{p0_ckpt}/sense_fuser.safetensors"
    if not os.path.exists(fuser_path):
        fuser_path = f"{p0_ckpt}/sense_adapter.safetensors"
    fuser.load_weights(fuser_path)
    god_encoder.load_weights(f"{p0_ckpt}/god_encoder.safetensors")
    decoder.load_weights(f"{p0_ckpt}/decoder.safetensors")
    
    # 2. Load Trajectory Planner (From Phase 1)
    print(f"Loading Mamba Planner strictly from Phase 1: {p1_ckpt}...")
    mamba_cfg = MambaConfig(d_model=d_model, n_layers=2)
    mamba = MambaPlanner(mamba_cfg, config.z_dim, residual_mode=False) 
    mamba.load_weights(f"{p1_ckpt}/mamba_planner.safetensors")
    
    fuser.eval()
    god_encoder.eval()
    decoder.eval()
    mamba.eval()
    
    return fuser, god_encoder, mamba, decoder, config

def decode_coordinate(decoder, mu_coord, tokenizer, max_len=64):
    """Autoregressively decodes a single spatial coordinate into text."""
    # mu_coord shape: (1, z_dim)
    BOS_TOKEN = tokenizer.bos_token_id
    tokens = [BOS_TOKEN]
    
    print(f"  [Decoding from mu] BOS={BOS_TOKEN}. Max Len={max_len}")
    for i in range(max_len):
        tok_arr = mx.array([tokens], dtype=mx.int32)
        # Decoder expects (z, tokens)
        logits = decoder(mu_coord, tok_arr)
        next_tok = mx.argmax(logits[:, -1, :], axis=-1).item()
        
        print(f"    Step {i}: predicted token ID {next_tok} -> '{tokenizer.decode([next_tok])}'")
        tokens.append(next_tok)
        
        if next_tok == tokenizer.pad_token_id or next_tok == tokenizer.eos_token_id:
            print(f"    (Hit PAD/EOS, but continuing to inspect raw outputs...)")
        
    return tokenizer.decode(tokens[1:])

def run_trajectory_test():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p0_ckpt", required=True, help="Frozen Phase 0 weights (Fuser & Decoder)")
    parser.add_argument("--p1_ckpt", required=True, help="Trained Phase 1 weights (Mamba)")
    args = parser.parse_args()
    
    tokenizer = CharTokenizer()
    fuser, god_encoder, mamba, decoder, config = load_models(args.p0_ckpt, args.p1_ckpt)
    
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
    
    # 1b. Absolute Truth Target Coordinates
    z_target = god_encoder(f_t)
    
    # 2. Forward pass thru Mamba to map the physical trajectory
    # Because Mamba is now autoregressive, mu[:, t, :] predicts the coordinate of step t+1!
    mu, logvar, _ = mamba(f_t) # mu: (1, L, z_dim)
    
    valid_len = int(masks[0].sum().item())
    
    print("\n==================================================")
    print("TASK 1: AUTOREGRESSIVE TENSOR TRANSLATION")
    print("==================================================")
    print(f"Document Length: {valid_len} chunks.\nTranslating Mamba's NEXT-STEP spatial coordinate predictions back to text...\n")
    
    for t in range(min(3, valid_len - 1)):
        print(f"\n--- Chunk {t+1} Predicting Chunk {t+2} ---")
        coord_truth = z_target[:, t+1, :] # The true future coordinate
        coord_mamba = mu[:, t, :] # Mamba's prediction of the future coordinate
        
        diff = mx.abs(coord_truth - coord_mamba).mean().item()
        print(f" GodEncoder Future Center = {coord_truth.mean().item():.5f} | Mamba Guessed Center = {coord_mamba.mean().item():.5f}")
        print(f" L1 Absolute Distance between targets = {diff:.5f}")
        
        print(f" GOD ENCODER (Truth) Decoding ->")
        decode_coordinate(decoder, coord_truth, tokenizer, max_len=10)
        print(f" MAMBA (Guessed) Decoding ->")
        decode_coordinate(decoder, coord_mamba, tokenizer, max_len=10)
        
    print("\n==================================================")
    print("TASK 2: PURE MOMENTUM EXTRAPOLATION")
    print("==================================================")
    print(f"Truncating real document sensory input at Chunk 5.")
    print(f"Feeding 'Silence' (Zeros) at Chunk 6. Forcing Mamba to blindly hallucinate Chunk 7 purely from momentum...")
    
    # Cut off sensory input at t=5 (Chunks 1 to 5)
    f_t_cutoff = f_t[:, :5, :]
    
    # Step 6: Total Sensory Deprivation (Zero Vector input)
    # Since it's blind, it MUST rely on the state momentum established in the first 5 steps
    blind_input = mx.zeros((1, 1, f_t.shape[-1]))
    extrapolated_f_t = mx.concatenate([f_t_cutoff, blind_input], axis=1) # Length 6
    
    mu_extrapolated, _, _ = mamba(extrapolated_f_t)
    
    # The output space for step 6 is at index 5. Because it's autoregressive, index 5 predicts Chunk 7.
    predicted_coord = mu_extrapolated[:, 5, :]
    
    print(f"\n [Blind Extrapolation] Decoding Mamba's hallucinated Chunk 7 ->")
    hallucinated_text = decode_coordinate(decoder, predicted_coord, tokenizer)
    print(f"[Blind Step 6] Mamba blindly extrapolated trajectory decoded as: {hallucinated_text}")

if __name__ == "__main__":
    run_trajectory_test()
