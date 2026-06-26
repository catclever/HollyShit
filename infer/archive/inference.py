# =========================================================================
# [ARCHIVED] LEGACY SCRIPT: End-to-End Inference (MLX)
#
# [Reason for Archival]:
# This was the complete end-to-end inference script for the original Phase 1 pipeline 
# on Apple Silicon (MLX). It chained together:
# Text -> TinyCharEncoder -> Z_truth -> Mamba -> Z_pred -> FlowMatcher -> Z_flow -> WeakDecoder
# Since the architecture is migrating away from Point-based Flow Matching to 
# Sequence Flow Matching, the middle components (FlowMatcher/WeakDecoder) are obsolete.
# =========================================================================

import os
import sys
import argparse
import re
import mlx.core as mx

# Ensure workspace root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from training.core.char_tokenizer import CharTokenizer
from distilled_emb.model import TinyCharEncoder
from model.mamba_planner import MambaConfig, MambaPlanner
# from model.flow_matcher import FlowMatcher
from model.archive.flow_matcher import FlowMatcher
from model.config import WeakDecoderConfig
from model.decoder import WeakDecoder
from training.core.checkpoint import Checkpointer

def split_sentences(text: str) -> list[str]:
    """Split text into sentences by punctuation."""
    # Split by Chinese and English punctuation
    parts = re.split(r'([。？！，\.,?!])', text)
    sentences = []
    current = ""
    for part in parts:
        current += part
        if part in "。？！，.,?!":
            sentences.append(current.strip())
            current = ""
    if current.strip():
        sentences.append(current.strip())
    return sentences

def run_inference(args):
    print("==================================================")
    print("      Physics-Engine Inference Shell (Phase 1)    ")
    print("==================================================")
    
    # 1. Initialize tokenizer
    tokenizer = CharTokenizer("training/core/char_vocab.json")
    print(f"[1/4] Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")
    
    # 2. Load Phase 0.5: TinyCharEncoder (Eyes)
    # Use sniffed z_dim
    weights = mx.load(args.tiny_ckpt)
    try:
        sniffed_z_dim = weights['out_proj.weight'].shape[0]
    except KeyError:
        raise ValueError("Cannot find 'out_proj.weight' in the checkpoint.")
        
    tiny_encoder = TinyCharEncoder(vocab_size=tokenizer.vocab_size, z_dim=sniffed_z_dim)
    tiny_encoder.load_weights(list(weights.items()))
    tiny_encoder.freeze()
    print(f"[2/4] TinyCharEncoder (Eyes) loaded from {args.tiny_ckpt}. Z_dim = {sniffed_z_dim}")
    
    # 3. Load Phase 1: MambaPlanner (Brain) & FlowMatcher (Motor Cortex)
    # Re-create config
    mamba_cfg = MambaConfig(
        d_model=1024, 
        n_layers=2,
        d_state=16,
        d_conv=4,
        expand_factor=2
    )
    mamba_planner = MambaPlanner(mamba_cfg)
    flow_matcher = FlowMatcher(d_model=1024, hidden_dim=2048)
    
    checkpointer = Checkpointer(args.phase1_ckpt)
    checkpointer.register_model("mamba_planner", mamba_planner)
    checkpointer.register_model("flow_matcher", flow_matcher)
    
    start_step = checkpointer.load_latest()
    if start_step == 0:
        print(f"WARNING: Could not load Phase 1 checkpoint from {args.phase1_ckpt}")
    else:
        print(f"[3/4] MambaPlanner & FlowMatcher loaded from step {start_step}.")
        
    mamba_planner.eval()
    flow_matcher.eval()
    
    # 4. Load Phase 0.5: WeakDecoder (Mouth)
    dec_weights = mx.load(args.decoder_ckpt)
    try:
        sniffed_d_model = dec_weights['z_proj.weight'].shape[0]
        layer_indices = set()
        for k in dec_weights.keys():
            if k.startswith("transformer.layers."):
                parts = k.split(".")
                if len(parts) > 2 and parts[2].isdigit():
                    layer_indices.add(int(parts[2]))
        sniffed_n_layers = max(layer_indices) + 1 if layer_indices else 4
        print(f"[Auto-Sniff] WeakDecoder dimensions inferred: d_model={sniffed_d_model}, n_layers={sniffed_n_layers}")
    except Exception as e:
        print(f"Warning: Could not sniff WeakDecoder dimensions ({e}). Falling back to 256 and 4.")
        sniffed_d_model = 256
        sniffed_n_layers = 4
        
    decoder = WeakDecoder(sniffed_z_dim, tokenizer.vocab_size, d_model=sniffed_d_model, n_layers=sniffed_n_layers)
    decoder.load_weights(args.decoder_ckpt)
    decoder.eval()
    print(f"[4/4] WeakDecoder (Mouth) loaded from {args.decoder_ckpt}.")
    
    print("==================================================\n")
    print("System Online. Entering Interactive Mode.")
    print("Type 'quit' or 'exit' to terminate.\n")
    
    # Pre-compile the ODE integration step for blazing fast inference
    def euler_step(x, t, h):
        v = flow_matcher(x, t, h)
        return v
    
    while True:
        try:
            prompt = input("\n[You] ")
            if prompt.strip().lower() in ['quit', 'exit']:
                break
            if not prompt.strip():
                continue
                
            sentences = split_sentences(prompt)
            if len(sentences) == 0:
                continue
                
            # A. Encode prompt into Z coordinates using TinyCharEncoder
            encoded_seqs = [tokenizer.encode(s, add_special_tokens=True)[:256] for s in sentences]
            max_len = max(len(seq) for seq in encoded_seqs)
            
            padded_ids = []
            masks = []
            for seq in encoded_seqs:
                pad_len = max_len - len(seq)
                padded_ids.append(seq + [tokenizer.pad_token_id] * pad_len)
                masks.append([1] * len(seq) + [0] * pad_len)
                
            ids_t = mx.array(padded_ids) # (S, L)
            att_t = mx.array(masks)      # (S, L)
            
            z_flat = tiny_encoder(ids_t, attention_mask=att_t) # (S, Z_dim)
            
            # Reshape to (Batch=1, S, Z_dim) for Mamba
            z_prompt = z_flat.reshape(1, len(sentences), -1) 
            
            # B. Get Context from Mamba Planner
            h_context = mamba_planner(z_prompt) # (1, S, d_model)
            h_last = h_context[:, -1:, :]       # (1, 1, d_model) Extract final temporal state
            
            # C. Euler Integration (ODE Solver) to predict Z_next
            N_STEPS = 20
            dt = 1.0 / N_STEPS
            x_t = mx.random.normal(shape=(1, 1, sniffed_z_dim))
            
            print("  [Physics Engine] Simulating trajectory...", end="", flush=True)
            for i in range(N_STEPS):
                # 必须保证维度是 (1, 1, 1)
                t_tensor = mx.array([float(i) * dt]).reshape(1, 1, 1)
                
                # Predict velocity
                v_pred = euler_step(x_t, t_tensor, h_last)
                
                # Update coordinate
                x_t = x_t + v_pred * dt
                
                mx.eval(x_t) # Force evaluation to prevent memory spike
            print(" Arrived at predicted coordinate.")
            
            z_pred = x_t # (1, 1, Z_dim)
            
            # D. Decode the physical coordinate back into human language
            z_target = z_pred.reshape(1, sniffed_z_dim)
            generated_ids = decoder.generate(
                z_target, 
                start_token=tokenizer.bos_token_id, 
                eos_token=tokenizer.eos_token_id, 
                max_tokens=60, 
                temperature=0.2
            )
            decoded_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            print(f"[Flow Dream] {decoded_text}")
            
        except KeyboardInterrupt:
            print("\nType 'quit' to exit.")
            continue
        except Exception as e:
            print(f"\n[Error] {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiny_ckpt", type=str, default="checkpoints/distilled/distilled_retina_mlx.safetensors")
    parser.add_argument("--phase1_ckpt", type=str, default="checkpoints/run/phase1_mamba_flow")
    parser.add_argument("--decoder_ckpt", type=str, default="checkpoints/run/p0_v2_step_202913/weak_decoder.safetensors")
    args = parser.parse_args()
    
    run_inference(args)
