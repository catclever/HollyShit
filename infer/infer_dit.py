import os
import sys
import argparse
import torch
import json
import torch.nn.functional as F

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from distilled_emb.model_cuda import TinyCharEncoderCUDA
from model.pre_phase1_dit_cuda import PrePhase1DiT
from training.core.char_tokenizer import CharTokenizer

def decode_embeddings(x_pred, tok_emb, tokenizer):
    """
    Finds the closest token in the vocabulary for each embedding vector.
    """
    # x_pred: (L, D)
    # tok_emb.weight: (vocab_size, D)
    
    # Cosine similarity is usually robust for high dimensional spaces
    x_norm = F.normalize(x_pred, p=2, dim=-1)
    emb_norm = F.normalize(tok_emb.weight, p=2, dim=-1)
    
    # (L, D) @ (D, vocab_size) -> (L, vocab_size)
    sims = torch.matmul(x_norm, emb_norm.transpose(0, 1))
    
    token_ids = torch.argmax(sims, dim=-1).tolist()
    return tokenizer.decode(token_ids)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tinybert_ckpt", type=str, required=True, help="Path to frozen student ckpt (e.g. true_god_step_78000)")
    parser.add_argument("--dit_ckpt", type=str, required=True, help="Path to DiT checkpoint (e.g. dit_v1_step_37000)")
    parser.add_argument("--prompt", type=str, default="今天天气真不错，我们一起去", help="Text to encode into Z")
    parser.add_argument("--steps", type=int, default=50, help="Number of Euler steps for ODE solver")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = args.device
    tokenizer = CharTokenizer()
    
    # 1. Load scale factor from args
    args_path = os.path.join(os.path.dirname(args.dit_ckpt), "training_args.json")
    scale_factor = 1.0
    if os.path.exists(args_path):
        with open(args_path, 'r') as f:
            targs = json.load(f)
            scale_factor = targs.get("scale_factor", 1.0)
            print(f"✅ Loaded scale_factor: {scale_factor:.4f}")
    
    # 2. Load TinyCharEncoder
    print(f"🔥 Loading Student Encoder: {args.tinybert_ckpt}")
    tiny_args_path = os.path.join(os.path.dirname(args.tinybert_ckpt), "training_args.json")
    student_d_model, student_z_dim = 1024, 1024
    if os.path.exists(tiny_args_path):
        with open(tiny_args_path, 'r') as f:
            t_tiny = json.load(f)
            student_d_model = t_tiny.get("student_d_model", 1024)
            student_z_dim = t_tiny.get("z_dim", 1024)
            
    tiny_encoder = TinyCharEncoderCUDA(vocab_size=tokenizer.vocab_size, d_model=student_d_model, z_dim=student_z_dim).to(device)
    from safetensors.torch import load_file
    student_safe = os.path.join(args.tinybert_ckpt, "student.safetensors")
    student_pt = os.path.join(args.tinybert_ckpt, "student.pt")
    if os.path.exists(student_safe):
        tiny_encoder.load_state_dict(load_file(student_safe), strict=False)
    elif os.path.exists(student_pt):
        tiny_encoder.load_state_dict(torch.load(student_pt, map_location="cpu"), strict=False)
    tiny_encoder.eval()
    
    # 3. Load DiT
    print(f"🔥 Loading DiT: {args.dit_ckpt}")
    dit = PrePhase1DiT(z_dim=student_z_dim, x_dim=student_d_model).to(device)
    dit_safe = os.path.join(args.dit_ckpt, "flow_matcher.safetensors")
    dit_pt = os.path.join(args.dit_ckpt, "flow_matcher.pt")
    if os.path.exists(dit_safe):
        dit.load_state_dict(load_file(dit_safe), strict=True)
    elif os.path.exists(dit_pt):
        dit.load_state_dict(torch.load(dit_pt, map_location="cpu"), strict=True)
    dit.eval()
    
    # 4. Extract Z
    print("\n" + "="*50)
    print(f"📝 Prompt: {args.prompt}")
    token_ids = tokenizer.encode(args.prompt, add_special_tokens=True)
    tokens = torch.tensor([token_ids], dtype=torch.long).to(device)
    
    with torch.no_grad():
        z_pred = tiny_encoder(tokens)
    
    L = len(token_ids)
    print(f"✨ Z-vector Extracted (Shape: {z_pred.shape}, Target Length: {L})")
    
    # 5. ODE Euler Solver
    print(f"🚀 Starting Flow Matching ODE Solver ({args.steps} steps)...")
    
    with torch.no_grad():
        x_t = torch.randn(1, L, student_d_model, device=device) # X_0 ~ N(0, 1)
        dt = 1.0 / args.steps
        
        for i in range(args.steps):
            t_val = i * dt
            t_tensor = torch.tensor([t_val], device=device)
            
            # Predict velocity
            v_pred = dit(x_t, t_tensor, z_pred)
            
            # Euler step
            x_t = x_t + v_pred * dt
            
            if (i+1) % 10 == 0 or i == args.steps - 1:
                print(f"   [Step {i+1:02d}/{args.steps}] t={t_val + dt:.2f}")

    # 6. Decode back to tokens
    x_1_pred = x_t.squeeze(0) # (L, D)
    
    # MUST downscale back to original embedding scale
    x_1_restored = x_1_pred / scale_factor
    
    result_text = decode_embeddings(x_1_restored, tiny_encoder.tok_emb, tokenizer)
    
    print("\n" + "="*50)
    print(f"🎯 Original:\n{args.prompt}")
    print(f"🧠 DiT Reconstruction:\n{result_text}")
    print("="*50)

if __name__ == "__main__":
    main()
