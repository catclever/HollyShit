import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.core.char_tokenizer import CharTokenizer
from model.phase1_charencoder_cuda import TinyCharEncoderCUDA
from model.flow_unet_cuda import FlowMatching1DUnet
from model.continuous_translator_cuda import ContinuousToTextTranslatorCUDA
from model.hippo_oracle import HiPPOOracleCUDA

def find_latest_ckpt(base_dir="checkpoints/run", prefix="phase4_hippo_flow_step_"):
    if not os.path.exists(base_dir):
        return None
    candidates = []
    for d in os.listdir(base_dir):
        if d.startswith(prefix):
            full_dir = os.path.join(base_dir, d)
            pt_file = os.path.join(full_dir, "unet.pt")
            if os.path.isfile(pt_file):
                try:
                    step = int(d.split("step_")[-1])
                    candidates.append((step, pt_file))
                except ValueError:
                    pass
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
    return None

def find_latest_translator_ckpt(base_dir="checkpoints/run"):
    if not os.path.exists(base_dir):
        return None
    candidates = []
    for d in os.listdir(base_dir):
        if "translator_conv1d_step_" in d:
            full_dir = os.path.join(base_dir, d)
            pt_file = os.path.join(full_dir, "translator.pt")
            if os.path.isfile(pt_file):
                try:
                    step = int(d.split("step_")[-1])
                    candidates.append((step, pt_file))
                except ValueError:
                    pass
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
    return None

@torch.no_grad()
def generate_response(
    prompt: str,
    encoder: TinyCharEncoderCUDA,
    unet: FlowMatching1DUnet,
    translator: ContinuousToTextTranslatorCUDA,
    hippo_oracle: HiPPOOracleCUDA,
    tokenizer: CharTokenizer,
    seq_len: int = 256,
    steps: int = 20,
    device: str = "cuda"
) -> str:
    # 1. Encode prompt
    p_ids = tokenizer.encode(prompt, add_special_tokens=True)
    if len(p_ids) > seq_len:
        p_ids = p_ids[:seq_len]
    else:
        p_ids = p_ids + [tokenizer.pad_token_id] * (seq_len - len(p_ids))
        
    w_t = torch.tensor([p_ids], dtype=torch.long, device=device) # (1, L)
    
    h_t = encoder.get_normalized_embeddings(w_t)
    for layer in encoder.layers:
        h_t = layer(h_t, None)
    x_prompt = encoder.final_ln(h_t) # (1, L, 1024)
    x_prompt_norm = unet.normalize(x_prompt)
    
    # Extract prompt potential curve via HiPPO in normalized space
    h_potential = hippo_oracle(x_prompt_norm) # (1, L, 1024)
    
    # 2. Sample from Gaussian White Noise via Euler ODE
    x = torch.randn((1, seq_len, 1024), device=device)
    dt = 1.0 / steps
    
    for i in range(steps):
        t_val = i / float(steps)
        t_tensor = torch.tensor([t_val], device=device, dtype=torch.float32)
        v_pred = unet(x, t_tensor, h_potential=h_potential, k_attractors=x_prompt_norm)
        x = x + v_pred * dt
        
    # 3. Auto-denormalize back to encoder manifold and decode continuous curve to text
    x_denorm = unet.denormalize(x)
    token_ids = translator.decode_greedy(x_denorm.float())[0].tolist()
    text = tokenizer.decode(token_ids)
    
    # Cut off at <eom> or <eos> if present
    for stop_token in ["<eom>", "<eos>", "<pad>"]:
        if stop_token in text:
            text = text.split(stop_token)[0]
            
    return text.strip()

def main():
    parser = argparse.ArgumentParser(description="Phase 4 Dialogue Inference (Flow Matching from Noise)")
    parser.add_argument("--prompt", type=str, default=None, help="Input prompt text. If None, enters interactive CLI mode.")
    parser.add_argument("--unet_ckpt", type=str, default=None, help="Path to trained Unet checkpoint (unet.pt)")
    parser.add_argument("--encoder_ckpt", type=str, default="checkpoints/basencoder_step_115000/student.pt")
    parser.add_argument("--translator_ckpt", type=str, default=None)
    parser.add_argument("--steps", type=int, default=20, help="Number of Euler ODE integration steps")
    parser.add_argument("--seq_len", type=int, default=256)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    tokenizer = CharTokenizer()
    
    # Load Encoder
    encoder = TinyCharEncoderCUDA(vocab_size=tokenizer.vocab_size).to(device)
    if os.path.exists(args.encoder_ckpt):
        encoder.load_state_dict(torch.load(args.encoder_ckpt, map_location=device, weights_only=True))
        print(f"✅ Loaded frozen encoder from {args.encoder_ckpt}")
    else:
        print(f"⚠️ Warning: Encoder checkpoint not found at {args.encoder_ckpt}!")
    encoder.eval()
    
    # Load Translator Lens
    trans_path = args.translator_ckpt or find_latest_translator_ckpt()
    if not trans_path or not os.path.exists(trans_path):
        print(f"❌ Error: Translator checkpoint not found! Please train or specify --translator_ckpt")
        sys.exit(1)
    translator = ContinuousToTextTranslatorCUDA(d_model=1024, vocab_size=tokenizer.vocab_size).to(device)
    translator.load_state_dict(torch.load(trans_path, map_location=device, weights_only=True))
    translator.eval()
    print(f"✅ Loaded frozen translator from {trans_path}")
    
    # Load HiPPO
    hippo_oracle = HiPPOOracleCUDA(hidden_dim=1024, N=64).to(device)
    hippo_oracle.eval()
    
    # Load Unet
    unet_path = args.unet_ckpt or find_latest_ckpt()
    if not unet_path or not os.path.exists(unet_path):
        print(f"❌ Error: Unet checkpoint not found! Please train or specify --unet_ckpt")
        sys.exit(1)
    unet = FlowMatching1DUnet(hidden_dim=1024, cond_dim=1024, num_layers=6).to(device)
    unet.load_state_dict(torch.load(unet_path, map_location=device, weights_only=True))
    unet.eval()
    print(f"✅ Loaded Flow Match Unet from {unet_path}")
    
    print("\n" + "="*50)
    print("✨ Phase 4 Continuous Flow Dialogue Engine Ready!")
    print("="*50 + "\n")
    
    if args.prompt:
        resp = generate_response(
            args.prompt, encoder, unet, translator, hippo_oracle, tokenizer,
            seq_len=args.seq_len, steps=args.steps, device=device
        )
        print(f"Prompt:   {args.prompt}")
        print(f"Response: {resp}")
    else:
        print("Entering interactive mode. Type 'quit' or 'exit' to exit.")
        while True:
            try:
                user_input = input("\nUser > ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ["quit", "exit"]:
                    break
                resp = generate_response(
                    user_input, encoder, unet, translator, hippo_oracle, tokenizer,
                    seq_len=args.seq_len, steps=args.steps, device=device
                )
                print(f"Model > {resp}")
            except (KeyboardInterrupt, EOFError):
                break

if __name__ == "__main__":
    main()
