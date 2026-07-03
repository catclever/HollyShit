import os
import sys
import torch
import argparse

# Ensure parent directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from training.core.char_tokenizer import CharTokenizer
from model.config import WeakDecoderConfig
from distilled_emb.model_cuda import TinyCharEncoderCUDA, WeakDecoderCUDA, load_mlx_safetensors_into_torch

def verify_tinybert_with_decoder(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"1. Loading architecture on [{device}]...")
    
    tokenizer = CharTokenizer()
    config = WeakDecoderConfig()
    d_model = config.decoder_heads * 64
    
    # Instantiate models
    # Z_dim is configured in WeakDecoderConfig (default 1024)
    decoder = WeakDecoderCUDA(config.z_dim, config.vocab_size, d_model=d_model, n_layers=config.decoder_layers).to(device)
    
    # 2. Load Phase 0 Decoder Weights
    print(f"2. Loading WeakDecoder weights from Phase 0: {args.p0_ckpt}...")
    decoder_path = f"{args.p0_ckpt}/decoder.safetensors"
    if os.path.exists(decoder_path):
        load_mlx_safetensors_into_torch(decoder, decoder_path)
    else:
        raise FileNotFoundError(f"Could not find decoder safetensors at {decoder_path}")
        
    # 3. Load Phase 0.5 TinyCharEncoder Weights
    print(f"3. Loading TinyCharEncoder weights from Phase 0.5: {args.tinybert_ckpt}...")
    if args.tinybert_ckpt.endswith(".safetensors"):
        from safetensors.torch import load_file
        weights = load_file(args.tinybert_ckpt)
    else:
        weights = torch.load(args.tinybert_ckpt, map_location='cpu', weights_only=True)
        
    sniffed_x_dim = weights['tok_emb.weight'].shape[1]
    sniffed_z_dim = weights['out_proj.weight'].shape[0]
    
    encoder = TinyCharEncoderCUDA(vocab_size=tokenizer.vocab_size, d_model=sniffed_x_dim, z_dim=sniffed_z_dim).to(device)
    encoder.load_state_dict(weights, strict=False)
    encoder.eval()
    
    print("\n" + "="*50)
    print("🚀 [Diagnostic Test] TinyCharEncoder -> WeakDecoder")
    print("="*50)
    
    prompts = [
        "你好",
        "你是猪么",
        "随便你想测试的句子放这里",
        "这是一个用来测试蒸馏质量的长句子，看看解码器能不能完美还原。"
    ]
    
    if args.prompt:
        prompts = [args.prompt]
        
    start_token = tokenizer.bos_token_id
    eos_token = tokenizer.eos_token_id
    
    for text in prompts:
        print(f"\n[Input Text] : {text}")
        
        # Encode
        ids = tokenizer.encode(text, add_special_tokens=False)
        # Pad to max_seq_len (e.g., 64) to match training behavior
        pad_len = 64 - len(ids)
        if pad_len < 0:
            ids = ids[:64]
            pad_len = 0
            
        padded_ids = ids + [tokenizer.pad_token_id] * pad_len
        mask_list = [1] * len(ids) + [0] * pad_len
        
        input_ids = torch.tensor([padded_ids], dtype=torch.long, device=device)
        mask = torch.tensor([mask_list], dtype=torch.float32, device=device)
        
        with torch.no_grad():
            z_macro = encoder(input_ids, attention_mask=mask)
            
            # Decode using WeakDecoder
            generated_ids = decoder.generate(
                z_macro, 
                start_token=start_token, 
                eos_token=eos_token, 
                max_tokens=64, 
                temperature=0.0 # Greedy decoding for exact match check
            )
            
        decoded_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        print(f"[Decoded]    : {decoded_text}")

    print("\n" + "="*50)
    print("Diagnostic Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify TinyCharEncoder's Z_macro with Phase 0 WeakDecoder")
    parser.add_argument("--tinybert_ckpt", type=str, required=True, help="Path to TinyCharEncoder weights")
    parser.add_argument("--p0_ckpt", type=str, required=True, help="Path to Phase 0 checkpoint directory (contains decoder.safetensors)")
    parser.add_argument("--prompt", type=str, default="", help="Optional specific text to test")
    
    args = parser.parse_args()
    verify_tinybert_with_decoder(args)
