import os
import sys
import argparse
import mlx.core as mx

# Ensure parent directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from training.core.char_tokenizer import CharTokenizer
from model.config import WeakDecoderConfig
from distilled_emb.model import TinyCharEncoder
from model.decoder import WeakDecoder

def load_safetensors_into_mlx(model, safetensors_path):
    import mlx.core as mx
    import mlx.utils as mx_utils
    # Using mlx built-in safetensors loader
    weights = mx.load(safetensors_path)
    
    # TinyCharEncoder MLX uses "layers" instead of "net.layers" in the GodEncoder,
    # but since this is loading either Decoder or TinyCharEncoder, we just load directly.
    # Note: MLX safetensors dict might have different layer naming conventions from PyTorch,
    # but the ones saved in MLX format (like decoder.safetensors) match exactly.
    # If the user downloaded the PyTorch safetensors of TinyCharEncoder, we might need mapping,
    # but we assume they have the MLX format or the names map correctly.
    
    # Try to load
    model.load_weights(list(weights.items()))
    mx.eval(model.parameters())
    return model

def verify_tinybert_with_decoder_mlx(args):
    print("1. Loading MLX Architecture...")
    
    tokenizer = CharTokenizer()
    config = WeakDecoderConfig()
    d_model = config.decoder_heads * 64
    
    # Instantiate models
    decoder = WeakDecoder(config.z_dim, config.vocab_size, d_model=d_model, n_layers=config.decoder_layers)
    
    print(f"2. Loading WeakDecoder weights from Phase 0: {args.p0_ckpt}...")
    decoder_path = f"{args.p0_ckpt}/decoder.safetensors"
    if os.path.exists(decoder_path):
        decoder.load_weights(decoder_path)
        mx.eval(decoder.parameters())
    else:
        raise FileNotFoundError(f"Could not find decoder safetensors at {decoder_path}")
        
    print(f"3. Loading TinyCharEncoder weights from Phase 0.5: {args.tinybert_ckpt}...")
    
    # We need to instantiate TinyCharEncoder with sniffing or default params
    # Assuming default from distilled_emb/model.py
    encoder = TinyCharEncoder(vocab_size=tokenizer.vocab_size, z_dim=config.z_dim)
    
    if os.path.exists(args.tinybert_ckpt):
        try:
            encoder.load_weights(args.tinybert_ckpt)
            mx.eval(encoder.parameters())
        except Exception as e:
            # Handle PyTorch to MLX translation if they downloaded the torch checkpoint
            print(f"Failed to load weights directly, attempting PyTorch mapping: {e}")
            import torch
            if args.tinybert_ckpt.endswith(".safetensors"):
                from safetensors.torch import load_file
                pt_weights = load_file(args.tinybert_ckpt)
            else:
                pt_weights = torch.load(args.tinybert_ckpt, map_location='cpu', weights_only=True)
            
            mlx_weights = {}
            for k, v in pt_weights.items():
                if "q_proj" in k or "k_proj" in k or "v_proj" in k or "out_proj" in k or "ff" in k or "fc" in k:
                    if "weight" in k:
                        # MLX Linear weights are stored transposed compared to PyTorch occasionally,
                        # BUT actually mlx.nn.Linear expects (out_features, in_features) the same as PyTorch when loading from dict!
                        pass
                mlx_weights[k] = mx.array(v.numpy())
            
            encoder.update(mlx_weights)
            mx.eval(encoder.parameters())
    else:
         raise FileNotFoundError(f"Could not find TinyCharEncoder safetensors at {args.tinybert_ckpt}")
    
    print("\n" + "="*50)
    print("🚀 [MLX Diagnostic Test] TinyCharEncoder -> WeakDecoder")
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
        pad_len = 64 - len(ids)
        if pad_len < 0:
            ids = ids[:64]
            pad_len = 0
            
        padded_ids = ids + [tokenizer.pad_token_id] * pad_len
        mask_list = [1] * len(ids) + [0] * pad_len
        
        input_ids = mx.array([padded_ids], dtype=mx.int32)
        mask = mx.array([mask_list], dtype=mx.float32)
        
        z_macro = encoder(input_ids, attention_mask=mask)
        
        # Decode using WeakDecoder
        generated_ids = decoder.generate(
            z_macro, 
            start_token=start_token, 
            eos_token=eos_token, 
            max_tokens=64, 
            temperature=0.0 # Greedy decoding
        )
            
        decoded_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        print(f"[Decoded]    : {decoded_text}")

    print("\n" + "="*50)
    print("Diagnostic Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify TinyCharEncoder's Z_macro with Phase 0 WeakDecoder (MLX Version)")
    parser.add_argument("--tinybert_ckpt", type=str, required=True, help="Path to TinyCharEncoder weights (safetensors or pt)")
    parser.add_argument("--p0_ckpt", type=str, required=True, help="Path to Phase 0 checkpoint directory (contains decoder.safetensors)")
    parser.add_argument("--prompt", type=str, default="", help="Optional specific text to test")
    
    args = parser.parse_args()
    verify_tinybert_with_decoder_mlx(args)
