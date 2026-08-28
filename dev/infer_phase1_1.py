import os
import mlx.core as mx
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from training.core.char_tokenizer import CharTokenizer
from model.phase1_6_latedecoder import LateFusionDecoder

def main():
    tokenizer = CharTokenizer()
    
    model = LateFusionDecoder(
        vocab_size=tokenizer.vocab_size,
        embed_dim=1024,
        hidden_dim=1024,
        num_layers=8,
        num_heads=16
    )
    
    ckpt_path = "checkpoints/phase1_1_decoder/epoch_2.safetensors"
    print(f"Loading {ckpt_path}...")
    model.load_weights(ckpt_path)
    mx.eval(model.parameters())
    
    gen_len = 100
    dummy_encoder_out = mx.zeros((1, gen_len + 1, 1024))
    
    print("Generating with temperature 0.8...")
    tokens = model.generate(dummy_encoder_out, bos_token_id=tokenizer.bos_token_id, temperature=0.8)
    
    tokens_list = tokens[0].tolist()
    print("Raw tokens:", tokens_list)
    text = tokenizer.decode(tokens_list)
    
    print("\n--- Generated Text ---")
    print(text)
    print("----------------------\n")

if __name__ == '__main__':
    main()
