import os
import mlx.core as mx
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from training.core.char_tokenizer import CharTokenizer
from model.latedecoder import LateFusionDecoder

def generate_with_prompt(model, tokenizer, prompt_text, max_len=100, temperature=0.8):
    prompt_ids = tokenizer.encode(prompt_text)
    current_tokens = [tokenizer.bos_token_id] + prompt_ids
    
    for t in range(max_len):
        x = mx.array(current_tokens)[None, :]
        seq_len = x.shape[1]
        
        embedded = model.char_embed(x) + model.pos_embed[:, :seq_len, :]
        mask = model._create_causal_mask(seq_len, dtype=embedded.dtype)
        
        h = embedded
        for block in model.blocks:
            h = block(h, mask)
            
        h_text = model.final_ln(h)
        h_t = h_text[:, -1:, :]
        
        z_t = mx.zeros((1, 1, 1024))
        fused = mx.concatenate([h_t, z_t], axis=-1)
        next_logits = model.fc(model.fusion_mlp(fused))[0, 0, :]
        
        next_token = mx.random.categorical(next_logits * (1.0 / temperature)).item()
        current_tokens.append(next_token)
        
    return tokenizer.decode(current_tokens)

def main():
    tokenizer = CharTokenizer()
    
    model = LateFusionDecoder(
        vocab_size=tokenizer.vocab_size,
        embed_dim=1024,
        hidden_dim=1024,
        num_layers=8,
        num_heads=16,
        window_size=None  # OPEN THE WINDOW (Global Attention)
    )
    
    ckpt_path = "checkpoints/phase1_2_decoder/partial_window_step_45000/decoder.safetensors"
    print(f"Loading {ckpt_path} (Window Size = FULL GLOBAL)...")
    model.load_weights(ckpt_path)
    mx.eval(model.parameters())
    
    prompts = [
        "PSR1你好，请问你是谁？PSR2",
        "在一个风和日丽的下午，",
        "我最喜欢的运动是篮球，因为",
        "人工智能的发展正在深刻地改变我们的生活。"
    ]
    
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        text = generate_with_prompt(model, tokenizer, prompt, max_len=50, temperature=0.8)
        print("--- Generated Text ---")
        print(text)
        print("----------------------")

if __name__ == '__main__':
    main()
