import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import mlx.core as mx
from training.core.char_tokenizer import CharTokenizer
from model.phase1_6_latedecoder import LateFusionDecoder

tokenizer = CharTokenizer()

# 1. Load model with local attention window_size=15
print("Loading Model...")
model = LateFusionDecoder(
    vocab_size=tokenizer.vocab_size,
    embed_dim=1024,
    hidden_dim=1024,
    num_layers=8,
    num_heads=16,
    window_size=15
)
model.load_weights("checkpoints/phase1_2_decoder/short_grammar_step_32594/decoder.safetensors")
mx.eval(model.parameters())
print("Model loaded successfully!")

def generate_text(prompt, max_len=100, temperature=0.7):
    # Dummy mock_z since we are evaluating the pure decoder
    prompt_ids = tokenizer.encode(prompt)
    
    # We must provide mock_z for all tokens up to max_len
    # Actually, model.generate expects sample_c which is mock_z
    mock_z = mx.zeros((1, max_len + len(prompt_ids), 1024))
    
    gen_ids = model.generate(
        c_zoh=mock_z,
        bos_token=tokenizer.bos_token_id,
        max_len=max_len,
        temperature=temperature,
        prompt_ids=prompt_ids
    )
    return tokenizer.decode(gen_ids, skip_special_tokens=True)

test_prompts = [
    "爱因斯坦是",
    "苹果公司由",
    "今天天气真好，",
    "你觉得我穿这件衣服",
    "因为天下雨了，所以我"
]

print("\n--- Phase 1.2 Local Attention (window_size=15) Evaluation ---")
for prompt in test_prompts:
    print(f"\n[Prompt]: {prompt}")
    gen = generate_text(prompt, max_len=50, temperature=0.7)
    print(f"[Generated]: {gen}")
    print("-" * 40)
