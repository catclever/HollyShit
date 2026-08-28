import os
import mlx.core as mx
import numpy as np
from training.core.char_tokenizer import CharTokenizer
from model.phase1_charencoder_mlx import TinyCharEncoderMLX, CharEncoderConfig
from model.depthwise_conv_decoder_mlx import ParallelDepthwiseConvDecoder, DepthwiseConvDecoderConfig
from tools.weight_utils import load_torch_weights_into_mlx

def add_noise(c, noise_min, noise_max):
    if noise_max <= 0.0:
        return c
    noise_scale = mx.random.uniform(
        shape=(c.shape[0], 1, 1),
        low=noise_min,
        high=noise_max
    )
    noise = mx.random.normal(shape=c.shape) * noise_scale
    return c + noise

def main():
    tokenizer = CharTokenizer()
    
    # Load Models
    encoder = TinyCharEncoderMLX(vocab_size=tokenizer.vocab_size)
    
    decoder_config = DepthwiseConvDecoderConfig(vocab_size=tokenizer.vocab_size, embed_dim=1024, hidden_dim=1024, num_layers=4, kernel_size=7)
    decoder = ParallelDepthwiseConvDecoder(decoder_config)
    
    # Load weights
    load_torch_weights_into_mlx(encoder, "checkpoints/basencoder_step_115000/student.pt")
    decoder.load_weights("checkpoints/phase2_2_conv/depthwise_pinned_10117/decoder.safetensors")
    
    encoder.eval()
    decoder.eval()
    
    test_cases = [
        # 基础短句
        "主人，请下达惩罚指令。",
        # 难词与成语 (魑魅魍魉 may be missing from vocab, we'll see)
        "深思熟虑，魑魅魍魉，高瞻远瞩，刻舟求剑。",
        # 嵌套的长逻辑句
        "如果在训练过程中加入高斯噪声，解码器就会被迫学习几何流形的平滑特征，而不是死记硬背具体的坐标。",
        # 文言文/诗词
        "落霞与孤鹜齐飞，秋水共长天一色。渔舟唱晚，响穷彭蠡之滨；雁阵惊寒，声断衡阳之浦。",
        # 包含数字和标点符号的混合文本
        "2026年8月27日，测试版本V2.2.2终于锁定了权重！成功率高达99.9%！"
    ]
    
    noise_levels = [0.0, 0.1, 0.2, 0.3]
    
    print(f"Vocab size: {tokenizer.vocab_size}")
    print("=" * 60)
    
    for text in test_cases:
        print(f"\n[TARGET] {text}")
        
        # Tokenize
        tokens = tokenizer.encode(text)
        mx_t = mx.array([tokens])
        
        # Get ground truth tokens
        orig_tokens = [i.item() for i in np.array(mx_t[0])]
        
        # Encode (Pure)
        h = encoder.get_normalized_embeddings(mx_t)
        for layer in encoder.layers:
            h = layer(h, None)
        c_pure = encoder.final_ln(h)
        
        # Decode at different noise levels
        for noise in noise_levels:
            if noise > 0:
                c_noisy = add_noise(c_pure, noise, noise)
                gen_ids = decoder.generate(c_noisy, temperature=0.0)
            else:
                gen_ids = decoder.generate(c_pure, temperature=0.0)
            
            gen_tokens = [i.item() for i in np.array(gen_ids[0])]
            
            # Truncate at EOS if it exists
            if tokenizer.eos_token_id in gen_tokens:
                gen_tokens = gen_tokens[:gen_tokens.index(tokenizer.eos_token_id)]
                
            gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
            print(f"  [Noise {noise:.1f}] -> {gen_text}")
            
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
