import os
import mlx.core as mx
import numpy as np
from transformers import AutoTokenizer
from model.phase1_charencoder_mlx import TinyCharEncoderMLX, CharEncoderConfig
from model.depthwise_conv_decoder_mlx import ParallelDepthwiseConvDecoder, DepthwiseConvDecoderConfig

def main():
    from training.core.char_tokenizer import CharTokenizer
    tokenizer = CharTokenizer()
    
    # Load Models
    encoder_config = CharEncoderConfig(vocab_size=tokenizer.vocab_size, d_model=1024, n_heads=8, n_layers=2)
    encoder = TinyCharEncoderMLX(encoder_config)
    
    decoder_config = DepthwiseConvDecoderConfig(vocab_size=tokenizer.vocab_size, embed_dim=1024, hidden_dim=1024, num_layers=4, kernel_size=7)
    decoder = ParallelDepthwiseConvDecoder(decoder_config)
    
    # Load weights
    from tools.weight_utils import load_torch_weights_into_mlx
    load_torch_weights_into_mlx(encoder, "checkpoints/basencoder_step_115000/student.pt")
    decoder.load_weights("checkpoints/phase2_2_conv/depthwise_pinned_10117/decoder.safetensors")
    
    encoder.eval()
    decoder.eval()
    
    test_cases = [
        "普通测试：小安是一只乖巧的猫咪。",
        "专业名词测试：在深度可分离卷积（Depthwise Separable Convolution）中，逐点卷积用于跨通道信息交融。",
        "代码测试：def calculate_loss(logits, targets):\n    return mx.mean(nn.losses.cross_entropy(logits, targets))",
        "多语言测试：The quick brown fox jumps over the lazy dog. 吾輩は猫である。名前はまだ無い。"
    ]
    
    for text in test_cases:
        tokens = tokenizer.encode(text)
        mx_t = mx.array([tokens])
        
        # Phase 1 Encode (Pure, no noise)
        h = encoder.get_normalized_embeddings(mx_t)
        for layer in encoder.layers:
            h = layer(h, None)
        c = encoder.final_ln(h)
        
        # Phase 2.2 Decode
        gen_ids = decoder.generate(c, temperature=0.0)
        
        gen_tokens = [i.item() for i in np.array(gen_ids[0])]
        gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        
        print("-" * 50)
        print(f"Target : {text}")
        print(f"Decoded: {gen_text}")
        print("-" * 50)

if __name__ == "__main__":
    main()
