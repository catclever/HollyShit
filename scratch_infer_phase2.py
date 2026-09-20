import mlx.core as mx
import os
import json
from training.core.char_tokenizer import CharTokenizer
from model.phase1_charencoder_mlx import TinyCharEncoderMLX
from model.concept_chunker_mlx import ConceptChunkerMLX
from model.depthwise_conv_decoder_mlx import ParallelDepthwiseConvDecoder, DepthwiseConvDecoderConfig
from tools.weight_utils import load_torch_weights_into_mlx

def main():
    tokenizer = CharTokenizer()
    
    # 1. Load Encoder
    encoder = TinyCharEncoderMLX(vocab_size=tokenizer.vocab_size)
    load_torch_weights_into_mlx(encoder, "checkpoints/basencoder_step_115000/student.pt")
    
    # 2. Initialize Chunker
    chunker = ConceptChunkerMLX(threshold=0.5)
    
    # 3. Load Decoder Config & Model
    ckpt_dir = "checkpoints/phase2_2_conv/depthwise_pinned_27005"
    with open(os.path.join(ckpt_dir, "decoder_config.json"), "r") as f:
        config_dict = json.load(f)
    config = DepthwiseConvDecoderConfig(**config_dict)
    decoder = ParallelDepthwiseConvDecoder(config)
    
    # Load Decoder Weights
    import mlx.utils
    weights = mx.load(os.path.join(ckpt_dir, "decoder.safetensors"))
    weights_flat = list(weights.items())
    decoder.update(mlx.utils.tree_unflatten(weights_flat))
    mx.eval(decoder.parameters())
    
    # 4. Test Text
    text = "你的成绩出来了吗？是的，我的整个GPA现在被搞砸了。为什么？发生了什么事？"
    tokens = tokenizer.encode(text)
    
    w_t = mx.array([tokens]) # [1, SeqLen]
    
    # Encode
    h = encoder.get_normalized_embeddings(w_t)
    for layer in encoder.layers:
        h = layer(h, None)
    h = encoder.final_ln(h)
    
    # Chunk
    c = chunker(h) # [1, L, D]
    
    # Decode
    pred_tokens = decoder.generate(c, temperature=0.0)[0]
    
    pred_text = tokenizer.decode(pred_tokens.tolist())
    
    print("\n--- INFERENCE RESULT ---")
    print(f"Target: {text}")
    print(f"Output: {pred_text}")
    print("------------------------\n")

if __name__ == "__main__":
    main()
