import argparse
import sys
import os
import torch
import math

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.mamba_planner_cuda import MambaPlanner
from model.flow_matcher_cuda import FlowMatcher
from training.core.char_tokenizer import CharTokenizer
from training.train_phase1_cuda import E2EModel
from distilled_emb.model_cuda import TinyCharEncoderCUDA, WeakDecoderCUDA

def load_e2e_model(ckpt_path, d_model=1024, mamba_d_state=16, mamba_d_conv=4, mamba_expand=2, flow_hidden_dim=2048, device="cuda"):
    mamba_planner = MambaPlanner(d_model=d_model, d_state=mamba_d_state, d_conv=mamba_d_conv, expand=mamba_expand)
    flow_matcher = FlowMatcher(d_model=d_model, hidden_dim=flow_hidden_dim)
    e2e_model = E2EModel(mamba_planner, flow_matcher).to(device)
    
    e2e_model.load_state_dict(torch.load(os.path.join(ckpt_path, "e2e_model.pt"), map_location=device, weights_only=True))
    e2e_model.eval()
    for param in e2e_model.parameters():
        param.requires_grad = False
    return e2e_model

def euler_ode_solve(flow_matcher, h_context, d_model=1024, num_steps=20, cfg_scale=3.0, device="cuda"):
    """
    Euler integration to generate z_next from standard Gaussian noise.
    """
    # 1. 抽取标准高斯白噪声作为起点
    x_t = torch.randn(1, 1, d_model, device=device, dtype=torch.float32)
    dt = 1.0 / num_steps
    
    # 2. 物理积分步进
    for i in range(num_steps):
        t_val = i / num_steps
        t_tensor = torch.full((1, 1, 1), t_val, device=device, dtype=torch.float32)
        
        # 喷嘴根据当前的坐标、时间、以及大脑给的背景势能，预测瞬时速度
        with torch.cuda.amp.autocast():
            v_pred = flow_matcher.predict_with_cfg(x_t, t_tensor, h_context, cfg_scale=cfg_scale)
        
        # 欧拉步进：沿着速度向量走一小段距离 dt
        x_t = x_t + v_pred.float() * dt
        
    # 最终的 x_1 就是我们在流形上生成的 z_next
    return x_t

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="在那片幽暗的森林深处，", help="Initial prompt text.")
    parser.add_argument("--phase0_encoder_ckpt", type=str, default="checkpoints/distilled/tinybert_pt_v1_step_100000")
    parser.add_argument("--phase0_decoder_ckpt", type=str, default="checkpoints/run/p0_v2_latest_emergency", help="Path to original MLX Phase 0 checkpoint containing the WeakDecoder")
    parser.add_argument("--phase1_ckpt", type=str, required=True, help="Path to E2E checkpoint directory")
    parser.add_argument("--num_sentences", type=int, default=5, help="How many sentences to generate.")
    parser.add_argument("--ode_steps", type=int, default=25, help="Number of Euler integration steps.")
    parser.add_argument("--cfg_scale", type=float, default=3.0, help="Classifier-Free Guidance scale.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Decoder sampling temperature.")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    tokenizer = CharTokenizer()
    
    print("Loading Phase 0.5 Encoder and Phase 0 Decoder...")
    # Load Encoder
    tiny_encoder = TinyCharEncoderCUDA(vocab_size=tokenizer.vocab_size, z_dim=1024)
    # 1. Load Phase 0.5 Encoder (PyTorch)
    enc_ckpt_path = args.phase0_encoder_ckpt
    if os.path.exists(f"{enc_ckpt_path}/student.pt"):
        enc_weights = torch.load(f"{enc_ckpt_path}/student.pt", map_location='cpu', weights_only=True)
    else:
        from safetensors.torch import load_file
        enc_weights = load_file(f"{enc_ckpt_path}/model.safetensors")
        
    tiny_encoder.load_state_dict(enc_weights, strict=False)
    tiny_encoder.to(device).eval()

    # 2. 🚀 Load Phase 0 Decoder (MLX safetensors bridging)
    from distilled_emb.model_cuda import load_mlx_safetensors_into_torch
    from safetensors.torch import load_file
    
    dec_ckpt_path = os.path.join(args.phase0_decoder_ckpt, "weak_decoder.safetensors")
    if not os.path.exists(dec_ckpt_path):
        # Fallback just in case
        dec_ckpt_path = os.path.join(args.phase0_decoder_ckpt, "model.safetensors")
        
    dec_state_dict = load_file(dec_ckpt_path)
    
    # Dynamically infer d_model from the projection layer weight shape (d_model, z_dim)
    # The key might be 'z_proj.weight'
    decoder_d_model = 128 # fallback
    if 'z_proj.weight' in dec_state_dict:
        decoder_d_model = dec_state_dict['z_proj.weight'].shape[0]
    elif 'out_proj.weight' in dec_state_dict:
        decoder_d_model = dec_state_dict['out_proj.weight'].shape[1]
        
    print(f"Dynamically inferred Decoder d_model: {decoder_d_model}")
    
    # Load Decoder with inferred d_model
    weak_decoder = WeakDecoderCUDA(z_dim=1024, vocab_size=tokenizer.vocab_size, d_model=decoder_d_model, n_layers=2)
    load_mlx_safetensors_into_torch(weak_decoder, dec_ckpt_path)
    # Force float32 to prevent LayerNorm overflow when Z is multiplied by 32
    weak_decoder = weak_decoder.to(torch.float32).to(device).eval()
    
    print(f"Loading Phase 1 E2E Model from {args.phase1_ckpt}...")
    e2e_model = load_e2e_model(args.phase1_ckpt, device=device)
    
    print(f"\n--- [ PROMPT ] ---\n{args.prompt}")
    
    # 1. 把引子文本变成特征空间的算子
    input_ids = tokenizer.encode(args.prompt, add_special_tokens=True)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device) # (1, L)
    mask_tensor = torch.ones_like(input_tensor, dtype=torch.float32)
    
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            z_prompt = tiny_encoder(input_tensor, attention_mask=mask_tensor) # (1, 1024)
        
        # (这里不需要缩放，因为 Phase 1 训练时就是用的原始 z_truth)
        z_scaled = z_prompt
        
        # 放入 Mamba 的记忆流里
        z_stream = z_scaled.unsqueeze(1) # (1, S=1, 1024)
        
        for step in range(args.num_sentences):
            # 2. Mamba 大脑积分：根据历史流，推理出此刻的势能
            with torch.cuda.amp.autocast():
                h_context_full = e2e_model.mamba(z_stream, aux_streams=[])
            h_context_last = h_context_full[:, -1:, :].float() # (1, 1, 1024) 转回 float32 保证 ODE 精度
            
            # 3. Flow Matcher 动力学求解：根据势能场，推演生成下一个算子
            z_next_scaled = euler_ode_solve(
                e2e_model.flow, 
                h_context_last, 
                d_model=1024, 
                num_steps=args.ode_steps, 
                cfg_scale=args.cfg_scale, 
                device=device
            )
            
            # 4. 把新生成的算子接回记忆流，为下一轮生成做准备
            z_stream = torch.cat([z_stream, z_next_scaled], dim=1) # (1, S+1, 1024)
            
            # 5. 因为 WeakDecoder 没有位置编码，很容易陷入马尔可夫死循环
            # 我们在这里强行把 z_next 放大 32 倍，利用 LayerNorm 冲刷掉历史 Token 的注意力，
            # 迫使 Decoder 退化为纯粹的“词袋采样器 (BoW Sampler)”，以此来观察 Mamba 真正生成的语义词汇！
            z_next_unscaled = z_next_scaled.squeeze(1) * math.sqrt(1024.0) # 放大 32 倍送给 Decoder
            
            gen_ids = weak_decoder.generate(
                z_target=z_next_unscaled,
                start_token=tokenizer.bos_token_id,
                eos_token=tokenizer.eos_token_id, 
                max_tokens=64,
                temperature=args.temperature
            )
            
            # decode 默认有 skip_special_tokens=True，会自动过滤 <BOS> <EOS> <PAD>
            text = tokenizer.decode(gen_ids).strip()
            
            print(f"\n--- [ CHUNK {step+1} ] ---\n{text}")

if __name__ == "__main__":
    main()
