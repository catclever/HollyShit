import os
import sys
import argparse
import torch

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from model.phase1_charencoder_cuda import TinyCharEncoderCUDA, load_mlx_safetensors_into_torch
from model.decoder_cuda import WeakDecoderCUDA
from training.core.char_tokenizer import CharTokenizer
from model.config import WeakDecoderConfig
import json
import glob

def get_latest_student_ckpt(distilled_dirs):
    if isinstance(distilled_dirs, str):
        distilled_dirs = [distilled_dirs]
        
    for d in distilled_dirs:
        if not os.path.exists(d):
            continue
        ckpts = glob.glob(os.path.join(d, "*step_*"))
        if ckpts:
            return max(ckpts, key=os.path.getmtime)
            
    return None

def get_latest_teacher_ckpt(base_dirs):
    if isinstance(base_dirs, str):
        base_dirs = [base_dirs]
        
    for d in base_dirs:
        if not os.path.exists(d):
            continue
        # Look for any directory containing 'p0'
        ckpts = [os.path.join(d, item) for item in os.listdir(d) 
                 if os.path.isdir(os.path.join(d, item)) and 'p0' in item]
        if ckpts:
            return max(ckpts, key=os.path.getmtime)
            
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p0_ckpt", type=str, default="", help="Phase 0 MLX Teacher checkpoint containing weak_decoder.safetensors (if empty, finds the latest containing 'p0')")
    parser.add_argument("--ckpt_dir", type=str, default="", help="Path to student checkpoint dir (if empty, finds the latest in checkpoints/phase1_charencoder)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--prompt", type=str, default="今天天气真不错，我们一起去", help="The text to encode into Z")
    parser.add_argument("--split_by_punct", action="store_true", help="If set, splits the prompt by punctuation and processes each chunk separately to avoid single-vector bottleneck")
    parser.add_argument("--max_tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    device = args.device
    tokenizer = CharTokenizer()
    config = WeakDecoderConfig()

    ckpt_dir = args.ckpt_dir
    if not ckpt_dir:
        ckpt_dir = get_latest_student_ckpt([
            "checkpoints/phase1", 
            "checkpoints/phase1_charencoder", 
            "checkpoints/distilled"
        ])
        if not ckpt_dir:
            print("❌ 找不到任何学生模型的 Checkpoint！")
            return
    print(f"🔥 加载学生模型 (TinyCharEncoder): {ckpt_dir}")

    # 1. Load Student
    args_path = os.path.join(os.path.dirname(ckpt_dir), "training_args.json")
    if not os.path.exists(args_path):
        args_path = glob.glob(os.path.join(os.path.dirname(ckpt_dir), "*_args.json"))
        args_path = args_path[0] if args_path else None
        
    student_d_model = 1024
    student_n_heads = 8
    student_n_layers = 6
    if args_path and os.path.exists(args_path):
        with open(args_path, 'r') as f:
            targs = json.load(f)
            student_d_model = targs.get("student_d_model", 1024)
            student_n_heads = targs.get("student_n_heads", 8)
            student_n_layers = targs.get("student_n_layers", 6)

    student = TinyCharEncoderCUDA(
        vocab_size=config.vocab_size,
        d_model=student_d_model,
        z_dim=1024
    ).to(device)
    
    from safetensors.torch import load_file
    student_safe = os.path.join(ckpt_dir, "student.safetensors")
    student_pt = os.path.join(ckpt_dir, "student.pt")
    if os.path.exists(student_safe):
        student.load_state_dict(load_file(student_safe), strict=True)
    elif os.path.exists(student_pt):
        student.load_state_dict(torch.load(student_pt, map_location="cpu"), strict=True)
    else:
        print(f"❌ 找不到 {student_safe} 或 {student_pt}")
        return
    student.eval()

    p0_ckpt = args.p0_ckpt
    if not p0_ckpt:
        p0_ckpt = get_latest_teacher_ckpt([
            "checkpoints",
            "checkpoints/run",
            "."
        ])
        if not p0_ckpt:
            print("❌ 找不到任何老师(Phase 0)的 Checkpoint！请确保目录中含有带有 'p0' 的文件夹")
            return
    print(f"🔥 加载老师解码器 (WeakDecoder): {p0_ckpt}")
    
    from safetensors.torch import load_file
    decoder_safe = os.path.join(p0_ckpt, "weak_decoder.safetensors")
    if not os.path.exists(decoder_safe):
        decoder_safe = os.path.join(p0_ckpt, "decoder.safetensors")
        if not os.path.exists(decoder_safe):
            print(f"❌ 找不到 {p0_ckpt}/weak_decoder.safetensors 也没有 decoder.safetensors")
            return
        
    decoder_state = load_file(decoder_safe)
    
    # Infer d_model and n_layers dynamically
    # z_proj.weight has shape (d_model, z_dim)
    # out_proj.weight has shape (vocab_size, d_model)
    decoder_d_model = 128 # Default fallback
    if "z_proj.weight" in decoder_state:
        decoder_d_model = decoder_state["z_proj.weight"].shape[0]
    elif "out_proj.weight" in decoder_state:
        decoder_d_model = decoder_state["out_proj.weight"].shape[1]
        
    decoder_n_layers = 0
    for k in decoder_state.keys():
        if k.startswith("transformer.layers.") or k.startswith("layers."):
            layer_idx = int(k.split(".")[2] if k.startswith("transformer") else k.split(".")[1])
            decoder_n_layers = max(decoder_n_layers, layer_idx + 1)
            
    if decoder_n_layers == 0:
        decoder_n_layers = getattr(config, "decoder_layers", 2)
        
    decoder = WeakDecoderCUDA(
        z_dim=1024,
        vocab_size=config.vocab_size,
        d_model=decoder_d_model,
        n_layers=decoder_n_layers
    ).to(device)
    
    load_mlx_safetensors_into_torch(decoder, decoder_safe)
    decoder.eval()

    print("\n" + "="*50)
    print(f"📝 原始输入 (Prompt): {args.prompt}")
    
    import re
    # If split_by_punct is true, split by common punctuation but KEEP the punctuation
    if args.split_by_punct:
        # Split pattern that captures the punctuation so it's kept in the result list
        # e.g., "你好，世界。" -> ["你好", "，", "世界", "。"] -> ["你好，", "世界。"]
        parts = re.split(r'([，。！？；：,\.!\?;:\n])', args.prompt)
        chunks = []
        current_chunk = ""
        for part in parts:
            current_chunk += part
            if re.match(r'[，。！？；：,\.!\?;:\n]', part) and current_chunk.strip():
                chunks.append(current_chunk.strip())
                current_chunk = ""
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
            
        if not chunks:
            chunks = [args.prompt]
        print(f"✂️ 开启了标点拆分模式，切分为 {len(chunks)} 个短句。")
    else:
        chunks = [args.prompt]
        print("⚠️ 未开启拆分模式 (整句压缩为一个Z向量)")

    print("="*50)
    
    final_reconstructed_text = ""
    
    for i, chunk in enumerate(chunks):
        if not chunk.strip():
            continue
            
        token_ids = tokenizer.encode(chunk, add_special_tokens=True)
        if not token_ids:
            continue
            
        tokens = torch.tensor([token_ids], dtype=torch.long).to(device)
        
        with torch.no_grad():
            z_pred = student(tokens) # (1, 1024)
            
        # print(f"✨ 提取第 {i+1} 句 Z 向量! L2 Norm: {torch.norm(z_pred).item():.4f}, Std: {z_pred.std().item():.4f}")
        
        start_token = tokenizer.bos_token_id
        with torch.no_grad():
            generated_ids = decoder.generate(
                z_pred, 
                start_token=start_token, 
                max_tokens=max(args.max_tokens, len(token_ids) + 10), 
                temperature=args.temperature,
                eos_token=tokenizer.eos_token_id
            )
            
        result_text = tokenizer.decode(generated_ids[1:])
        final_reconstructed_text += result_text
        
        if args.split_by_punct:
            print(f"  [{i+1}] 原文: {chunk}")
            print(f"      重建: {result_text}")
    
    print("="*50)
    print(f"🎯 完整原始输入:\n{args.prompt}")
    print(f"🧠 完整Z-空间重建:\n{final_reconstructed_text}")
    print("="*50)

if __name__ == "__main__":
    main()
