import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import sys
import glob
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir in sys.path and os.path.exists(os.path.join(script_dir, "model.py")):
    sys.path = [p for p in sys.path if p != script_dir] + [script_dir]

from training.core.char_tokenizer import CharTokenizer
from training.data.dialogue_dataset import GenericPairDataset
from model.hippo_oracle import HiPPOOracleCUDA
from model.flow_unet_cuda import FlowMatching1DUnet
from model.loss_hippo_cfm import hippo_cfm_loss
from model.phase1_charencoder_cuda import TinyCharEncoderCUDA
from model.continuous_translator_cuda import ContinuousToTextTranslatorCUDA
from training.core.args import get_training_parser
from training.core.checkpoint import Checkpointer
from torch.utils.data import DataLoader

def find_latest_translator_ckpt(base_dirs=["checkpoints/run", "checkpoints"]):
    candidates = []
    for bdir in base_dirs:
        if not bdir or not os.path.exists(bdir):
            continue
        for d in os.listdir(bdir):
            if "translator" in d and "step_" in d:
                full_dir = os.path.join(bdir, d)
                pt_file = os.path.join(full_dir, "translator.pt")
                if os.path.isfile(pt_file):
                    try:
                        step = int(d.split("step_")[-1])
                        candidates.append((step, pt_file))
                    except ValueError:
                        pass
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
    return None

def decode_with_eos_marker(tokenizer, token_ids):
    """Decodes token IDs and explicitly marks the first <EOS> as '<eos>' so sequence termination is visible."""
    if tokenizer.eos_token_id in token_ids:
        eos_idx = token_ids.index(tokenizer.eos_token_id)
        head = tokenizer.decode(token_ids[:eos_idx], skip_special_tokens=True)
        tail = tokenizer.decode(token_ids[eos_idx + 1:], skip_special_tokens=True)
        return f"{head}<eos>{tail}"
    return tokenizer.decode(token_ids, skip_special_tokens=True)

@torch.no_grad()
def sample_euler_ode(unet, x_prompt, h_potential, steps=10, seq_len=256, device='cuda'):
    """
    Euler ODE Integrator:
    dX_t = v(X_t, t, h_potential, k_attractors=x_prompt_norm) * dt
    Integrates from white noise X_0 ~ N(0, I) -> X_1_norm, then auto-denormalizes back to encoder manifold.
    """
    B = x_prompt.shape[0]
    D = x_prompt.shape[-1]
    x_prompt_norm = unet.normalize(x_prompt)
    x = torch.randn((B, seq_len, D), device=device, dtype=x_prompt.dtype)
    dt = 1.0 / steps
    for i in range(steps):
        t_val = i / float(steps)
        t_tensor = torch.full((B,), t_val, device=device, dtype=torch.float32)
        v_pred = unet(x, t_tensor, h_potential=h_potential, k_attractors=x_prompt_norm)
        x = x + v_pred * dt
    return unet.denormalize(x)

def train():
    parser = get_training_parser("Phase 4 HiPPO Oracle + Flow Match Engine (CUDA)")
    parser.add_argument("--data_path", type=str, default="data/Basic_ZH/moss_dialogues.jsonl", help="Path to dialogue dataset (file, dir, or glob)")
    parser.add_argument("--prompt_key", type=str, default=None, help="Explicit prompt key in dataset items")
    parser.add_argument("--target_key", type=str, default=None, help="Explicit target key in dataset items")
    parser.add_argument("--max_samples", type=int, default=None, help="Max number of samples to load (useful for smoke tests)")
    parser.add_argument("--encoder_ckpt", type=str, default="checkpoints/basencoder_step_115000/student.pt", help="Path to frozen Phase 1 encoder checkpoint")
    parser.add_argument("--translator_ckpt", type=str, default=None, help="Path to frozen translator checkpoint (defaults to auto-discovering latest)")
    parser.add_argument("--eval_interval", type=int, default=200, help="Print qualitative generation test every N steps")
    parser.add_argument("--euler_steps", type=int, default=10, help="Euler ODE integration steps for qualitative evaluation")
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["bf16", "fp16", "no"], help="Mixed precision mode (bf16, fp16, no)")
    parser.add_argument("--accum_steps", type=int, default=1, help="Number of gradient accumulation steps")
    parser.add_argument("--min_target_len", type=int, default=30, help="Filter out ultra-short target fragments (< N chars) to avoid 95%% padding dilution")
    parser.add_argument("--pad_loss_weight", type=float, default=0.05, help="Relative weight for <PAD> tokens in flow matching MSE loss")
    parser.set_defaults(ckpt_prefix="phase4_hippo_flow", batch_size=64, save_steps=1000)

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Mixed Precision Setup
    amp_enabled = False
    amp_dtype = torch.float32
    scaler = None

    if device.type == 'cuda' and args.mixed_precision != "no":
        if args.mixed_precision == "bf16":
            if torch.cuda.is_bf16_supported():
                amp_enabled = True
                amp_dtype = torch.bfloat16
                print("⚡ Mixed Precision: Enabled (Native BFloat16)")
            else:
                print("⚠️ BFloat16 not supported on this GPU, falling back to Float16")
                amp_enabled = True
                amp_dtype = torch.float16
                scaler = torch.cuda.amp.GradScaler()
                print("⚡ Mixed Precision: Enabled (Float16 + GradScaler)")
        elif args.mixed_precision == "fp16":
            amp_enabled = True
            amp_dtype = torch.float16
            scaler = torch.cuda.amp.GradScaler()
            print("⚡ Mixed Precision: Enabled (Float16 + GradScaler)")
    else:
        print("⚡ Mixed Precision: Disabled (FP32)")

    tokenizer = CharTokenizer()

    # 1. 通用数据加载器 (Generic Pair Dataset)
    dataset = GenericPairDataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        prompt_key=args.prompt_key,
        target_key=args.target_key,
        max_samples=args.max_samples,
        min_target_len=args.min_target_len,
        verbose=True
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # 2. Phase 1 Encoder (Frozen Continuous Geometric Mapper)
    encoder = TinyCharEncoderCUDA(vocab_size=tokenizer.vocab_size).to(device)
    if os.path.exists(args.encoder_ckpt):
        encoder.load_state_dict(torch.load(args.encoder_ckpt, map_location=device, weights_only=True))
        print(f"✅ Loaded frozen encoder from {args.encoder_ckpt}")
    else:
        print(f"⚠️ WARNING: Encoder checkpoint {args.encoder_ckpt} not found! Using random weights (for testing only).")
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # 3. Phase 4 Translator Lens (Frozen Continuous-to-Text Dictionary)
    translator_path = args.translator_ckpt
    if translator_path and ("*" in translator_path or "?" in translator_path):
        matched = sorted(glob.glob(translator_path))
        translator_path = matched[-1] if matched else None
    elif translator_path and os.path.isdir(translator_path):
        translator_path = os.path.join(translator_path, "translator.pt")
    if not translator_path:
        translator_path = find_latest_translator_ckpt(base_dirs=[args.out_dir, "checkpoints/run", "checkpoints"])

    translator = None
    if translator_path and os.path.exists(translator_path):
        translator = ContinuousToTextTranslatorCUDA(d_model=1024, vocab_size=tokenizer.vocab_size).to(device)
        translator.load_state_dict(torch.load(translator_path, map_location=device, weights_only=True))
        translator.eval()
        for p in translator.parameters():
            p.requires_grad = False
        print(f"✅ Loaded frozen translator lens from {translator_path}")
    else:
        print(f"⚠️ No translator checkpoint found at '{translator_path}'. Text preview will be disabled.")

    # 4. Phase 4 HiPPO Oracle (Frozen)
    hippo_oracle = HiPPOOracleCUDA(hidden_dim=1024, N=64).to(device)
    hippo_oracle.eval()

    # 5. Flow Match Unet (Trainable)
    unet = FlowMatching1DUnet(hidden_dim=1024, cond_dim=1024, num_layers=6).to(device)

    optimizer = optim.AdamW(unet.parameters(), lr=args.lr)

    checkpointer = Checkpointer(out_dir=args.out_dir, prefix=args.ckpt_prefix, keep_last_k=args.keep_last_k)
    checkpointer.register_model("unet", unet)
    checkpointer.register_optimizer("engine_opt", optimizer)
    checkpointer.register_args(args)

    global_step = checkpointer.load_latest() if args.auto_resume else 0
    start_epoch = global_step // len(dataloader) if len(dataloader) > 0 else 0

    print("🚀 Starting Phase 4 HiPPO + Flow Match Training (Curriculum Learning)...")

    max_steps = args.epochs * len(dataloader) if len(dataloader) > 0 else 1
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(start_epoch, args.epochs):
        resumed_loader = checkpointer.wrap_dataloader(dataloader, epoch, global_step)
        unet.train()

        for batch_idx, (w_t, w_tgt) in enumerate(resumed_loader):
            w_t = w_t.to(device)
            w_tgt = w_tgt.to(device)
            valid_mask = (w_tgt != tokenizer.pad_token_id)

            with torch.no_grad():
                h_t = encoder.get_normalized_embeddings(w_t)
                for layer in encoder.layers:
                    h_t = layer(h_t, None)
                x_prompt = encoder.final_ln(h_t)

                h_tgt = encoder.get_normalized_embeddings(w_tgt)
                for layer in encoder.layers:
                    h_tgt = layer(h_tgt, None)
                x_target = encoder.final_ln(h_tgt)

                # 初次前向时做尺度方差体检（仅统计非 PAD 真实字符位置）并启动自适应尺度标定
                if global_step == 0 and batch_idx == 0:
                    valid_tokens = x_target[valid_mask] if valid_mask.any() else x_target
                    std_val = valid_tokens.std().item()
                    mean_val = valid_tokens.mean().item()
                    unet.calibrate_manifold(x_target, valid_mask=valid_mask)
                    norm_x = unet.normalize(x_target)
                    norm_std = (norm_x[valid_mask] if valid_mask.any() else norm_x).std().item()
                    scale_factor = 1.0 / unet.manifold_std.item()
                    valid_pct = valid_mask.float().mean().item() * 100.0
                    print(f"📊 Manifold Geometry Check (Valid Tokens: {valid_pct:.1f}%) -> Raw Mean: {mean_val:.4f}, Raw Valid Std: {std_val:.4f}")
                    print(f"⚖️ Adaptive Manifold Scaler -> Auto Scale: {scale_factor:.2f}x | Normalized Valid Std: {norm_std:.4f}")

            # Curriculum Schedule: Attractor probability increases smoothly from 0 to 0.5
            p_attractor = min(0.5, (global_step / max_steps) * 1.0)
            use_attractor = torch.rand(1).item() < p_attractor

            # Forward + Loss under autocast
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                loss = hippo_cfm_loss(
                    unet,
                    hippo_oracle,
                    x_prompt,
                    x_target,
                    use_knowledge_attractor=use_attractor,
                    valid_mask=valid_mask,
                    pad_loss_weight=args.pad_loss_weight
                )
                loss_scaled = loss / args.accum_steps

            if scaler is not None:
                scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()

            # Gradient accumulation step
            if (batch_idx + 1) % args.accum_steps == 0 or (batch_idx + 1) == len(resumed_loader):
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            global_step += 1
            if global_step % 50 == 0:
                scale_factor = 1.0 / unet.manifold_std.item()
                print(f"Epoch {epoch+1} | Step {global_step} | Phase 4 Loss: {loss.item():.4f} | Attractor Prob: {p_attractor:.2f} | Scale: {scale_factor:.2f}x")

            # 定期进行 Euler ODE 真实文本生成测试
            if translator is not None and (global_step == 50 or global_step % args.eval_interval == 0):
                unet.eval()
                with torch.no_grad():
                    # 提取第一条样本进行生成评估 (在归一化流形空间提取 HiPPO 势能曲线)
                    p_sample = x_prompt[:1]
                    tgt_sample = x_target[:1]
                    tgt_sample_norm = unet.normalize(tgt_sample)
                    h_potential = hippo_oracle(tgt_sample_norm)

                    # 10 步 Euler ODE 从高斯白噪声中积分生成连续曲线
                    with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                        x_gen = sample_euler_ode(
                            unet=unet,
                            x_prompt=p_sample,
                            h_potential=h_potential,
                            steps=args.euler_steps,
                            seq_len=args.seq_len,
                            device=device
                        )

                    # 解码生成轨迹与真实 Target (显式标注 <eos> 停止符位置)
                    gen_tokens = translator.decode_greedy(x_gen.float())[0].tolist()
                    gen_text = decode_with_eos_marker(tokenizer, gen_tokens)

                    ref_tokens = translator.decode_greedy(tgt_sample.float())[0].tolist()
                    ref_text = decode_with_eos_marker(tokenizer, ref_tokens)

                    prompt_text = tokenizer.decode(w_t[0].tolist())

                    print("\n" + "="*50)
                    print(f"🔮 [Step {global_step}] Flow Match Euler ODE Generation Test ({args.euler_steps} steps)")
                    print(f"Prompt:    {prompt_text}")
                    print(f"Target:    {ref_text}")
                    print(f"Generated: {gen_text}")
                    print("="*50 + "\n")

                unet.train()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if args.save_steps and global_step % args.save_steps == 0:
                checkpointer.save(global_step)

        checkpointer.save(global_step)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    train()
