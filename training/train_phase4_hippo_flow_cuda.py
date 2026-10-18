import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
import glob

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
        if not os.path.exists(bdir):
            continue
        for d in os.listdir(bdir):
            if "translator_conv1d_step_" in d:
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

@torch.no_grad()
def sample_euler_ode(unet, x_prompt, h_potential, steps=10, seq_len=256, device='cuda'):
    """
    Euler ODE Integrator:
    dX_t = v(X_t, t, h_potential, k_attractors=x_prompt) * dt
    Integrates from white noise X_0 ~ N(0, I) -> X_1
    """
    B = x_prompt.shape[0]
    D = x_prompt.shape[-1]
    x = torch.randn((B, seq_len, D), device=device)
    dt = 1.0 / steps
    for i in range(steps):
        t_val = i / float(steps)
        t_tensor = torch.full((B,), t_val, device=device, dtype=torch.float32)
        v_pred = unet(x, t_tensor, h_potential=h_potential, k_attractors=x_prompt)
        x = x + v_pred * dt
    return x

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
    parser.set_defaults(ckpt_prefix="phase4_hippo_flow")

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    tokenizer = CharTokenizer()

    # 1. 通用数据加载器 (Generic Pair Dataset)
    dataset = GenericPairDataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        prompt_key=args.prompt_key,
        target_key=args.target_key,
        max_samples=args.max_samples,
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
    translator_path = args.translator_ckpt or find_latest_translator_ckpt()
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

    for epoch in range(start_epoch, args.epochs):
        resumed_loader = checkpointer.wrap_dataloader(dataloader, epoch, global_step)
        unet.train()

        for batch_idx, (w_t, w_tgt) in enumerate(resumed_loader):
            w_t = w_t.to(device)
            w_tgt = w_tgt.to(device)

            with torch.no_grad():
                h_t = encoder.get_normalized_embeddings(w_t)
                for layer in encoder.layers:
                    h_t = layer(h_t, None)
                x_prompt = encoder.final_ln(h_t)

                h_tgt = encoder.get_normalized_embeddings(w_tgt)
                for layer in encoder.layers:
                    h_tgt = layer(h_tgt, None)
                x_target = encoder.final_ln(h_tgt)

                # 初次前向时做尺度方差体检
                if global_step == 0 and batch_idx == 0:
                    std_val = x_target.std().item()
                    mean_val = x_target.mean().item()
                    print(f"📊 Manifold Geometry Check -> Target Mean: {mean_val:.4f}, Std: {std_val:.4f}")
                    if abs(std_val - 1.0) > 0.5:
                        print(f"⚠️ Warning: Target std ({std_val:.4f}) deviates significantly from 1.0.")

            optimizer.zero_grad()

            # Curriculum Schedule: Attractor probability increases smoothly from 0 to 0.5
            p_attractor = min(0.5, (global_step / max_steps) * 1.0)
            use_attractor = torch.rand(1).item() < p_attractor

            loss = hippo_cfm_loss(unet, hippo_oracle, x_prompt, x_target, use_knowledge_attractor=use_attractor)
            loss.backward()
            optimizer.step()

            global_step += 1
            if global_step % 50 == 0:
                print(f"Epoch {epoch+1} | Step {global_step} | Phase 4 Loss: {loss.item():.4f} | Attractor Prob: {p_attractor:.2f}")

            # 定期进行 Euler ODE 真实文本生成测试
            if translator is not None and (global_step == 50 or global_step % args.eval_interval == 0):
                unet.eval()
                with torch.no_grad():
                    # 提取第一条样本进行生成评估
                    p_sample = x_prompt[:1]
                    tgt_sample = x_target[:1]
                    h_potential = hippo_oracle(tgt_sample)

                    # 10 步 Euler ODE 从高斯白噪声中积分生成连续曲线
                    x_gen = sample_euler_ode(
                        unet=unet,
                        x_prompt=p_sample,
                        h_potential=h_potential,
                        steps=args.euler_steps,
                        seq_len=args.seq_len,
                        device=device
                    )

                    # 解码生成轨迹
                    gen_tokens = translator.decode_greedy(x_gen)[0].tolist()
                    gen_text = tokenizer.decode(gen_tokens)

                    # 真实 Target 解码 (Sanity Check)
                    ref_tokens = translator.decode_greedy(tgt_sample)[0].tolist()
                    ref_text = tokenizer.decode(ref_tokens)

                    prompt_text = tokenizer.decode(w_t[0].tolist())

                    print("\n" + "="*50)
                    print(f"🔮 [Step {global_step}] Flow Match Euler ODE Generation Test ({args.euler_steps} steps)")
                    print(f"Prompt:    {prompt_text}")
                    print(f"Target:    {ref_text}")
                    print(f"Generated: {gen_text}")
                    print("="*50 + "\n")

                unet.train()

        checkpointer.save(global_step)

if __name__ == "__main__":
    train()
