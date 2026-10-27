import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import sys
import math
import glob
import json
import re
import random
from typing import List, Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir in sys.path and os.path.exists(os.path.join(script_dir, "model.py")):
    sys.path = [p for p in sys.path if p != script_dir] + [script_dir]

from training.core.char_tokenizer import CharTokenizer

from model.hippo_oracle import HiPPOOracleCUDA
from model.flow_unet_cuda import FlowMatching1DUnet
from model.loss_hippo_cfm import hippo_cfm_loss

try:
    from training.data.dialogue_dataset import GenericPairDataset
except ImportError:
    from training.data.knowledge_dataset import GenericPairDataset

from model.phase1_charencoder_cuda import TinyCharEncoderCUDA
from model.continuous_translator_cuda import ContinuousToTextTranslatorCUDA
from training.core.args import get_training_parser
from training.core.checkpoint import Checkpointer
from torch.utils.data import DataLoader

CANONICAL_DOMAINS = [
    "grounded_instruct",
    "code_api",
    "classical_quote",
    "encyclopedia_rag",
    "math_rule",
    "literature_lore",
    "general",
    "dialogue_social",
]


def get_warmup_cosine_scheduler(
    optimizer: optim.Optimizer,
    warmup_steps: int,
    max_steps: int,
    min_lr_ratio: float = 0.1,
) -> optim.lr_scheduler.LambdaLR:
    warmup_steps = max(0, int(warmup_steps))
    max_steps = max(1, int(max_steps))
    min_lr_ratio = float(min_lr_ratio)

    def lr_lambda(current_step: int) -> float:
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        if current_step >= max_steps:
            return min_lr_ratio
        decay_steps = max(1, max_steps - warmup_steps)
        progress = float(current_step - warmup_steps) / float(decay_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def select_fixed_benchmark_indices(dataset, seed: int = 42, max_domains: int = 8) -> List[Tuple[str, int]]:
    """
    Deterministically selects 1 representative sample index per available domain (up to `max_domains`)
    using a fixed random seed (42) so qualitative cross-domain evaluation is reproducible and benchmarks all domains.
    """
    rng = random.Random(seed)
    domain_to_indices: Dict[str, List[int]] = {}

    if hasattr(dataset, "samples") and isinstance(dataset.samples, list):
        for idx, rec in enumerate(dataset.samples):
            dom = "general"
            if isinstance(rec, tuple):
                if len(rec) >= 4 and isinstance(rec[3], str) and rec[3].strip():
                    dom = rec[3].strip()
                elif len(rec) >= 3 and isinstance(rec[2], str) and rec[2].strip():
                    dom = rec[2].strip()
                elif len(rec) >= 1 and isinstance(rec[0], str):
                    m = re.match(r"^\s*【([^】]{1,20})】", rec[0])
                    if m:
                        dom = m.group(1).strip()
            elif isinstance(rec, dict):
                dom = str(rec.get("domain", "general")).strip()
            domain_to_indices.setdefault(dom, []).append(idx)
    else:
        n = min(len(dataset), max_domains)
        return [("general", i) for i in range(n)]

    if len(domain_to_indices) <= 1 and len(dataset) > 1:
        n = min(len(dataset), max_domains)
        step = max(1, len(dataset) // n)
        return [(f"sample_{i*step}", i * step) for i in range(n)]

    ordered_domains = [d for d in CANONICAL_DOMAINS if d in domain_to_indices]
    for d in sorted(domain_to_indices.keys()):
        if d not in ordered_domains:
            ordered_domains.append(d)

    selected: List[Tuple[str, int]] = []
    for dom in ordered_domains[:max_domains]:
        cands = domain_to_indices[dom]
        chosen_idx = rng.choice(cands)
        selected.append((dom, chosen_idx))

    return selected


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
                        step_str = d.split("step_")[-1].replace("_emergency", "")
                        step = int(step_str)
                        candidates.append((step, pt_file))
                    except ValueError:
                        pass
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
    return None


def find_compatible_unet_ckpt(unet, out_dir, prefix="phase4_hippo_flow", device="cpu"):
    """
    Finds the latest checkpoint with matching architecture by testing strict `load_state_dict`
    compatibility in descending order of global_step. Skips incompatible legacy checkpoints cleanly.
    """
    if not out_dir or not os.path.exists(out_dir):
        return None, 0

    candidates = []
    expected_step_prefix = f"{prefix}_step_" if prefix else "step_"
    expected_emergency = f"{prefix}_latest_emergency" if prefix else "latest_emergency"

    for d in os.listdir(out_dir):
        d_path = os.path.join(out_dir, d)
        if not os.path.isdir(d_path):
            continue
        if prefix and d != expected_emergency and not d.startswith(expected_step_prefix):
            continue

        step = -1
        for file in os.listdir(d_path):
            if file.endswith(".json") and "dataloader" in file:
                try:
                    with open(os.path.join(d_path, file), "r") as f:
                        state = json.load(f)
                        step = state.get("global_step", -1)
                except Exception:
                    pass
        if step == -1:
            m = re.search(r"step_(\d+)", d)
            if m:
                step = int(m.group(1))
        if step >= 0:
            unet_pt = os.path.join(d_path, "unet.pt")
            if os.path.isfile(unet_pt):
                candidates.append((step, d_path, d))

    candidates.sort(key=lambda x: x[0], reverse=True)

    for step, c_path, c_name in candidates:
        unet_pt = os.path.join(c_path, "unet.pt")
        try:
            state_dict = torch.load(unet_pt, map_location=device, weights_only=True)
            unet.load_state_dict(state_dict, strict=True)
            print(f"✅ Verified compatible Phase 4 Unet checkpoint '{c_name}' (step {step})")
            return c_path, step
        except Exception as e:
            print(f"ℹ️ Skipping incompatible legacy checkpoint '{c_name}' (step {step}): {e}")

    return None, 0


def decode_with_eos_marker(tokenizer, token_ids):
    """Decodes token IDs and explicitly marks the first <EOS> as '<eos>' so sequence termination is visible."""
    if tokenizer.eos_token_id in token_ids:
        eos_idx = token_ids.index(tokenizer.eos_token_id)
        head = tokenizer.decode(token_ids[:eos_idx], skip_special_tokens=True)
        tail = tokenizer.decode(token_ids[eos_idx + 1:], skip_special_tokens=True)
        return f"{head}<eos>{tail}"
    return tokenizer.decode(token_ids, skip_special_tokens=True)


@torch.no_grad()
def encode_tokens(encoder, token_ids: torch.Tensor) -> torch.Tensor:
    h = encoder.get_normalized_embeddings(token_ids)
    for layer in encoder.layers:
        h = layer(h, None)
    return encoder.final_ln(h)


@torch.no_grad()
def sample_euler_ode(unet, x_prompt, h_potential, steps=10, seq_len=128, prompt_mask=None, device="cpu"):
    """
    Standard Stable Euler ODE Integrator conditioned on 2D Sentence Vector `h_potential: (B, 1024)` + `k_attractors`:
    dX_t = v(X_t, t, h_potential=z_sent, k_attractors=x_prompt_norm, k_mask=prompt_mask) * dt
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
        v_pred = unet(x, t_tensor, h_potential=h_potential, k_attractors=x_prompt_norm, k_mask=prompt_mask)
        x = x + v_pred * dt
    return unet.denormalize(x)


def train():
    default_data_path = (
        "/content/omni_knowledge_corpus.jsonl"
        if os.path.exists("/content/omni_knowledge_corpus.jsonl")
        else "data/Basic_ZH/moss_dialogues.jsonl"
    )

    parser = get_training_parser("Phase 4 HiPPO 2D Sentence Vector + Flow Match Engine (CUDA 021)")
    parser.add_argument("--data_path", type=str, default=default_data_path, help="Path to dialogue/knowledge dataset (file, dir, or glob)")
    parser.add_argument("--prompt_seq_len", type=int, default=256, help="Max token length for Prompt + Knowledge k_attractors")
    parser.add_argument("--prompt_key", type=str, default=None, help="Explicit prompt key in dataset items")
    parser.add_argument("--target_key", type=str, default=None, help="Explicit target key in dataset items")
    parser.add_argument("--max_samples", type=int, default=None, help="Max number of samples to load (useful for smoke tests)")
    parser.add_argument("--encoder_ckpt", type=str, default="checkpoints/basencoder_step_115000/student.pt", help="Path to frozen Phase 1 encoder checkpoint")
    parser.add_argument("--translator_ckpt", type=str, default=None, help="Path to frozen translator checkpoint (defaults to auto-discovering latest)")
    parser.add_argument("--eval_interval", type=int, default=200, help="Print qualitative generation test every N steps")
    parser.add_argument("--euler_steps", type=int, default=10, help="Euler ODE integration steps for qualitative evaluation")
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["bf16", "fp16", "no"], help="Mixed precision mode (bf16, fp16, no)")
    parser.add_argument("--accum_steps", type=int, default=1, help="Number of gradient accumulation steps")
    parser.add_argument("--min_target_len", type=int, default=8, help="Minimum clause character length after punctuation split")
    parser.add_argument("--max_clause_chars", type=int, default=120, help="Maximum clause character length after punctuation split")
    parser.add_argument("--split_by_punctuation", action="store_true", default=True, help="Split multi-sentence targets into progressive single clauses")
    parser.add_argument("--no_split_by_punctuation", dest="split_by_punctuation", action="store_false", help="Disable punctuation splitting")
    parser.add_argument("--attractor_prob", type=float, default=0.90, help="Probability of enabling k_attractors during Phase 4 training (default 0.90)")
    parser.add_argument("--pad_loss_weight", type=float, default=0.05, help="Relative weight for <PAD> tokens in flow matching MSE loss")
    parser.set_defaults(
        seq_len=128,
        warmup_steps=500,
        ckpt_prefix="phase4_sentvec_flow",
        batch_size=64,
        save_steps=1000,
    )

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Mixed Precision Setup
    amp_enabled = False
    amp_dtype = torch.float32
    scaler = None

    if device.type == "cuda" and args.mixed_precision != "no":
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

    # 1. Punctuation-Split Single-Clause Dataset with Negative Distractor Removal
    dataset = GenericPairDataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        prompt_seq_len=args.prompt_seq_len,
        target_seq_len=args.seq_len,
        prompt_key=args.prompt_key,
        target_key=args.target_key,
        max_samples=args.max_samples,
        min_target_len=args.min_target_len,
        split_by_punctuation=args.split_by_punctuation,
        max_clause_chars=args.max_clause_chars,
        min_clause_chars=args.min_target_len,
        verbose=True,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Cross-domain benchmark: select 1 representative sample per domain (up to 8 domains)
    eval_benchmark_samples = select_fixed_benchmark_indices(dataset, seed=42, max_domains=8)
    bench_domains_str = ", ".join(f"{dom} (#{idx})" for dom, idx in eval_benchmark_samples)
    print(f"📌 Fixed Cross-Domain Benchmark Samples ({len(eval_benchmark_samples)} domains): {bench_domains_str}")

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

    # 4. Phase 4 HiPPO Oracle (Frozen 2D Sentence Vector Bottleneck)
    hippo_oracle = HiPPOOracleCUDA(hidden_dim=1024, N=64, num_subbands=16).to(device)
    hippo_oracle.eval()

    # 5. Flow Match Unet (Trainable Right Brain 021)
    unet = FlowMatching1DUnet(hidden_dim=1024, cond_dim=1024, num_layers=6, num_subbands=16).to(device)
    optimizer = optim.AdamW(unet.parameters(), lr=args.lr)

    max_steps = max(1, args.epochs * len(dataloader))
    scheduler = get_warmup_cosine_scheduler(
        optimizer=optimizer,
        warmup_steps=args.warmup_steps,
        max_steps=max_steps,
    )

    checkpointer = Checkpointer(out_dir=args.out_dir, prefix=args.ckpt_prefix, keep_last_k=args.keep_last_k)
    checkpointer.register_model("unet", unet)
    checkpointer.register_optimizer("engine_opt", optimizer)
    checkpointer.register_optimizer("lr_sched", scheduler)
    checkpointer.register_args(args)

    if args.auto_resume or args.resume_from:
        if args.resume_from and os.path.isdir(args.resume_from):
            compat_path, compat_step = args.resume_from, 0
        else:
            compat_path, compat_step = find_compatible_unet_ckpt(
                unet=unet,
                out_dir=args.out_dir,
                prefix=args.ckpt_prefix,
                device=device,
            )
        if compat_path:
            try:
                loaded_step = checkpointer.load(compat_path)
                global_step = loaded_step if loaded_step > 0 else compat_step
                checkpointer.set_step(global_step)
                sched_file = os.path.join(compat_path, "lr_sched.pt")
                if not os.path.exists(sched_file) and global_step > 0:
                    scheduler.step(global_step)
                print(f"✅ Successfully resumed Phase 4 training from step {global_step} ({compat_path})")
            except Exception as e:
                print(f"⚠️ Error loading checkpoint state ({e}). Starting fresh from step 0!")
                global_step = 0
                checkpointer.set_step(0)
        else:
            print(f"ℹ️ No compatible Phase 4 checkpoints found in '{args.out_dir}'. Starting fresh from step 0!")
            global_step = 0
            checkpointer.set_step(0)
    else:
        global_step = 0
        checkpointer.set_step(0)

    start_epoch = global_step // len(dataloader) if len(dataloader) > 0 else 0

    print(
        f"🚀 Starting Phase 4 2D Sentence Vector + Sign-Aligned Flow Match Training (021) "
        f"(seq_len={args.seq_len}, prompt_seq_len={args.prompt_seq_len}, "
        f"attractor_prob={args.attractor_prob:.2f}, warmup={args.warmup_steps})..."
    )

    optimizer.zero_grad(set_to_none=True)

    for epoch in range(start_epoch, args.epochs):
        resumed_loader = checkpointer.wrap_dataloader(dataloader, epoch, global_step)
        unet.train()

        for batch_idx, batch_data in enumerate(resumed_loader):
            if len(batch_data) == 2:
                w_t, w_tgt = batch_data
            else:
                w_t, w_tgt = batch_data[0], batch_data[-1]

            w_t = w_t.to(device)
            w_tgt = w_tgt.to(device)
            valid_mask = (w_tgt != tokenizer.pad_token_id)
            prompt_mask = (w_t != tokenizer.pad_token_id)

            with torch.no_grad():
                x_prompt = encode_tokens(encoder, w_t)
                x_target = encode_tokens(encoder, w_tgt)

                if global_step == 0 and batch_idx == 0:
                    valid_tokens = x_target[valid_mask] if valid_mask.any() else x_target
                    std_val = valid_tokens.std().item()
                    mean_val = valid_tokens.mean().item()
                    unet.calibrate_manifold(x_target, valid_mask=valid_mask)
                    norm_x = unet.normalize(x_target)
                    norm_std = (norm_x[valid_mask] if valid_mask.any() else norm_x).std().item()
                    scale_factor = 1.0 / unet.manifold_std.item()
                    valid_pct = valid_mask.float().mean().item() * 100.0
                    z_check = hippo_oracle(
                        norm_x,
                        return_sequence=False,
                        valid_mask=valid_mask,
                        subspace_proj=unet.subspace_proj,
                    )
                    h_wave_check = unet.unpack_sentence_vector(z_check, args.seq_len)
                    wave_cos = (
                        torch.nn.functional.cosine_similarity(
                            norm_x[valid_mask].float(), h_wave_check[valid_mask].float(), dim=-1
                        )
                        .mean()
                        .item()
                        if valid_mask.any()
                        else 0.0
                    )
                    print(
                        f"📊 Manifold Geometry Check (Valid Tokens: {valid_pct:.1f}%) -> "
                        f"Raw Mean: {mean_val:.4f}, Raw Valid Std: {std_val:.4f}"
                    )
                    print(
                        f"⚖️ Adaptive Manifold Scaler -> Auto Scale: {scale_factor:.2f}x | "
                        f"Normalized Valid Std: {norm_std:.4f} | 2D Z_sent Shape: {tuple(z_check.shape)} "
                        f"(Std: {z_check.std().item():.4f}, Unpacked Wave Cos: {wave_cos:.4f})"
                    )

            p_attractor = float(args.attractor_prob)
            use_attractor = torch.rand(1).item() < p_attractor

            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                loss, metrics = hippo_cfm_loss(
                    unet,
                    hippo_oracle,
                    x_prompt,
                    x_target,
                    use_knowledge_attractor=use_attractor,
                    valid_mask=valid_mask,
                    prompt_mask=prompt_mask,
                    pad_loss_weight=args.pad_loss_weight,
                    return_metrics=True,
                )
                loss_scaled = loss / args.accum_steps

            if scaler is not None:
                scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()

            if (batch_idx + 1) % args.accum_steps == 0 or (batch_idx + 1) == len(resumed_loader):
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            global_step += 1
            checkpointer.set_step(global_step)

            if global_step % 50 == 0:
                scale_factor = 1.0 / unet.manifold_std.item()
                print(
                    f"Epoch {epoch+1} | Step {global_step} | "
                    f"Loss: {loss.item():.4f} (Vel: {metrics['loss_vel'].item():.4f}, Cos: {metrics['loss_cos'].item():.4f}) | "
                    f"Z_sent Std: {metrics['z_sent_std'].item():.3f} | "
                    f"Attractor Prob: {p_attractor:.2f} | Scale: {scale_factor:.2f}x"
                )

            # Upgraded cross-domain evaluation: 1 representative sample per domain (up to 8 domains)
            if translator is not None and (global_step == 50 or global_step % args.eval_interval == 0):
                unet.eval()
                with torch.no_grad():
                    print("\n" + "=" * 70)
                    print(
                        f"🔮 [Step {global_step}] 2D Sentence Vector Cross-Domain Euler ODE Evaluation "
                        f"({len(eval_benchmark_samples)} domains, {args.euler_steps} steps)"
                    )
                    for s_idx, (dom_name, ds_idx) in enumerate(eval_benchmark_samples):
                        item_data = dataset[ds_idx]
                        if len(item_data) == 2:
                            w_p_s, w_t_s = item_data
                        else:
                            w_p_s, w_t_s = item_data[0], item_data[-1]

                        w_p_s = w_p_s.unsqueeze(0).to(device)
                        w_t_s = w_t_s.unsqueeze(0).to(device)
                        tgt_valid_mask = (w_t_s != tokenizer.pad_token_id)
                        p_valid_mask = (w_p_s != tokenizer.pad_token_id)

                        p_sample = encode_tokens(encoder, w_p_s)
                        tgt_sample = encode_tokens(encoder, w_t_s)
                        tgt_sample_norm = unet.normalize(tgt_sample)

                        z_sent = hippo_oracle(
                            tgt_sample_norm,
                            return_sequence=False,
                            valid_mask=tgt_valid_mask,
                            subspace_proj=unet.subspace_proj,
                        )

                        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                            x_gen = sample_euler_ode(
                                unet=unet,
                                x_prompt=p_sample,
                                h_potential=z_sent,
                                steps=args.euler_steps,
                                seq_len=args.seq_len,
                                prompt_mask=p_valid_mask,
                                device=device,
                            )

                        gen_tokens = translator.decode_greedy(x_gen.float())[0].tolist()
                        gen_text = decode_with_eos_marker(tokenizer, gen_tokens)

                        ref_tokens = translator.decode_greedy(tgt_sample.float())[0].tolist()
                        ref_text = decode_with_eos_marker(tokenizer, ref_tokens)

                        prompt_text = tokenizer.decode(w_p_s[0].tolist(), skip_special_tokens=True)
                        valid_len = int(tgt_valid_mask.sum().item())

                        print(f"--- [{dom_name}] Sample #{s_idx + 1} (idx={ds_idx}) ---")
                        print(f"Prompt + Knowledge (k_attractors): {prompt_text}")
                        print(f"Target Sentence (len={valid_len}, Z_sent shape: {tuple(z_sent.shape)}): {ref_text}")
                        print(f"Right-Brain Generated (from 1x1024 Z_sent + k_attractors): {gen_text}")
                    print("=" * 70 + "\n")

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
