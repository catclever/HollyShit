import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import sys
import glob
import inspect
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir in sys.path and os.path.exists(os.path.join(script_dir, "model.py")):
    sys.path = [p for p in sys.path if p != script_dir] + [script_dir]

from training.core.char_tokenizer import CharTokenizer
from training.data.dialogue_dataset import GenericPairDataset

try:
    from training.data.knowledge_dataset import MultiStreamKnowledgeDataset
except ImportError:
    try:
        from training.data.knowledge_dataset_008 import MultiStreamKnowledgeDataset
    except ImportError:
        MultiStreamKnowledgeDataset = None

try:
    from model.flow_mamba_012 import MultiStreamMambaPlanner, MambaReader
except ImportError:
    from model.flow_mamba import MultiStreamMambaPlanner, MambaReader

from model.hippo_oracle import HiPPOOracleCUDA
from model.flow_unet_cuda import FlowMatching1DUnet

try:
    from model.loss_mamba_cfm_012 import mamba_cfm_loss
except ImportError:
    from model.loss_mamba_cfm import mamba_cfm_loss as _base_mamba_cfm_loss
    if "x_knowledge" not in inspect.signature(_base_mamba_cfm_loss).parameters:
        from model.loss_mamba_cfm_009 import mamba_cfm_loss
    else:
        mamba_cfm_loss = _base_mamba_cfm_loss

from model.phase1_charencoder_cuda import TinyCharEncoderCUDA
from model.continuous_translator_cuda import ContinuousToTextTranslatorCUDA
from training.core.args import get_training_parser
from training.core.checkpoint import Checkpointer
from torch.utils.data import DataLoader


def find_latest_unet_ckpt(base_dirs=["checkpoints/run", "checkpoints"]):
    candidates = []
    for bdir in base_dirs:
        if not bdir or not os.path.exists(bdir):
            continue
        for d in os.listdir(bdir):
            if ("phase4_omni_flow" in d or "phase4_hippo_flow" in d) and "step_" in d:
                full_dir = os.path.join(bdir, d)
                pt_file = os.path.join(full_dir, "unet.pt")
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
    """Encodes token IDs (N, L) through the frozen Phase 1 TinyCharEncoderCUDA."""
    h = encoder.get_normalized_embeddings(token_ids)
    for layer in encoder.layers:
        h = layer(h, None)
    return encoder.final_ln(h)


@torch.no_grad()
def encode_knowledge_streams(encoder, w_know: torch.Tensor, pad_token_id: int):
    """
    Encodes multi-stream knowledge tokens (B, M, L_k) through the frozen Phase 1 encoder
    in a single batched pass over active non-empty streams.
    Returns:
        x_knowledge: (B, M, L_k, D)
        knowledge_mask: (B, M, L_k) boolean mask
    """
    B, M, L_k = w_know.shape
    knowledge_mask = (w_know != pad_token_id)
    stream_active = knowledge_mask.any(dim=-1)  # (B, M)

    if not stream_active.any():
        x_knowledge = torch.zeros((B, M, L_k, 1024), device=w_know.device, dtype=torch.float32)
        return x_knowledge, knowledge_mask

    flat_w = w_know.reshape(B * M, L_k)
    active_idx = torch.nonzero(stream_active.reshape(B * M), as_tuple=False).squeeze(-1)
    active_emb = encode_tokens(encoder, flat_w[active_idx])  # (N_active, L_k, D)

    D = active_emb.shape[-1]
    flat_out = torch.zeros((B * M, L_k, D), device=w_know.device, dtype=active_emb.dtype)
    flat_out[active_idx] = active_emb
    x_knowledge = flat_out.reshape(B, M, L_k, D)
    return x_knowledge, knowledge_mask


@torch.no_grad()
def build_fixed_length_attractors(
    x_context_norm: torch.Tensor,
    context_mask: torch.Tensor = None,
    x_knowledge_norm: torch.Tensor = None,
    knowledge_mask: torch.Tensor = None,
    seq_len: int = 512,
) -> torch.Tensor:
    """
    Builds a fixed-length (B, seq_len, D) attractor tensor matching Phase 4 FlowMatching1DUnet's
    exact training distribution (`k_attractors = unet.normalize(x_prompt)` of length `seq_len=512`).
    Places valid context tokens + active valid knowledge tokens into the front of a (B, seq_len, D)
    buffer padded with normalized <PAD> embeddings from `x_context_norm` up to `seq_len`.
    """
    B, L_c, D = x_context_norm.shape
    pad_vec = x_context_norm[:, -1:, :].expand(B, seq_len, D).clone()
    attractor_buf = pad_vec.clone()

    for b in range(B):
        pieces = []
        if context_mask is not None and context_mask[b].any():
            pieces.append(x_context_norm[b][context_mask[b].bool()])
        else:
            pieces.append(x_context_norm[b])

        if x_knowledge_norm is not None:
            if knowledge_mask is not None:
                k_mask_b = knowledge_mask[b].bool()
                if k_mask_b.any():
                    pieces.append(x_knowledge_norm[b][k_mask_b])
            else:
                pieces.append(x_knowledge_norm[b].reshape(-1, D))

        combined = torch.cat(pieces, dim=0)[:seq_len]
        n_valid = combined.shape[0]
        if n_valid > 0:
            attractor_buf[b, :n_valid, :] = combined

    return attractor_buf


@torch.no_grad()
def sample_euler_ode(unet, x_prompt, h_potential, k_attractors_norm=None, steps=10, seq_len=256, device='cuda'):
    """
    Euler ODE Integrator:
    dX_t = v(X_t, t, h_potential, k_attractors=k_attractors_norm) * dt
    Integrates from white noise X_0 ~ N(0, I) -> X_1_norm, then auto-denormalizes back to encoder manifold.
    """
    B = x_prompt.shape[0]
    D = x_prompt.shape[-1]
    if k_attractors_norm is None:
        k_attractors_norm = unet.normalize(x_prompt)
    x = torch.randn((B, seq_len, D), device=device, dtype=x_prompt.dtype)
    dt = 1.0 / steps
    for i in range(steps):
        t_val = i / float(steps)
        t_tensor = torch.full((B,), t_val, device=device, dtype=torch.float32)
        v_pred = unet(x, t_tensor, h_potential=h_potential, k_attractors=k_attractors_norm)
        x = x + v_pred * dt
    return unet.denormalize(x)


def train():
    default_data = (
        "/content/omni_corpus.jsonl"
        if os.path.exists("/content/omni_corpus.jsonl") and not os.path.exists("data/Basic_ZH/moss_dialogues.jsonl")
        else "data/Basic_ZH/moss_dialogues.jsonl"
    )
    parser = get_training_parser("Phase 5 Multi-Stream Mamba Predictive Planner (CUDA)")
    parser.add_argument("--data_path", type=str, default=default_data, help="Path to dialogue/knowledge dataset (file, dir, or glob)")
    parser.add_argument("--prompt_key", type=str, default=None, help="Explicit prompt key in dataset items")
    parser.add_argument("--target_key", type=str, default=None, help="Explicit target key in dataset items")
    parser.add_argument("--max_samples", type=int, default=None, help="Max number of samples to load (useful for smoke tests)")
    parser.add_argument("--encoder_ckpt", type=str, default="checkpoints/basencoder_step_115000/student.pt", help="Path to frozen Phase 1 encoder checkpoint")
    parser.add_argument("--unet_ckpt", type=str, default=None, help="Path to frozen Phase 4 Unet checkpoint (defaults to auto-discovering latest)")
    parser.add_argument("--translator_ckpt", type=str, default=None, help="Path to frozen translator checkpoint (defaults to auto-discovering latest)")
    parser.add_argument("--eval_interval", type=int, default=200, help="Print qualitative Dual-Brain generation test every N steps")
    parser.add_argument("--euler_steps", type=int, default=10, help="Euler ODE integration steps for qualitative evaluation")
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["bf16", "fp16", "no"], help="Mixed precision mode (bf16, fp16, no)")
    parser.add_argument("--accum_steps", type=int, default=1, help="Number of gradient accumulation steps")
    parser.add_argument("--min_target_len", type=int, default=30, help="Filter out ultra-short target fragments (< N chars) to avoid 95%% padding dilution")
    parser.add_argument("--pad_loss_weight", type=float, default=0.05, help="Relative weight for <PAD> tokens in Mamba-HiPPO CFM MSE loss")
    parser.add_argument("--disable_multistream", action="store_true", help="Fallback to GenericPairDataset without multi-stream knowledge")
    parser.add_argument("--max_know_streams", type=int, default=4, help="Max number of parallel external knowledge streams M")
    parser.add_argument("--know_seq_len", type=int, default=128, help="Token length L_k per external knowledge stream")
    parser.add_argument("--num_beats", type=int, default=32, help="Number of macroscopic narrative beats K_beats")
    parser.add_argument("--mamba_layers", type=int, default=6, help="Number of stacked Mamba planner layers")
    parser.add_argument("--know_drop_prob", type=float, default=0.15, help="Curriculum knowledge stream dropout probability during training")
    parser.add_argument("--verbosity_rho", type=float, default=None, help="Optional float in (0, 1] for manual verbosity control during evaluation")
    parser.add_argument("--auto_resume", dest="auto_resume", action="store_true", help="Automatically find and resume from the latest checkpoint (alias for --continue)")
    parser.set_defaults(ckpt_prefix="phase5_mamba_flow", batch_size=64, save_steps=1000)

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

    # 1. Dataset Selection (Multi-Stream Knowledge Dataset with graceful fallback)
    if MultiStreamKnowledgeDataset is not None and not args.disable_multistream:
        dataset = MultiStreamKnowledgeDataset(
            data_path=args.data_path,
            tokenizer=tokenizer,
            seq_len=args.seq_len,
            max_know_streams=args.max_know_streams,
            know_seq_len=args.know_seq_len,
            prompt_key=args.prompt_key,
            target_key=args.target_key,
            max_samples=args.max_samples,
            min_target_len=args.min_target_len,
            verbose=True,
        )
    else:
        if not args.disable_multistream:
            print("⚠️ MultiStreamKnowledgeDataset not found; falling back to GenericPairDataset.")
        dataset = GenericPairDataset(
            data_path=args.data_path,
            tokenizer=tokenizer,
            seq_len=args.seq_len,
            prompt_key=args.prompt_key,
            target_key=args.target_key,
            max_samples=args.max_samples,
            min_target_len=args.min_target_len,
            verbose=True,
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

    # 3. Phase 4 Flow Match Unet (Frozen Motor Cortex + Calibrated Manifold Buffers)
    unet_path = args.unet_ckpt
    if unet_path and ("*" in unet_path or "?" in unet_path):
        matched = sorted(glob.glob(unet_path))
        unet_path = matched[-1] if matched else None
    if unet_path and os.path.isdir(unet_path):
        unet_path = os.path.join(unet_path, "unet.pt")
    if not unet_path:
        unet_path = find_latest_unet_ckpt(base_dirs=[args.out_dir, "checkpoints/run", "checkpoints"])

    unet = FlowMatching1DUnet(hidden_dim=1024, cond_dim=1024, num_layers=6).to(device)
    if unet_path and os.path.exists(unet_path):
        unet.load_state_dict(torch.load(unet_path, map_location=device, weights_only=True))
        scale_factor = 1.0 / unet.manifold_std.item()
        print(f"✅ Loaded frozen Phase 4 Unet from {unet_path} (Scale: {scale_factor:.2f}x, Mean: {unet.manifold_mean.item():.4f}, Std: {unet.manifold_std.item():.4f})")
    else:
        print(f"⚠️ WARNING: No Phase 4 Unet checkpoint found at '{unet_path}'. Unet will auto-calibrate manifold scale on first batch.")
    unet.eval()
    for p in unet.parameters():
        p.requires_grad = False

    # 4. Phase 4 Translator Lens (Frozen Continuous-to-Text Dictionary)
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

    # 5. Phase 4 HiPPO Oracle (Frozen)
    hippo_oracle = HiPPOOracleCUDA(hidden_dim=1024, N=64).to(device)
    hippo_oracle.eval()

    # 6. Phase 5 Multi-Stream Mamba Predictive Planner (Trainable)
    mamba = MultiStreamMambaPlanner(
        d_model=1024,
        num_layers=args.mamba_layers,
        num_beats=args.num_beats,
    ).to(device)
    optimizer = optim.AdamW(mamba.parameters(), lr=args.lr)

    checkpointer = Checkpointer(out_dir=args.out_dir, prefix=args.ckpt_prefix, keep_last_k=args.keep_last_k)
    checkpointer.register_model("mamba", mamba)
    checkpointer.register_optimizer("engine_opt", optimizer)
    checkpointer.register_args(args)

    if args.auto_resume:
        try:
            global_step = checkpointer.load_latest()
        except Exception as e:
            print(f"⚠️ Legacy or incompatible checkpoint detected ({e}). Starting fresh from step 0!")
            global_step = 0
            optimizer = optim.AdamW(mamba.parameters(), lr=args.lr)
            checkpointer.register_optimizer("engine_opt", optimizer)
    else:
        global_step = 0
    start_epoch = global_step // len(dataloader) if len(dataloader) > 0 else 0

    print(
        f"🚀 Starting Phase 5 Multi-Stream Mamba Planner Training "
        f"(Layers={args.mamba_layers}, Beats={args.num_beats}, MaxKnowStreams={args.max_know_streams})..."
    )

    optimizer.zero_grad(set_to_none=True)

    for epoch in range(start_epoch, args.epochs):
        resumed_loader = checkpointer.wrap_dataloader(dataloader, epoch, global_step)
        mamba.train()

        for batch_idx, batch_data in enumerate(resumed_loader):
            if len(batch_data) == 3:
                w_t, w_know, w_tgt = batch_data
                w_know = w_know.to(device)
            else:
                w_t, w_tgt = batch_data
                w_know = None

            w_t = w_t.to(device)
            w_tgt = w_tgt.to(device)
            context_mask = (w_t != tokenizer.pad_token_id)
            target_mask = (w_tgt != tokenizer.pad_token_id)

            with torch.no_grad():
                x_context = encode_tokens(encoder, w_t)
                x_target = encode_tokens(encoder, w_tgt)

                if w_know is not None:
                    x_knowledge, knowledge_mask = encode_knowledge_streams(
                        encoder, w_know, tokenizer.pad_token_id
                    )
                else:
                    x_knowledge, knowledge_mask = None, None

                if global_step == 0 and batch_idx == 0:
                    if int(unet.calib_steps.item()) == 0:
                        unet.calibrate_manifold(x_target, valid_mask=target_mask)
                    valid_tokens = x_target[target_mask] if target_mask.any() else x_target
                    std_val = valid_tokens.std().item()
                    mean_val = valid_tokens.mean().item()
                    norm_x = unet.normalize(x_target)
                    norm_std = (norm_x[target_mask] if target_mask.any() else norm_x).std().item()
                    scale_factor = 1.0 / unet.manifold_std.item()
                    ctx_valid_pct = context_mask.float().mean().item() * 100.0
                    tgt_valid_pct = target_mask.float().mean().item() * 100.0
                    know_active_pct = (
                        knowledge_mask.any(dim=-1).float().mean().item() * 100.0
                        if knowledge_mask is not None else 0.0
                    )
                    print(
                        f"📊 Manifold Geometry Check (Context Valid: {ctx_valid_pct:.1f}%, "
                        f"Target Valid: {tgt_valid_pct:.1f}%, Active Know Streams: {know_active_pct:.1f}%) "
                        f"-> Raw Mean: {mean_val:.4f}, Raw Valid Std: {std_val:.4f}"
                    )
                    print(f"⚖️ Manifold Scaler -> Scale: {scale_factor:.2f}x | Normalized Valid Std: {norm_std:.4f}")

            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                loss, metrics = mamba_cfm_loss(
                    mamba=mamba,
                    hippo_oracle=hippo_oracle,
                    x_context=x_context,
                    x_target=x_target,
                    unet=unet,
                    context_mask=context_mask,
                    target_mask=target_mask,
                    x_knowledge=x_knowledge,
                    knowledge_mask=knowledge_mask,
                    pad_loss_weight=args.pad_loss_weight,
                    know_drop_prob=args.know_drop_prob,
                    return_sequence=True,
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
                    torch.nn.utils.clip_grad_norm_(mamba.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(mamba.parameters(), 1.0)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            global_step += 1
            if global_step % 50 == 0:
                scale_factor = 1.0 / unet.manifold_std.item()
                print(
                    f"Epoch {epoch+1} | Step {global_step} | "
                    f"Total Loss: {loss.item():.4f} (Pot: {metrics['loss_potential'].item():.4f}, "
                    f"Len: {metrics['loss_len'].item():.4f}, "
                    f"Rho: {metrics['rho_pred_mean'].item():.2f}/{metrics['rho_true_mean'].item():.2f}) | "
                    f"Scale: {scale_factor:.2f}x"
                )

            # Periodic Dual-Brain End-to-End Qualitative Evaluation
            if translator is not None and unet is not None and (global_step == 50 or global_step % args.eval_interval == 0):
                mamba.eval()
                with torch.no_grad():
                    p_sample = x_context[:1]
                    p_mask_sample = context_mask[:1]
                    tgt_sample = x_target[:1]

                    x_context_norm = unet.normalize(p_sample)
                    if x_knowledge is not None:
                        k_sample = x_knowledge[:1]
                        k_mask_sample = knowledge_mask[:1]
                        x_knowledge_norm = unet.normalize(k_sample)
                    else:
                        x_knowledge_norm = None
                        k_mask_sample = None

                    with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                        h_potential, eval_aux = mamba(
                            x_context_norm,
                            valid_mask=p_mask_sample,
                            x_knowledge=x_knowledge_norm,
                            knowledge_mask=k_mask_sample,
                            target_len=args.seq_len,
                            rho_override=args.verbosity_rho,
                            return_sequence=True,
                            return_aux=True,
                        )

                        # Build fixed-length (1, seq_len, D) attractors matching Phase 4 Unet's training distribution
                        k_attractors_norm = build_fixed_length_attractors(
                            x_context_norm=x_context_norm,
                            context_mask=p_mask_sample,
                            x_knowledge_norm=x_knowledge_norm,
                            knowledge_mask=k_mask_sample,
                            seq_len=args.seq_len,
                        )

                        x_gen = sample_euler_ode(
                            unet=unet,
                            x_prompt=p_sample,
                            h_potential=h_potential,
                            k_attractors_norm=k_attractors_norm,
                            steps=args.euler_steps,
                            seq_len=args.seq_len,
                            device=device,
                        )

                    gen_tokens = translator.decode_greedy(x_gen.float())[0].tolist()
                    gen_text = decode_with_eos_marker(tokenizer, gen_tokens)

                    ref_tokens = translator.decode_greedy(tgt_sample.float())[0].tolist()
                    ref_text = decode_with_eos_marker(tokenizer, ref_tokens)

                    prompt_text = tokenizer.decode(w_t[0].tolist(), skip_special_tokens=True)

                    know_texts = []
                    if w_know is not None:
                        for m_idx in range(w_know.shape[1]):
                            m_ids = w_know[0, m_idx].tolist()
                            if any(tid != tokenizer.pad_token_id for tid in m_ids):
                                know_texts.append(tokenizer.decode(m_ids, skip_special_tokens=True))
                    know_summary = " | ".join(know_texts) if know_texts else "<None>"

                    rho_pred_val = eval_aux["rho_pred"][0].item()
                    rho_used_val = eval_aux["rho_used"][0].item()
                    est_chars = int(round(rho_used_val * args.seq_len))

                    print("\n" + "="*60)
                    print(f"🧠 [Step {global_step}] Phase 5 Dual-Brain Euler ODE Generation Test ({args.euler_steps} steps)")
                    print(f"Prompt:                      {prompt_text}")
                    print(f"Knowledge Streams (Active):  {know_summary}")
                    print(f"Pred Verbosity (rho -> len): rho_pred={rho_pred_val:.2f} (used={rho_used_val:.2f} -> ~{est_chars} chars)")
                    print(f"Target:                      {ref_text}")
                    print(f"Mamba+Unet Generated:        {gen_text}")
                    print("="*60 + "\n")

                mamba.train()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if args.save_steps and global_step % args.save_steps == 0:
                checkpointer.save(global_step)

        checkpointer.save(global_step)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    train()
