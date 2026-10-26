import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import sys
import math
import glob
import random
import inspect
from typing import Dict, List, Optional, Tuple
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
    from training.data.knowledge_dataset import (
        MultiStreamKnowledgeDataset,
        CANONICAL_DOMAINS,
        DOMAIN_TO_ID,
        ID_TO_DOMAIN,
    )
except ImportError:
    MultiStreamKnowledgeDataset = None
    CANONICAL_DOMAINS = [
        "grounded_instruct",
        "code_api",
        "classical_quote",
        "encyclopedia_rag",
        "math_rule",
        "literature_lore",
        "general",
    ]
    DOMAIN_TO_ID = {d: i for i, d in enumerate(CANONICAL_DOMAINS)}
    ID_TO_DOMAIN = {i: d for i, d in enumerate(CANONICAL_DOMAINS)}

from model.flow_mamba import MultiStreamMambaPlanner, MambaReader

from model.hippo_oracle import HiPPOOracleCUDA
from model.flow_unet_cuda import FlowMatching1DUnet
from model.loss_mamba_cfm import mamba_cfm_loss

from model.phase1_charencoder_cuda import TinyCharEncoderCUDA
from model.continuous_translator_cuda import ContinuousToTextTranslatorCUDA
from training.core.args import get_training_parser
from training.core.checkpoint import Checkpointer
from torch.utils.data import DataLoader


def get_warmup_cosine_scheduler(
    optimizer: optim.Optimizer,
    warmup_steps: int,
    max_steps: int,
    min_lr_ratio: float = 0.1,
) -> optim.lr_scheduler.LambdaLR:
    """
    Linear warmup from 0 -> 1 over `warmup_steps`, followed by cosine decay down to `min_lr_ratio`
    at `max_steps` (and clamped at `min_lr_ratio` beyond `max_steps`).
    """
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


def select_fixed_benchmark_indices(dataset, seed: int = 42, max_domains: int = 6) -> List[Tuple[str, int]]:
    """
    Deterministically selects 1 representative sample index per available domain (up to `max_domains`)
    using a fixed random seed (`42`) so qualitative ODE and validation comparisons are identical across runs.
    """
    rng = random.Random(seed)
    domain_to_indices: Dict[str, List[int]] = {}

    if hasattr(dataset, "samples") and isinstance(dataset.samples, list):
        for idx, rec in enumerate(dataset.samples):
            if isinstance(rec, tuple) and len(rec) >= 4:
                dom = str(rec[3] or "general")
                has_know = rec[2] is not None and len(rec[2]) > 0
            else:
                dom = "general"
                has_know = False
            domain_to_indices.setdefault(dom, []).append((idx, has_know))
    else:
        n = min(len(dataset), max_domains)
        return [("general", i) for i in range(n)]

    selected: List[Tuple[str, int]] = []
    ordered_domains = [d for d in CANONICAL_DOMAINS if d in domain_to_indices]
    for d in sorted(domain_to_indices.keys()):
        if d not in ordered_domains:
            ordered_domains.append(d)

    for dom in ordered_domains[:max_domains]:
        cands = domain_to_indices[dom]
        with_know = [idx for idx, hk in cands if hk]
        pool = with_know if with_know else [idx for idx, _ in cands]
        chosen_idx = rng.choice(pool)
        selected.append((dom, chosen_idx))

    return selected


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


def find_compatible_mamba_ckpt(mamba, out_dir, prefix="phase5_mamba_flow", device="cpu"):
    """
    Finds the latest checkpoint with matching architecture by testing state_dict compatibility
    in descending order of global_step. Skips incompatible legacy checkpoints.
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
                    import json
                    with open(os.path.join(d_path, file), "r") as f:
                        state = json.load(f)
                        step = state.get("global_step", -1)
                except Exception:
                    pass
        if step == -1:
            import re
            m = re.search(r'step_(\d+)', d)
            if m:
                step = int(m.group(1))
        if step >= 0:
            mamba_pt = os.path.join(d_path, "mamba.pt")
            if os.path.isfile(mamba_pt):
                candidates.append((step, d_path, d))

    candidates.sort(key=lambda x: x[0], reverse=True)

    for step, c_path, c_name in candidates:
        mamba_pt = os.path.join(c_path, "mamba.pt")
        try:
            state_dict = torch.load(mamba_pt, map_location=device, weights_only=True)
            mamba.load_state_dict(state_dict, strict=True)
            print(f"✅ Verified compatible Mamba checkpoint '{c_name}' (step {step})")
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
    attractor_mode: str = "full",
) -> Optional[torch.Tensor]:
    """
    Builds a fixed-length (B, seq_len, D) attractor tensor for Phase 4 FlowMatching1DUnet:
      - `attractor_mode == "none"`: returns `None`
      - `attractor_mode == "prompt_only"`: only normalized prompt tokens matching Phase 4's exact training distribution
      - `attractor_mode == "full"`: valid context tokens + active valid knowledge tokens padded to `seq_len`
    """
    if attractor_mode == "none":
        return None

    B, L_c, D = x_context_norm.shape
    if attractor_mode == "prompt_only" and L_c == seq_len:
        return x_context_norm

    pad_vec = x_context_norm[:, -1:, :].expand(B, seq_len, D).clone()
    attractor_buf = pad_vec.clone()

    for b in range(B):
        pieces = []
        if context_mask is not None and context_mask[b].any():
            pieces.append(x_context_norm[b][context_mask[b].bool()])
        else:
            pieces.append(x_context_norm[b])

        if attractor_mode == "full" and x_knowledge_norm is not None:
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
def sample_euler_ode(
    unet,
    x_prompt,
    h_potential=None,
    k_attractors_norm=None,
    mamba=None,
    macro_state=None,
    steps=10,
    seq_len=256,
    device='cuda',
    attractor_mode: str = "full",
):
    """
    Closed-Loop Two-Stage Euler ODE Integrator:
    - Stage 1 (`macro_state`) is computed ONCE outside the ODE loop via `mamba.encode_macro(...)`.
    - Stage 2 (`mamba.forward_canvas`) is called at each Euler step i in [0, steps-1] to dynamically
      articulate `h_potential(X_t, t)` and `v_mamba(X_t, t)` from the current trajectory state `x`,
      blending `v_pred = 0.8 * v_unet + 0.2 * v_mamba`.
    - Respects `attractor_mode` in `{"full", "prompt_only", "none"}`.
    """
    B = x_prompt.shape[0]
    D = x_prompt.shape[-1]
    if attractor_mode == "none":
        k_attr_used = None
    elif attractor_mode == "prompt_only":
        k_attr_used = unet.normalize(x_prompt)
    else:
        k_attr_used = k_attractors_norm if k_attractors_norm is not None else unet.normalize(x_prompt)

    x = torch.randn((B, seq_len, D), device=device, dtype=x_prompt.dtype)
    dt = 1.0 / steps
    for i in range(steps):
        t_val = i / float(steps)
        t_tensor = torch.full((B,), t_val, device=device, dtype=torch.float32)
        if mamba is not None and macro_state is not None:
            h_pot_t, v_mamba = mamba.forward_canvas(x, t_tensor, macro_state, return_velocity=True)
            v_unet = unet(x, t_tensor, h_potential=h_pot_t, k_attractors=k_attr_used)
            v_pred = 0.8 * v_unet + 0.2 * v_mamba
        else:
            v_pred = unet(x, t_tensor, h_potential=h_potential, k_attractors=k_attr_used)
        x = x + v_pred * dt
    return unet.denormalize(x)


@torch.no_grad()
def evaluate_validation_loss(
    mamba,
    hippo_oracle,
    unet,
    x_context: torch.Tensor,
    x_target: torch.Tensor,
    context_mask: Optional[torch.Tensor] = None,
    target_mask: Optional[torch.Tensor] = None,
    x_knowledge: Optional[torch.Tensor] = None,
    knowledge_mask: Optional[torch.Tensor] = None,
    pad_loss_weight: float = 0.05,
    t_val: float = 0.75,
) -> Dict[str, torch.Tensor]:
    """
    Evaluates validation loss and metrics on a sample/batch at fixed timestep `t_val` (default 0.75),
    explicitly passing `x_0=x_0` into `mamba(..., x_t=x_t, t=t, x_0=x_0, ...)` so `FlowMamba.forward`
    uses the exact `x_0` noise vector to form the self-conditioned prefix `x_t_prefix`.
    """
    B, L_t, D = x_target.shape
    device = x_target.device
    t_tensor = torch.full((B,), float(t_val), device=device, dtype=torch.float32)

    if unet is not None:
        x_ctx_norm = unet.normalize(x_context)
        x_tgt_norm = unet.normalize(x_target)
        x_know_norm = unet.normalize(x_knowledge) if x_knowledge is not None else None
    else:
        x_ctx_norm = x_context
        x_tgt_norm = x_target
        x_know_norm = x_knowledge

    h_target = hippo_oracle(x_tgt_norm, return_sequence=True)
    x_0 = torch.randn_like(h_target)
    t_expand = t_tensor.view(B, 1, 1).to(dtype=h_target.dtype)
    x_t = t_expand * h_target + (1.0 - t_expand) * x_0

    # Verify direct model call with explicit x_0=x_0
    _ = mamba(
        x_ctx_norm,
        valid_mask=context_mask,
        x_knowledge=x_know_norm,
        knowledge_mask=knowledge_mask,
        target_len=L_t,
        x_t=x_t,
        t=t_tensor,
        x_0=x_0,
        return_sequence=True,
        return_aux=True,
        return_velocity=True,
    )

    _, metrics = mamba_cfm_loss(
        mamba=mamba,
        hippo_oracle=hippo_oracle,
        x_context=x_context,
        x_target=x_target,
        unet=unet,
        context_mask=context_mask,
        target_mask=target_mask,
        x_knowledge=x_knowledge,
        knowledge_mask=knowledge_mask,
        pad_loss_weight=pad_loss_weight,
        know_drop_prob=0.0,
        verbosity_hint_prob=0.0,
        t=t_tensor,
        return_sequence=True,
        return_metrics=True,
    )
    return metrics


@torch.no_grad()
def run_attractor_ablation_benchmark(
    mamba,
    unet,
    encoder,
    hippo_oracle,
    translator,
    tokenizer,
    dataset,
    benchmark_samples: List[Tuple[str, int]],
    args,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
):
    """
    Runs fixed cross-domain benchmark samples under all 3 attractor modes (`full`, `prompt_only`, `none`),
    prints a side-by-side comparison table of `||x_gen - h_potential||`, `||x_gen - x_target||`,
    and decoded text for each mode.
    """
    mamba.eval()
    unet.eval()
    print("\n" + "=" * 88)
    print("🔬 [Attractor Mode Ablation Benchmark] Comparing 'full' vs 'prompt_only' vs 'none'")
    print("=" * 88)

    for dom_name, sample_idx in benchmark_samples:
        item = dataset[sample_idx]
        if len(item) >= 3:
            w_t, w_know, w_tgt = item[0], item[1], item[2]
            w_know = w_know.unsqueeze(0).to(device)
        else:
            w_t, w_tgt = item[0], item[1]
            w_know = None

        w_t = w_t.unsqueeze(0).to(device)
        w_tgt = w_tgt.unsqueeze(0).to(device)
        ctx_mask = (w_t != tokenizer.pad_token_id)
        tgt_mask = (w_tgt != tokenizer.pad_token_id)

        x_ctx = encode_tokens(encoder, w_t)
        x_tgt = encode_tokens(encoder, w_tgt)
        if w_know is not None:
            x_know, know_mask = encode_knowledge_streams(encoder, w_know, tokenizer.pad_token_id)
        else:
            x_know, know_mask = None, None

        if int(unet.calib_steps.item()) == 0:
            unet.calibrate_manifold(x_tgt, valid_mask=tgt_mask)

        x_ctx_norm = unet.normalize(x_ctx)
        x_tgt_norm = unet.normalize(x_tgt)
        x_know_norm = unet.normalize(x_know) if x_know is not None else None
        h_target = hippo_oracle(x_tgt_norm, return_sequence=True)

        prompt_str = tokenizer.decode(w_t[0].tolist(), skip_special_tokens=True)
        know_texts = []
        if w_know is not None:
            for m_i in range(w_know.shape[1]):
                m_ids = w_know[0, m_i].tolist()
                if any(tid != tokenizer.pad_token_id for tid in m_ids):
                    know_texts.append(tokenizer.decode(m_ids, skip_special_tokens=True))
        know_summary = " | ".join(know_texts) if know_texts else "<None>"
        ref_text = (
            decode_with_eos_marker(tokenizer, translator.decode_greedy(x_tgt.float())[0].tolist())
            if translator is not None
            else tokenizer.decode(w_tgt[0].tolist(), skip_special_tokens=True)
        )

        print(f"\n📌 Domain: [{dom_name}] (Sample #{sample_idx})")
        print(f"   Prompt:    {prompt_str[:120]}")
        print(f"   Knowledge: {know_summary[:140]}")
        print(f"   Target:    {ref_text[:140]}")
        print(f"   {'Mode':<14} | {'||x_gen_norm - h_pot||':<22} | {'||x_gen_norm - h_tgt||':<22} | Generated Preview")
        print("   " + "-" * 84)

        for mode in ["full", "prompt_only", "none"]:
            torch.manual_seed(42)
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                macro_state = mamba.encode_macro(
                    x_context=x_ctx_norm,
                    valid_mask=ctx_mask,
                    x_knowledge=x_know_norm,
                    knowledge_mask=know_mask,
                    target_len=args.seq_len,
                    rho_override=args.verbosity_rho,
                )
                h_pot_final, _ = mamba.forward_canvas(
                    macro_state["h_macro_potential"].detach(),
                    torch.ones((1,), device=device, dtype=torch.float32),
                    macro_state,
                    return_velocity=True,
                    x_t_prefix=macro_state["h_macro_potential"].detach(),
                )
                k_attr = build_fixed_length_attractors(
                    x_context_norm=x_ctx_norm,
                    context_mask=ctx_mask,
                    x_knowledge_norm=x_know_norm,
                    knowledge_mask=know_mask,
                    seq_len=args.seq_len,
                    attractor_mode=mode,
                )
                x_gen = sample_euler_ode(
                    unet=unet,
                    x_prompt=x_ctx,
                    h_potential=macro_state["h_macro_potential"],
                    k_attractors_norm=k_attr,
                    mamba=mamba,
                    macro_state=macro_state,
                    steps=args.euler_steps,
                    seq_len=args.seq_len,
                    device=device,
                    attractor_mode=mode,
                )

            x_gen_norm = unet.normalize(x_gen.float())
            valid_f = tgt_mask.float().unsqueeze(-1)
            denom = (valid_f.sum() * x_gen_norm.shape[-1]).clamp(min=1.0)
            dist_pot = torch.sqrt(((x_gen_norm - h_pot_final.float()) ** 2 * valid_f).sum() / denom).item()
            dist_tgt = torch.sqrt(((x_gen_norm - h_target.float()) ** 2 * valid_f).sum() / denom).item()

            if translator is not None:
                gen_tokens = translator.decode_greedy(x_gen.float())[0].tolist()
                gen_text = decode_with_eos_marker(tokenizer, gen_tokens)
            else:
                gen_text = "<Translator not loaded>"

            print(f"   {mode:<14} | {dist_pot:<22.4f} | {dist_tgt:<22.4f} | {gen_text[:80]}")

    print("=" * 88 + "\n")


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
    parser.add_argument("--min_target_len", type=int, default=15, help="Filter out ultra-short target fragments (< N chars)")
    parser.add_argument("--pad_loss_weight", type=float, default=0.05, help="Relative weight for <PAD> tokens in Mamba-HiPPO CFM MSE loss")
    parser.add_argument("--disable_multistream", action="store_true", help="Fallback to GenericPairDataset without multi-stream knowledge")
    parser.add_argument("--max_know_streams", type=int, default=4, help="Max number of parallel external knowledge streams M")
    parser.add_argument("--know_seq_len", type=int, default=128, help="Token length L_k per external knowledge stream")
    parser.add_argument("--num_beats", type=int, default=32, help="Number of macroscopic narrative beats K_beats")
    parser.add_argument("--mamba_layers", type=int, default=6, help="Number of stacked Mamba planner layers")
    parser.add_argument("--know_drop_prob", type=float, default=0.15, help="Curriculum knowledge stream dropout probability during training")
    parser.add_argument("--verbosity_rho", type=float, default=None, help="Optional float in (0, 1] for manual verbosity control during evaluation")
    parser.add_argument("--auto_resume", dest="auto_resume", action="store_true", help="Automatically find and resume from the latest checkpoint (alias for --continue)")
    parser.add_argument("--max_steps", type=int, default=50000, help="Cosine LR decay horizon (if > 0, else epochs * len(dataloader))")
    parser.add_argument("--min_lr_ratio", type=float, default=0.1, help="Minimum LR ratio at end of cosine decay as fraction of --lr")
    parser.add_argument("--ema_decay", type=float, default=0.98, help="EMA smoothing factor for training logs")
    parser.add_argument(
        "--active_domains",
        type=str,
        default="grounded_instruct,encyclopedia_rag,code_api,classical_quote,math_rule",
        help="Comma-separated active domains to load (default excludes ungrounded literature_lore; pass 'all' for all domains)",
    )
    parser.add_argument(
        "--attractor_mode",
        type=str,
        default="full",
        choices=["full", "prompt_only", "none"],
        help="Controls k_attractors passed to Phase 4 unet during Euler ODE sampling",
    )
    parser.add_argument(
        "--run_attractor_ablation",
        action="store_true",
        help="Run fixed cross-domain benchmark under all 3 attractor modes (full, prompt_only, none) and exit",
    )
    parser.set_defaults(ckpt_prefix="phase5_mamba_flow", batch_size=64, save_steps=1000, warmup_steps=1000)

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

    # 1. Dataset Selection (Multi-Stream Knowledge Dataset with return_domain_id=True and active_domains)
    if MultiStreamKnowledgeDataset is not None and not args.disable_multistream:
        sig = inspect.signature(MultiStreamKnowledgeDataset.__init__)
        ds_kwargs = dict(
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
        if "return_domain_id" in sig.parameters:
            ds_kwargs["return_domain_id"] = True
        if "active_domains" in sig.parameters:
            ds_kwargs["active_domains"] = args.active_domains
        dataset = MultiStreamKnowledgeDataset(**ds_kwargs)
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

    # Select fixed cross-domain benchmark samples once (deterministic seed 42)
    fixed_benchmark_samples = select_fixed_benchmark_indices(dataset, seed=42, max_domains=6)
    if fixed_benchmark_samples:
        bench_desc = ", ".join(f"{dom}:#{idx}" for dom, idx in fixed_benchmark_samples)
        print(f"📌 Fixed Cross-Domain Benchmark Samples (seed=42): {bench_desc}")

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

    total_horizon_steps = args.max_steps if args.max_steps > 0 else max(1, args.epochs * len(dataloader))
    scheduler = get_warmup_cosine_scheduler(
        optimizer=optimizer,
        warmup_steps=args.warmup_steps,
        max_steps=total_horizon_steps,
        min_lr_ratio=args.min_lr_ratio,
    )

    checkpointer = Checkpointer(out_dir=args.out_dir, prefix=args.ckpt_prefix, keep_last_k=args.keep_last_k)
    checkpointer.register_model("mamba", mamba)
    checkpointer.register_optimizer("engine_opt", optimizer)
    checkpointer.register_optimizer("lr_sched", scheduler)
    checkpointer.register_args(args)

    if args.auto_resume or args.resume_from:
        if args.resume_from and os.path.isdir(args.resume_from):
            compat_path, compat_step = args.resume_from, 0
        else:
            compat_path, compat_step = find_compatible_mamba_ckpt(
                mamba=mamba,
                out_dir=args.out_dir,
                prefix=args.ckpt_prefix,
                device=device,
            )
        if compat_path:
            try:
                loaded_step = checkpointer.load(compat_path)
                global_step = loaded_step if loaded_step > 0 else compat_step
                if hasattr(checkpointer, "set_step"):
                    checkpointer.set_step(global_step)
                else:
                    checkpointer.current_step = global_step
                sched_file = os.path.join(compat_path, "lr_sched.pt")
                if not os.path.exists(sched_file) and global_step > 0:
                    scheduler.step(global_step)
                print(f"✅ Successfully resumed training from step {global_step} ({compat_path})")
            except Exception as e:
                print(f"⚠️ Error loading checkpoint state ({e}). Starting fresh from step 0!")
                global_step = 0
                if hasattr(checkpointer, "set_step"):
                    checkpointer.set_step(0)
                else:
                    checkpointer.current_step = 0
                optimizer = optim.AdamW(mamba.parameters(), lr=args.lr)
                scheduler = get_warmup_cosine_scheduler(
                    optimizer=optimizer,
                    warmup_steps=args.warmup_steps,
                    max_steps=total_horizon_steps,
                    min_lr_ratio=args.min_lr_ratio,
                )
                checkpointer.register_optimizer("engine_opt", optimizer)
                checkpointer.register_optimizer("lr_sched", scheduler)
        else:
            print(f"ℹ️ No compatible checkpoints found in '{args.out_dir}'. Starting fresh from step 0!")
            global_step = 0
            if hasattr(checkpointer, "set_step"):
                checkpointer.set_step(0)
            else:
                checkpointer.current_step = 0
    else:
        global_step = 0
        if hasattr(checkpointer, "set_step"):
            checkpointer.set_step(0)
        else:
            checkpointer.current_step = 0

    if args.run_attractor_ablation:
        run_attractor_ablation_benchmark(
            mamba=mamba,
            unet=unet,
            encoder=encoder,
            hippo_oracle=hippo_oracle,
            translator=translator,
            tokenizer=tokenizer,
            dataset=dataset,
            benchmark_samples=fixed_benchmark_samples,
            args=args,
            device=device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
        )
        return

    start_epoch = global_step // len(dataloader) if len(dataloader) > 0 else 0

    print(
        f"🚀 Starting Phase 5 Closed-Loop Multi-Stream Mamba Planner Training "
        f"(Layers={args.mamba_layers}, Beats={args.num_beats}, MaxKnowStreams={args.max_know_streams}, "
        f"Warmup={args.warmup_steps}, MaxSteps={total_horizon_steps}, AttractorMode={args.attractor_mode}, "
        f"ActiveDomains={args.active_domains})..."
    )

    optimizer.zero_grad(set_to_none=True)

    # EMA & 50-step Window Accumulators
    ema_state: Dict[str, Optional[float]] = {
        "total": None,
        "pot": None,
        "pot_raw": None,
        "cfm": None,
        "macro": None,
        "cos": None,
        "len": None,
        "copy_prob": None,
    }
    alpha_ema = float(args.ema_decay)

    def _update_ema(key: str, val: float) -> float:
        if math.isnan(val) or math.isinf(val):
            return ema_state[key] if ema_state[key] is not None else 0.0
        if ema_state[key] is None:
            ema_state[key] = val
        else:
            ema_state[key] = alpha_ema * ema_state[key] + (1.0 - alpha_ema) * val
        return ema_state[key]

    win_bucket_sum = {"low": 0.0, "mid": 0.0, "high": 0.0}
    win_bucket_cnt = {"low": 0, "mid": 0, "high": 0}
    win_domain_sum: Dict[int, float] = {}
    win_domain_cnt: Dict[int, int] = {}
    eval_rotation_counter = 0

    for epoch in range(start_epoch, args.epochs):
        resumed_loader = checkpointer.wrap_dataloader(dataloader, epoch, global_step)
        mamba.train()

        for batch_idx, batch_data in enumerate(resumed_loader):
            domain_ids = None
            if len(batch_data) == 4:
                w_t, w_know, w_tgt, domain_ids = batch_data
                w_know = w_know.to(device)
                domain_ids = domain_ids.to(device)
            elif len(batch_data) == 3:
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

                if batch_idx == 0:
                    if int(unet.calib_steps.item()) == 0:
                        unet.calibrate_manifold(x_target, valid_mask=target_mask)

                    # Calibrate Mamba's <PAD> tail attractor from the frozen Phase 1 encoder's <PAD> coordinate
                    if float(mamba.expander.pad_tail_emb.detach().abs().max().item()) < 1e-6:
                        pad_ids = torch.full((1, 4), tokenizer.pad_token_id, dtype=torch.long, device=device)
                        pad_raw = encode_tokens(encoder, pad_ids)[:, -1:, :]
                        pad_norm = unet.normalize(pad_raw)
                        mamba.expander.pad_tail_emb.data.copy_(pad_norm.to(dtype=mamba.expander.pad_tail_emb.dtype))

                if global_step == 0 and batch_idx == 0:
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
                scheduler.step()

            global_step += 1
            if hasattr(checkpointer, "set_step"):
                checkpointer.set_step(global_step)
            else:
                checkpointer.current_step = global_step

            # Update EMA and Window Trackers
            raw_total = loss.item()
            raw_pot = metrics["loss_potential"].item()
            raw_pot_unweighted = (
                metrics["loss_potential_raw"].item()
                if "loss_potential_raw" in metrics
                else raw_pot
            )
            raw_cfm = metrics["loss_cfm"].item() if "loss_cfm" in metrics else 0.0
            raw_macro = metrics["loss_macro"].item() if "loss_macro" in metrics else 0.0
            raw_cos = metrics["loss_cos"].item() if "loss_cos" in metrics else 0.0
            raw_len = metrics["loss_len"].item() if "loss_len" in metrics else 0.0
            raw_copy = metrics["copy_prob_mean"].item() if "copy_prob_mean" in metrics else 0.0

            ema_total = _update_ema("total", raw_total)
            ema_pot = _update_ema("pot", raw_pot)
            ema_pot_raw = _update_ema("pot_raw", raw_pot_unweighted)
            ema_cfm = _update_ema("cfm", raw_cfm)
            ema_macro = _update_ema("macro", raw_macro)
            ema_cos = _update_ema("cos", raw_cos)
            ema_len = _update_ema("len", raw_len)
            ema_copy = _update_ema("copy_prob", raw_copy)

            for b_key, m_key in [("low", "pot_t_low"), ("mid", "pot_t_mid"), ("high", "pot_t_high")]:
                if m_key in metrics:
                    v = metrics[m_key].item()
                    if not math.isnan(v) and v >= 0.0:
                        win_bucket_sum[b_key] += v
                        win_bucket_cnt[b_key] += 1

            if domain_ids is not None and "per_sample_pot" in metrics:
                ps_pot = metrics["per_sample_pot"].detach().cpu().tolist()
                dom_list = domain_ids.detach().cpu().tolist()
                for d_id, p_val in zip(dom_list, ps_pot):
                    win_domain_sum[d_id] = win_domain_sum.get(d_id, 0.0) + float(p_val)
                    win_domain_cnt[d_id] = win_domain_cnt.get(d_id, 0) + 1

            if global_step % 50 == 0:
                scale_factor = 1.0 / unet.manifold_std.item()
                cur_lr = optimizer.param_groups[0]["lr"]

                def _fmt_bucket(k: str) -> str:
                    if win_bucket_cnt[k] > 0:
                        return f"{win_bucket_sum[k] / win_bucket_cnt[k]:.4f}"
                    return "n/a"

                print(
                    f"Epoch {epoch+1} | Step {global_step} | LR: {cur_lr:.2e} | "
                    f"EMA Loss: {ema_total:.4f} (Raw: {raw_total:.4f}) | "
                    f"Pot: {ema_pot:.4f} (Raw: {ema_pot_raw:.4f}) | "
                    f"CFM: {ema_cfm:.4f} | Macro: {ema_macro:.4f} | Cos: {ema_cos:.4f} | "
                    f"Len: {ema_len:.4f} | CopyProb: {ema_copy:.3f} | "
                    f"Rho: {metrics['rho_pred_mean'].item():.2f}/{metrics['rho_true_mean'].item():.2f} | "
                    f"Scale: {scale_factor:.2f}x"
                )
                print(
                    f"   ├─ t-Buckets Pot MSE: t<0.3: {_fmt_bucket('low')} | "
                    f"0.3≤t<0.7: {_fmt_bucket('mid')} | "
                    f"t≥0.7: {_fmt_bucket('high')}"
                )
                if win_domain_cnt:
                    dom_parts = []
                    for d_id in sorted(win_domain_cnt.keys()):
                        d_name = ID_TO_DOMAIN.get(d_id, f"dom_{d_id}")
                        d_mean = win_domain_sum[d_id] / max(1, win_domain_cnt[d_id])
                        dom_parts.append(f"{d_name}: {d_mean:.4f}")
                    print(f"   └─ Domain Pot MSE:    {' | '.join(dom_parts)}")

                # Reset 50-step window accumulators
                win_bucket_sum = {"low": 0.0, "mid": 0.0, "high": 0.0}
                win_bucket_cnt = {"low": 0, "mid": 0, "high": 0}
                win_domain_sum.clear()
                win_domain_cnt.clear()

            # Periodic Dual-Brain Closed-Loop End-to-End Fixed Cross-Domain Evaluation
            if unet is not None and (global_step == 50 or global_step % args.eval_interval == 0):
                mamba.eval()
                with torch.no_grad():
                    # 1. Compute mean validation pot_mse and copy_prob across fixed cross-domain benchmark samples
                    val_pot_list = []
                    val_copy_list = []
                    bench_records = []

                    for dom_name, s_idx in fixed_benchmark_samples:
                        s_item = dataset[s_idx]
                        if len(s_item) >= 3:
                            b_wt, b_wknow, b_wtgt = s_item[0], s_item[1], s_item[2]
                            b_wknow = b_wknow.unsqueeze(0).to(device)
                        else:
                            b_wt, b_wtgt = s_item[0], s_item[1]
                            b_wknow = None

                        b_wt = b_wt.unsqueeze(0).to(device)
                        b_wtgt = b_wtgt.unsqueeze(0).to(device)
                        b_ctx_mask = (b_wt != tokenizer.pad_token_id)
                        b_tgt_mask = (b_wtgt != tokenizer.pad_token_id)

                        b_x_ctx = encode_tokens(encoder, b_wt)
                        b_x_tgt = encode_tokens(encoder, b_wtgt)
                        if b_wknow is not None:
                            b_x_know, b_know_mask = encode_knowledge_streams(
                                encoder, b_wknow, tokenizer.pad_token_id
                            )
                        else:
                            b_x_know, b_know_mask = None, None

                        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                            b_metrics = evaluate_validation_loss(
                                mamba=mamba,
                                hippo_oracle=hippo_oracle,
                                unet=unet,
                                x_context=b_x_ctx,
                                x_target=b_x_tgt,
                                context_mask=b_ctx_mask,
                                target_mask=b_tgt_mask,
                                x_knowledge=b_x_know,
                                knowledge_mask=b_know_mask,
                                pad_loss_weight=args.pad_loss_weight,
                                t_val=0.75,
                            )
                        val_pot_list.append(b_metrics["loss_potential_raw"].item())
                        val_copy_list.append(b_metrics["copy_prob_mean"].item())
                        bench_records.append((
                            dom_name, s_idx, b_wt, b_wknow, b_wtgt,
                            b_ctx_mask, b_tgt_mask, b_x_ctx, b_x_tgt, b_x_know, b_know_mask
                        ))

                    mean_val_pot = sum(val_pot_list) / max(1, len(val_pot_list))
                    mean_val_copy = sum(val_copy_list) / max(1, len(val_copy_list))

                    print("\n" + "=" * 72)
                    print(
                        f"🧠 [Step {global_step}] Fixed Cross-Domain Benchmark "
                        f"(Val Pot MSE @ t=0.75: {mean_val_pot:.4f} | Val CopyProb: {mean_val_copy:.3f} | "
                        f"AttractorMode: {args.attractor_mode})"
                    )

                    # Pick up to 2 samples for Euler ODE generation: anchor domain (code_api / grounded_instruct) + 1 rotating domain
                    if translator is not None and bench_records:
                        ode_indices = [0]
                        for i_r, rec_r in enumerate(bench_records):
                            if rec_r[0] in ("code_api", "grounded_instruct"):
                                ode_indices[0] = i_r
                                break
                        if len(bench_records) > 1:
                            rot_idx = eval_rotation_counter % len(bench_records)
                            if rot_idx == ode_indices[0]:
                                rot_idx = (rot_idx + 1) % len(bench_records)
                            ode_indices.append(rot_idx)
                        eval_rotation_counter += 1

                        for sel_i in ode_indices:
                            (
                                dom_name, s_idx, b_wt, b_wknow, b_wtgt,
                                b_ctx_mask, b_tgt_mask, b_x_ctx, b_x_tgt, b_x_know, b_know_mask
                            ) = bench_records[sel_i]

                            x_context_norm = unet.normalize(b_x_ctx)
                            x_knowledge_norm = unet.normalize(b_x_know) if b_x_know is not None else None

                            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                                macro_state = mamba.encode_macro(
                                    x_context=x_context_norm,
                                    valid_mask=b_ctx_mask,
                                    x_knowledge=x_knowledge_norm,
                                    knowledge_mask=b_know_mask,
                                    target_len=args.seq_len,
                                    rho_override=args.verbosity_rho,
                                )
                                eval_aux = macro_state["aux_dict"]
                                h_potential = macro_state["h_macro_potential"]

                                k_attractors_norm = build_fixed_length_attractors(
                                    x_context_norm=x_context_norm,
                                    context_mask=b_ctx_mask,
                                    x_knowledge_norm=x_knowledge_norm,
                                    knowledge_mask=b_know_mask,
                                    seq_len=args.seq_len,
                                    attractor_mode=args.attractor_mode,
                                )

                                x_gen = sample_euler_ode(
                                    unet=unet,
                                    x_prompt=b_x_ctx,
                                    h_potential=h_potential,
                                    k_attractors_norm=k_attractors_norm,
                                    mamba=mamba,
                                    macro_state=macro_state,
                                    steps=args.euler_steps,
                                    seq_len=args.seq_len,
                                    device=device,
                                    attractor_mode=args.attractor_mode,
                                )

                            gen_tokens = translator.decode_greedy(x_gen.float())[0].tolist()
                            gen_text = decode_with_eos_marker(tokenizer, gen_tokens)

                            ref_tokens = translator.decode_greedy(b_x_tgt.float())[0].tolist()
                            ref_text = decode_with_eos_marker(tokenizer, ref_tokens)

                            prompt_text = tokenizer.decode(b_wt[0].tolist(), skip_special_tokens=True)

                            know_texts = []
                            if b_wknow is not None:
                                for m_idx in range(b_wknow.shape[1]):
                                    m_ids = b_wknow[0, m_idx].tolist()
                                    if any(tid != tokenizer.pad_token_id for tid in m_ids):
                                        know_texts.append(tokenizer.decode(m_ids, skip_special_tokens=True))
                            know_summary = " | ".join(know_texts) if know_texts else "<None>"

                            rho_pred_val = eval_aux["rho_pred"][0].item()
                            rho_used_val = eval_aux["rho_used"][0].item()
                            copy_prob_val = (
                                eval_aux["copy_prob"][0].item()
                                if "copy_prob" in eval_aux
                                else 0.0
                            )
                            est_chars = int(round(rho_used_val * args.seq_len))

                            print("-" * 72)
                            print(f"Domain [{dom_name}] (Sample #{s_idx}) | CopyProb: {copy_prob_val:.3f}")
                            print(f"Prompt:                      {prompt_text}")
                            print(f"Knowledge Streams (Active):  {know_summary}")
                            print(f"Pred Verbosity (rho -> len): rho_pred={rho_pred_val:.2f} (used={rho_used_val:.2f} -> ~{est_chars} chars)")
                            print(f"Target:                      {ref_text}")
                            print(f"Mamba+Unet Generated:        {gen_text}")

                    print("=" * 72 + "\n")

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
