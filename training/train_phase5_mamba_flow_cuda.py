# =====================================================================================================
# [Design Change Documentation - Task #029: Right-Brain v2 Trainer & Dual-Brain Rollout Integration]
# 1. What changed vs #027:
#    - Model API: `MultiStreamRightBrain` v2 (1 main + N isomorphic companion streams, 32-band adaptive
#      prior cone, K-branch soft-competition flow matcher, stop head). The trainer now packs companions
#      via `pack_aux_streams(z_goal, goal_valid, z_cand, cand_valid)` (stream 0 = goal sequence,
#      streams 1..C = each candidate clause) instead of hand-building a K+-ordered `z_knowledge_seq`
#      (that #027 path leaked the K+ labels into the right brain's input ordering).
#    - Stream dropout (§2.7): per-sample group 0=full (p_full) / 1=goal_only (p_goal_only) / 2=none,
#      implemented by AND-ing `aux_valid`. `--active_streams discourse` forces group 2 for every sample.
#    - Exposure-bias robustness (§2.6): high-band norm-preserving jitter of the main-stream input and
#      scheduled self-feeding (p_self linearly ramped 0 -> p_self_max over
#      [self_feed_start, self_feed_start + self_feed_ramp]) using no_grad teacher-forced samples.
#    - KnowledgeSelector moved AFTER the right brain (§2.5): per-clause soft-label selection loss with
#      noisy ground-truth clause probes (`cand_clause_rel`) + the old sample-level goal-probe loss as
#      an auxiliary term. Rollout probes the selector with `z_thought` at EVERY clause step.
#    - Stop head (§2.4): trained from `discourse_stop_mask`; rollout stops when stop_prob > threshold.
#    - Observability (§3): sigma_sem / sigma_syn overall and per stream group, 32-band sigma curve,
#      winner histogram, branch diversity, gates, K+ vs K- gate means, multi-sample diversity demo.
# 2. Why: the design (design/tentative/2026-11-02_phase5_right_brain_v2_adaptive_prior.md) requires the
#    right brain to emit a REGION of plausible next clauses whose semantic width is controlled by how
#    decisive the (main + companion) evidence is; the trainer must therefore expose all conditions
#    (full / goal_only / none) and log the sigma statistics needed to verify "certainty-controlled range".
# 3. Caveats for subsequent development:
#    - NEW default checkpoint prefix `bicameral_right_brain_v2` (v2 is state-dict incompatible with #027;
#      `find_compatible_right_brain_ckpt` strict-loads and skips old checkpoints anyway). The old Colab
#      notebook Cell 4 passes `--ckpt_prefix bicameral_right_brain_s32` and MUST be updated, otherwise the
#      v2 run will write into / try to resume from the #027 namespace.
#    - Imports use the FINAL module names (`model.bicameral_right_brain`,
#      `training.data.knowledge_dataset`); the #027 `_027` fallbacks were removed on purpose.
#    - Legacy flags still parse: `--n_layers_discourse`/`--mamba_layers` -> `--n_layers_main`,
#      `--n_layers_goal`/`--n_layers_knowledge` -> `--n_layers_aux`, `--selector_weight` ->
#      `--sample_selector_weight`, `--max_know_streams` -> `--top_m_clauses`.
#    - `--sigma_max` defaults to 3.0 here (model default 1.0) because low bands were observed to saturate
#      at 1.0; the trainer warns if mean low-band sigma > 0.95 * sigma_max.
#    - Self-feeding only starts after the flow matcher's z calibration has run (calib_steps > 0).
#    - Under `--active_streams discourse` every eval condition is equivalent to `none`, so the eval
#      collapses the condition list to `none` only.
# =====================================================================================================
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import sys
import re
import math
import glob
import json
import time
import random
from typing import Dict, List, Optional, Tuple, Sequence, Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir in sys.path and os.path.exists(os.path.join(script_dir, "model.py")):
    sys.path = [p for p in sys.path if p != script_dir] + [script_dir]

from training.core.char_tokenizer import CharTokenizer
from training.core.args import get_training_parser
from training.core.checkpoint import Checkpointer

from model.phase1_charencoder_cuda import TinyCharEncoderCUDA, load_mlx_safetensors_into_torch
from model.continuous_translator_cuda import ContinuousToTextTranslatorCUDA
from model.hippo_oracle import HiPPOOracleCUDA
from model.flow_unet_cuda import LeftBrainCrystallizer

# #029 / #059: draft module names during verification (merger restores canonical imports).
from model.bicameral_right_brain import (
    MultiStreamRightBrain,
    KnowledgeSelector,
)
from training.data.knowledge_dataset import (
    BicameralClauseChainDataset,
    create_clause_chain_dataloader,
    CANONICAL_DOMAINS,
)

STREAM_GROUPS: Dict[str, int] = {"full": 0, "goal_only": 1, "none": 2}
GROUP_NAMES: Dict[int, str] = {v: k for k, v in STREAM_GROUPS.items()}


def sample_stream_groups(
    batch_size: int,
    p_full: float,
    p_goal_only: float,
    force_none: bool,
    device: torch.device,
) -> torch.Tensor:
    """
    Samples the per-sample stream-dropout group `(B,)` long: 0=full (p_full), 1=goal_only (p_goal_only),
    2=none (remainder). `force_none=True` (single-stream warmup) returns all 2.
    """
    if force_none:
        return torch.full((batch_size,), STREAM_GROUPS["none"], device=device, dtype=torch.long)
    p_f = min(max(float(p_full), 0.0), 1.0)
    p_g = min(max(float(p_goal_only), 0.0), 1.0 - p_f)
    u = torch.rand((batch_size,), device=device)
    group = torch.full((batch_size,), STREAM_GROUPS["none"], device=device, dtype=torch.long)
    group = torch.where(u < p_f + p_g, torch.full_like(group, STREAM_GROUPS["goal_only"]), group)
    group = torch.where(u < p_f, torch.full_like(group, STREAM_GROUPS["full"]), group)
    return group


def apply_stream_group_mask(aux_valid: Optional[torch.Tensor], group: torch.Tensor) -> Optional[torch.Tensor]:
    """
    Applies stream dropout to `aux_valid: (B, N, S)` (stream 0 = goal, streams 1.. = candidates):
      group 1 (goal_only) -> candidate streams masked; group 2 (none) -> every stream masked.
    """
    if aux_valid is None:
        return None
    B, N, _ = aux_valid.shape
    keep = torch.ones((B, N), device=aux_valid.device, dtype=torch.bool)
    if N > 1:
        keep[:, 1:] = (group == STREAM_GROUPS["full"]).unsqueeze(-1).expand(B, N - 1)
    keep[:, 0] = group != STREAM_GROUPS["none"]
    return aux_valid.bool() & keep.unsqueeze(-1)


def compute_p_self(step: int, start: int, ramp: int, p_max: float) -> float:
    """Linear scheduled self-feeding ramp: 0 before `start`, -> `p_max` at `start + ramp`, constant after."""
    if p_max <= 0.0 or step < start:
        return 0.0
    if ramp <= 0:
        return float(p_max)
    return float(p_max) * min(1.0, float(step - start) / float(ramp))


def _fmt_opt(v: Optional[float], fmt: str = "{:.3f}") -> str:
    return "n/a" if v is None else fmt.format(v)


def check_low_band_saturation(sigma_band: Sequence[float], sem_bands: int, sigma_max: float, tag: str = "") -> None:
    """Prints a WARNING if the mean low-band (semantic) sigma is within 5% of the `sigma_max` ceiling."""
    if not sigma_band:
        return
    low = list(sigma_band)[: max(1, int(sem_bands))]
    low_mean = sum(low) / len(low)
    if low_mean > 0.95 * float(sigma_max):
        print(
            f"⚠️ WARNING{(' [' + tag + ']') if tag else ''}: mean low-band sigma {low_mean:.3f} > 0.95 * sigma_max "
            f"({sigma_max:.2f}) — the prior is saturating; consider raising --sigma_max."
        )


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


def select_fixed_benchmark_indices(
    dataset: BicameralClauseChainDataset,
    seed: int = 42,
    max_domains: int = 6,
) -> List[Tuple[str, int]]:
    """
    Deterministically selects 1 representative sample index per available domain (up to `max_domains`)
    using a fixed random seed (`42`) for reproducible Dual-Brain multi-clause rollout benchmarks.
    """
    rng = random.Random(seed)
    domain_to_indices: Dict[str, List[int]] = {}

    if hasattr(dataset, "records") and isinstance(dataset.records, list):
        for idx, rec in enumerate(dataset.records):
            dom = str(rec.get("domain", "general") or "general").strip()
            domain_to_indices.setdefault(dom, []).append(idx)
    else:
        n = min(len(dataset), max_domains)
        return [("general", i) for i in range(n)]

    if len(domain_to_indices) <= 1 and len(dataset) > 1:
        n = min(len(dataset), max_domains)
        step = max(1, len(dataset) // n)
        return [(f"sample_{i * step}", i * step) for i in range(n)]

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


def find_latest_left_brain_ckpt(
    unet: nn.Module,
    base_dirs: Sequence[str] = ("checkpoints/run", "checkpoints"),
    preferred_prefixes: Sequence[str] = ("left_brain_s32_fullrank", "phase4_sentvec_flow", "hippo_flow_moss"),
    device: Union[str, torch.device] = "cpu",
) -> Optional[str]:
    """
    Auto-discovers the latest compatible `LeftBrainCrystallizer` (`unet.pt`) checkpoint across `base_dirs`,
    prioritizing `preferred_prefixes` in order (`left_brain_s32_fullrank` -> `phase4_sentvec_flow` -> `hippo_flow_moss`),
    and verifying `strict=True` state_dict compatibility.
    """
    candidates: List[Tuple[int, int, str]] = []
    seen_files = set()

    for bdir in base_dirs:
        if not bdir or not os.path.exists(bdir):
            continue
        for d in os.listdir(bdir):
            full_dir = os.path.join(bdir, d)
            if not os.path.isdir(full_dir):
                continue
            pt_file = os.path.join(full_dir, "unet.pt")
            if not os.path.isfile(pt_file) or pt_file in seen_files:
                continue

            priority = -1
            for p_idx, pref in enumerate(preferred_prefixes):
                if pref and pref in d:
                    priority = len(preferred_prefixes) - p_idx
                    break
            if priority < 0 and ("phase4" in d or "left_brain" in d or "hippo_flow" in d):
                priority = 0
            if priority < 0:
                continue

            step = 0
            m = re.search(r"step_(\d+)", d)
            if m:
                step = int(m.group(1))
            seen_files.add(pt_file)
            candidates.append((priority, step, pt_file))

    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    for _, step, pt_file in candidates:
        try:
            sd = torch.load(pt_file, map_location=device, weights_only=True)
            unet.load_state_dict(sd, strict=True)
            return pt_file
        except Exception as e:
            print(f"ℹ️ Skipping incompatible Left-Brain checkpoint '{pt_file}' (step {step}): {e}")
    return None


def find_latest_translator_ckpt(base_dirs: Sequence[str] = ("checkpoints/run", "checkpoints")) -> Optional[str]:
    candidates: List[Tuple[int, int, str]] = []
    for bdir in base_dirs:
        if not bdir or not os.path.exists(bdir):
            continue
        for d in os.listdir(bdir):
            if "translator" in d and "step_" in d:
                full_dir = os.path.join(bdir, d)
                pt_file = os.path.join(full_dir, "translator.pt")
                if os.path.isfile(pt_file):
                    m = re.search(r"step_(\d+)", d)
                    step = int(m.group(1)) if m else 0
                    if "s32_clean" in d:
                        prio = 2
                    elif "deblur" not in d:
                        prio = 1
                    else:
                        prio = 0
                    candidates.append((prio, step, pt_file))
    if candidates:
        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return candidates[0][2]
    return None


def find_compatible_right_brain_ckpt(
    right_brain: nn.Module,
    knowledge_selector: nn.Module,
    out_dir: str,
    prefix: str = "bicameral_right_brain_v2",
    device: Union[str, torch.device] = "cpu",
) -> Tuple[Optional[str], int]:
    """
    Finds the latest checkpoint directory in `out_dir` whose `right_brain.pt` (and optional
    `knowledge_selector.pt`) strictly match the current architecture.
    """
    if not out_dir or not os.path.exists(out_dir):
        return None, 0

    candidates: List[Tuple[int, str, str]] = []
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
            rb_pt = os.path.join(d_path, "right_brain.pt")
            if os.path.isfile(rb_pt):
                candidates.append((step, d_path, d))

    candidates.sort(key=lambda x: x[0], reverse=True)
    for step, c_path, c_name in candidates:
        rb_pt = os.path.join(c_path, "right_brain.pt")
        ks_pt = os.path.join(c_path, "knowledge_selector.pt")
        try:
            rb_sd = torch.load(rb_pt, map_location=device, weights_only=True)
            right_brain.load_state_dict(rb_sd, strict=True)
            if os.path.isfile(ks_pt):
                ks_sd = torch.load(ks_pt, map_location=device, weights_only=True)
                knowledge_selector.load_state_dict(ks_sd, strict=True)
            print(f"✅ Verified compatible Right-Brain checkpoint '{c_name}' (step {step})")
            return c_path, step
        except Exception as e:
            print(f"ℹ️ Skipping incompatible Right-Brain checkpoint '{c_name}' (step {step}): {e}")

    return None, 0


def decode_clause_clean(tokenizer: CharTokenizer, token_ids: List[int]) -> str:
    """
    Decodes token IDs for a single clause, truncating at the first `<EOS>` token.
    """
    if tokenizer.eos_token_id in token_ids:
        eos_idx = token_ids.index(tokenizer.eos_token_id)
        token_ids = token_ids[:eos_idx]
    return tokenizer.decode(token_ids, skip_special_tokens=True).strip()


@torch.no_grad()
def encode_tokens(encoder: TinyCharEncoderCUDA, token_ids: torch.Tensor) -> torch.Tensor:
    """
    Encodes token IDs `(N, L)` through the frozen Phase 1 v2 `TinyCharEncoderCUDA` into `(N, L, 1024)`
    with mandatory `<PAD>` attention masking and strict zeroing of padded positions.
    """
    return encoder.encode_tokens(token_ids)


@torch.no_grad()
def encode_clause_bank_to_2d(
    encoder: TinyCharEncoderCUDA,
    unet: LeftBrainCrystallizer,
    hippo_oracle: HiPPOOracleCUDA,
    w_bank: torch.Tensor,
    clause_valid_mask: torch.Tensor,
    pad_token_id: int,
) -> torch.Tensor:
    """
    Encodes a 3D token bank `w_bank: (B, M, L)` of punctuation clauses into 2D HiPPO clause vectors
    `z_bank: (B, M, 1024)` using `unet.normalize(...)` and `hippo_oracle(..., subspace_proj=unet.subspace_proj)`.
    Invalid/padded clause slots (`clause_valid_mask == False`) are strictly zeroed out.
    """
    B, M, L = w_bank.shape
    device = w_bank.device
    D = unet.hidden_dim

    tok_valid = (w_bank != pad_token_id) & clause_valid_mask.bool().unsqueeze(-1)
    active_clause_mask = tok_valid.any(dim=-1)  # (B, M)

    if not active_clause_mask.any():
        return torch.zeros((B, M, D), device=device, dtype=torch.float32)

    flat_w = w_bank.reshape(B * M, L)
    flat_tok_valid = tok_valid.reshape(B * M, L)
    active_idx = torch.nonzero(active_clause_mask.reshape(B * M), as_tuple=False).squeeze(-1)

    active_tok_mask = flat_tok_valid[active_idx]
    x_active = encoder.encode_tokens(flat_w[active_idx], active_tok_mask.float())  # (N_active, L, D)

    if int(unet.calib_steps.item()) == 0:
        unet.calibrate_manifold(x_active, valid_mask=active_tok_mask)

    x_active_norm = unet.normalize(x_active)
    z_active = hippo_oracle(
        x_active_norm,
        return_sequence=False,
        valid_mask=active_tok_mask,
        subspace_proj=unet.subspace_proj,
    )  # (N_active, D)

    z_flat = torch.zeros((B * M, D), device=device, dtype=z_active.dtype)
    z_flat[active_idx] = z_active
    z_bank = z_flat.reshape(B, M, D)
    z_bank = z_bank * clause_valid_mask.unsqueeze(-1).to(dtype=z_bank.dtype)
    return z_bank


@torch.no_grad()
def encode_clause_bank_dual(
    encoder: TinyCharEncoderCUDA,
    unet: LeftBrainCrystallizer,
    hippo_oracle: HiPPOOracleCUDA,
    w_bank: torch.Tensor,
    clause_valid_mask: torch.Tensor,
    pad_token_id: int,
    semantic_source: str = "z_god",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    [Task #059] Encodes a 3D clause token bank `w_bank: (B, M, L)` into BOTH:
      1) `z_bank: (B, M, 1024)` — 2D HiPPO Legendre character-waveform vector (`Z_clause`)
      2) `s_bank: (B, M, 1024)` — 1024-D clause semantic planning vector (`z_god` or `s_pool`)
    in a SINGLE masked Transformer forward pass (`encoder.encode_dual(..., return_hidden=True)`).
    Invalid/padded clause rows (`clause_valid_mask == False`) are strictly zeroed out in both outputs.
    """
    if semantic_source not in ("z_god", "s_pool"):
        raise ValueError(f"Unsupported semantic_source='{semantic_source}'; expected 'z_god' or 's_pool'.")

    B, M, L = w_bank.shape
    device = w_bank.device
    D = unet.hidden_dim
    D_sem = encoder.z_dim if semantic_source == "z_god" else encoder.d_model

    tok_valid = (w_bank != pad_token_id) & clause_valid_mask.bool().unsqueeze(-1)
    active_clause_mask = tok_valid.any(dim=-1)  # (B, M)

    if not active_clause_mask.any():
        return (
            torch.zeros((B, M, D), device=device, dtype=torch.float32),
            torch.zeros((B, M, D_sem), device=device, dtype=torch.float32),
        )

    flat_w = w_bank.reshape(B * M, L)
    flat_tok_valid = tok_valid.reshape(B * M, L)
    active_idx = torch.nonzero(active_clause_mask.reshape(B * M), as_tuple=False).squeeze(-1)
    active_tok_mask = flat_tok_valid[active_idx]

    dual = encoder.encode_dual(
        flat_w[active_idx],
        active_tok_mask.float(),
        return_hidden=True,
    )
    s_active = dual["z_god"] if semantic_source == "z_god" else dual["s_pool"]
    x_active = dual["h_tokens"]  # (N_active, L, D)

    if int(unet.calib_steps.item()) == 0:
        unet.calibrate_manifold(x_active, valid_mask=active_tok_mask)

    x_active_norm = unet.normalize(x_active)
    z_active = hippo_oracle(
        x_active_norm,
        return_sequence=False,
        valid_mask=active_tok_mask,
        subspace_proj=unet.subspace_proj,
    )  # (N_active, D)

    z_flat = torch.zeros((B * M, D), device=device, dtype=z_active.dtype)
    z_flat[active_idx] = z_active
    z_bank = z_flat.reshape(B, M, D)
    z_bank = z_bank * clause_valid_mask.unsqueeze(-1).to(dtype=z_bank.dtype)

    s_flat = torch.zeros((B * M, D_sem), device=device, dtype=s_active.dtype)
    s_flat[active_idx] = s_active
    s_bank = s_flat.reshape(B, M, D_sem)
    s_bank = s_bank * clause_valid_mask.unsqueeze(-1).to(dtype=s_bank.dtype)

    return z_bank, s_bank


@torch.no_grad()
def crystallize_clause_euler_ode(
    unet: LeftBrainCrystallizer,
    z_clause: torch.Tensor,
    x_attractors_norm: Optional[torch.Tensor] = None,
    k_mask: Optional[torch.Tensor] = None,
    steps: int = 10,
    seq_len: int = 32,
) -> torch.Tensor:
    """
    Integrates the frozen `LeftBrainCrystallizer` (`unet`) for `steps` Euler ODE steps conditioned on
    the 2D clause vector `h_potential = z_clause: (B, 1024)` and span-preserving `k_attractors`
    to crystallize continuous character embeddings `x_clause_gen: (B, seq_len, 1024)`.
    """
    B, D = z_clause.shape
    device = z_clause.device
    dtype = z_clause.dtype

    x = torch.randn((B, seq_len, D), device=device, dtype=dtype)
    dt = 1.0 / float(max(steps, 1))
    for i in range(steps):
        t_val = i / float(max(steps, 1))
        t_tensor = torch.full((B,), t_val, device=device, dtype=torch.float32)
        v_pred = unet(
            x,
            t_tensor,
            h_potential=z_clause,
            k_attractors=x_attractors_norm,
            k_mask=k_mask,
        )
        x = x + v_pred * dt
    return unet.denormalize(x)


@torch.no_grad()
def select_and_crystallize_clause(
    z_probe: torch.Tensor,
    knowledge_selector: KnowledgeSelector,
    unet: LeftBrainCrystallizer,
    encoder: TinyCharEncoderCUDA,
    z_cand: torch.Tensor,
    w_cand: torch.Tensor,
    w_goal: torch.Tensor,
    cand_valid: torch.Tensor,
    args,
    pad_id: int,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    #029 per-clause post-thought retrieval: probes `KnowledgeSelector.select_active_clause_attractors`
    with THIS clause's vector `z_probe: (B, D)` (z_thought at rollout, GT clause for oracle decode),
    builds span-preserving left-brain attractors `[w_goal, top-M selected candidates]` and integrates
    the frozen `LeftBrainCrystallizer` Euler ODE. Returns `(x_clause_gen (B, L, D), top_scores (B, M))`.
    """
    w_attractors, k_mask, top_scores = knowledge_selector.select_active_clause_attractors(
        z_query=z_probe.float(),
        z_candidates=z_cand,
        w_candidates=w_cand,
        w_goal=w_goal,
        cand_valid_mask=cand_valid,
        top_m_clauses=args.top_m_clauses,
        min_score=args.min_selector_score,
        pad_token_id=pad_id,
    )
    x_attractors_norm = unet.normalize(encode_tokens(encoder, w_attractors))
    with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
        x_clause_gen = crystallize_clause_euler_ode(
            unet=unet,
            z_clause=z_probe,
            x_attractors_norm=x_attractors_norm,
            k_mask=k_mask,
            steps=args.left_ode_steps,
            seq_len=args.seq_len,
        )
    return x_clause_gen, top_scores


@torch.no_grad()
def decode_clause_vector(
    x_clause_gen: torch.Tensor,
    translator: Optional[ContinuousToTextTranslatorCUDA],
    tokenizer: CharTokenizer,
    seq_len: int,
    device: torch.device,
) -> Tuple[str, torch.Tensor]:
    """Translator greedy decode of `x_clause_gen: (1, L, D)` -> (clause text, token valid mask (1, L))."""
    pad_id = tokenizer.pad_token_id
    if translator is None:
        return "<no-translator>", torch.ones((1, seq_len), device=device, dtype=torch.bool)
    pred_ids = translator.decode_greedy(x_clause_gen.float())[0].tolist()
    gen_valid_mask = torch.tensor(pred_ids, device=device, dtype=torch.long).unsqueeze(0) != pad_id
    return decode_clause_clean(tokenizer, pred_ids), gen_valid_mask


def _sigma_sem_syn(sigma_band: torch.Tensor, sem_bands: int) -> Tuple[float, float]:
    """Mean low-band / high-band sigma of a single `(nb,)` per-band sigma vector."""
    s = sigma_band.float().reshape(-1)
    L = max(1, min(int(sem_bands), s.numel()))
    sem = float(s[:L].mean().item())
    syn = float(s[L:].mean().item()) if L < s.numel() else 0.0
    return sem, syn


@torch.no_grad()
def evaluate_dual_brain_rollout(
    step: int,
    right_brain: MultiStreamRightBrain,
    knowledge_selector: KnowledgeSelector,
    unet: LeftBrainCrystallizer,
    encoder: TinyCharEncoderCUDA,
    hippo_oracle: HiPPOOracleCUDA,
    translator: Optional[ContinuousToTextTranslatorCUDA],
    tokenizer: CharTokenizer,
    dataset: BicameralClauseChainDataset,
    benchmark_samples: List[Tuple[str, int]],
    active_streams: Sequence[str],
    args,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
):
    """
    End-to-End Dual-Brain Autoregressive Multi-Clause Rollout Evaluation (#029, Right-Brain v2):
      For every benchmark sample:
        1. Encodes `w_goal` / `w_cand` / `w_discourse` -> 2D HiPPO clause vectors and packs companion
           streams with `right_brain.pack_aux_streams` (stream 0 = goal, 1..C = candidates).
        2. Oracle decode: per GT clause, probes the KnowledgeSelector with the GT clause vector to pick
           THAT clause's attractors, then LeftBrain Euler ODE -> Translator.
        3. For every condition in `--eval_conditions` (full / goal_only / none; companions masked exactly
           like training stream dropout) runs the autoregressive rollout:
             `step_clause(return_all=True)` -> per-step selector probing with `z_query = z_thought`
             -> LeftBrain ODE -> Translator -> closed-loop re-encode (`--closed_loop_reencode`)
             -> stop when `stop_prob > --stop_threshold` (>= 1 clause, <= `--num_rollout_clauses`).
           Prints per-step text, stop_prob, sigma_sem / sigma_syn, top selector scores and (full) K+ vs K-
           companion gate means.
      Finally a diversity demo samples clause 1 of the first benchmark sample `--eval_num_samples` times
      under `full` and `none` (decoded text + sigma_sem + pairwise cosine of the sampled vectors).
    """
    right_brain.eval()
    knowledge_selector.eval()
    unet.eval()

    pad_id = tokenizer.pad_token_id
    sem_bands = int(getattr(right_brain, "sem_bands", args.sem_bands))
    discourse_only = all(s == "discourse" for s in active_streams)
    conditions = [c for c in args.eval_conditions_list if c in STREAM_GROUPS]
    if discourse_only:
        conditions = ["none"]
    if not conditions:
        conditions = ["full"]

    mode_tag = "Closed-Loop Re-Encode" if args.closed_loop_reencode else "Open-Loop Latent"
    print("\n" + "=" * 92)
    print(
        f"🧠 [Step {step}] Right-Brain v2 Dual-Brain Rollout "
        f"(Streams={','.join(active_streams)} | Conditions={','.join(conditions)} | Mode={mode_tag} | "
        f"RightODE={args.right_ode_steps} | LeftODE={args.left_ode_steps} | T={args.temperature} | "
        f"MaxClauses={args.num_rollout_clauses} | StopThr={args.stop_threshold})"
    )
    if discourse_only:
        print("   ℹ️ --active_streams discourse: companions ignored, evaluating condition 'none' only.")
    print("=" * 92)

    first_sample_cache: Optional[Dict[str, Any]] = None
    dual_mode = getattr(args, "planning_space", "z_clause") in ("z_god", "s_pool")

    for dom_name, sample_idx in benchmark_samples:
        item = dataset[sample_idx]
        w_goal = item["w_goal"].unsqueeze(0).to(device)                  # (1, G, L)
        goal_valid = item["goal_valid_mask"].unsqueeze(0).to(device)     # (1, G)
        w_cand = item["w_cand"].unsqueeze(0).to(device)                  # (1, C, L)
        cand_pos = item["cand_pos_mask"].unsqueeze(0).to(device)         # (1, C)
        cand_valid = item["cand_valid_mask"].unsqueeze(0).to(device)     # (1, C)
        w_disc = item["w_discourse"].unsqueeze(0).to(device)             # (1, T, L)
        disc_valid = item["discourse_valid_mask"].unsqueeze(0).to(device)  # (1, T)

        if dual_mode:
            z_goal, s_goal = encode_clause_bank_dual(
                encoder, unet, hippo_oracle, w_goal, goal_valid, pad_id, semantic_source=args.planning_space
            )
            z_cand, s_cand = encode_clause_bank_dual(
                encoder, unet, hippo_oracle, w_cand, cand_valid, pad_id, semantic_source=args.planning_space
            )
            z_disc, s_disc = encode_clause_bank_dual(
                encoder, unet, hippo_oracle, w_disc, disc_valid, pad_id, semantic_source=args.planning_space
            )
            s_aux_full, _ = right_brain.pack_aux_streams(s_goal, goal_valid, s_cand, cand_valid)
        else:
            z_goal = encode_clause_bank_to_2d(encoder, unet, hippo_oracle, w_goal, goal_valid, pad_id)
            z_cand = encode_clause_bank_to_2d(encoder, unet, hippo_oracle, w_cand, cand_valid, pad_id)
            z_disc = encode_clause_bank_to_2d(encoder, unet, hippo_oracle, w_disc, disc_valid, pad_id)
            s_disc = None
            s_aux_full = None

        aux_full, aux_valid_full = right_brain.pack_aux_streams(z_goal, goal_valid, z_cand, cand_valid)

        # 1. Sample-level KnowledgeSelector diagnostics (goal-mean probe, kept from #027)
        goal_f = goal_valid.unsqueeze(-1).to(dtype=z_goal.dtype)
        z_query = ((z_goal * goal_f).sum(dim=1) + z_disc[:, 0, :]) / (goal_f.sum(dim=1).clamp(min=0.0) + 1.0)
        _, all_gates = knowledge_selector.score_clause_candidates(
            z_query=z_query, z_candidates=z_cand, cand_valid_mask=cand_valid
        )
        valid_b = cand_valid[0].bool()
        if valid_b.any():
            preds_bin = (all_gates[0][valid_b] >= 0.5).float()
            labels_bin = (cand_pos[0][valid_b] >= 0.5).float()
            sel_acc = (preds_bin == labels_bin).float().mean().item() * 100.0
        else:
            sel_acc = 100.0

        # 2. Ground-truth target clauses & Oracle Left-Brain Decode (per-clause GT-probe selection)
        num_valid_disc = int(disc_valid[0].sum().item())
        num_gt_ans = max(1, min(num_valid_disc - 1, args.num_rollout_clauses))
        gt_clause_strs: List[str] = []
        oracle_clause_strs: List[str] = []
        for c_i in range(1, num_gt_ans + 1):
            gt_clause_strs.append(decode_clause_clean(tokenizer, w_disc[0, c_i].tolist()))
            x_oracle_c, _ = select_and_crystallize_clause(
                z_disc[:, c_i, :], knowledge_selector, unet, encoder, z_cand, w_cand, w_goal, cand_valid,
                args, pad_id, device, amp_enabled, amp_dtype,
            )
            ora_str, _ = decode_clause_vector(x_oracle_c, translator, tokenizer, args.seq_len, device)
            oracle_clause_strs.append(ora_str)

        print(f"📌 Domain [{dom_name}] (Sample #{sample_idx})")
        print(f"   [Question]:                {item['question_text']}")
        print(f"   [Target Clauses]:          {' | '.join(gt_clause_strs)}")
        print(
            f"   [KnowledgeSelector]:       goal-probe K+/K- Acc={sel_acc:.1f}% | "
            f"C_valid={int(valid_b.sum().item())} | K+={int((cand_pos[0][valid_b] >= 0.5).sum().item())}"
        )
        print(f"   [Oracle Left-Brain Decode]: {' | '.join(oracle_clause_strs)}")

        # 3. Autoregressive rollout per stream condition
        for cond in conditions:
            group = torch.full((1,), STREAM_GROUPS[cond], device=device, dtype=torch.long)
            aux_valid_c = apply_stream_group_mask(aux_valid_full, group)

            rb_states: Optional[Dict[str, Any]] = None
            z_curr = z_disc[:, 0, :]  # Clause 0: lead question departure clause
            s_curr = s_disc[:, 0, :] if dual_mode and s_disc is not None else None
            rollout_strs: List[str] = []
            step_lines: List[str] = []
            clause1_cos = 0.0
            gate_pos_acc: List[float] = []
            gate_neg_acc: List[float] = []
            gate_goal_acc: List[float] = []
            stopped_at: Optional[int] = None

            for c_step in range(1, args.num_rollout_clauses + 1):
                with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                    z_thought_c, _h_t, rb_states, info = right_brain.step_clause(
                        z_curr=z_curr,
                        states=rb_states,
                        aux=aux_full,
                        aux_valid=aux_valid_c,
                        ode_steps=args.right_ode_steps,
                        temperature=args.temperature,
                        return_all=True,
                        s_curr=s_curr,
                        s_aux=s_aux_full,
                    )
                z_thought_c = z_thought_c.float()
                x_clause_gen, top_scores = select_and_crystallize_clause(
                    z_thought_c, knowledge_selector, unet, encoder, z_cand, w_cand, w_goal, cand_valid,
                    args, pad_id, device, amp_enabled, amp_dtype,
                )
                if c_step == 1 and num_valid_disc >= 2:
                    clause1_cos = F.cosine_similarity(z_thought_c, z_disc[:, 1, :].float(), dim=-1).mean().item()

                pred_str, gen_valid_mask = decode_clause_vector(
                    x_clause_gen, translator, tokenizer, args.seq_len, device
                )
                if translator is None:
                    pred_str = f"<z_norm={z_thought_c.norm().item():.2f}>"
                rollout_strs.append(pred_str)

                stop_prob = float(info["stop_prob"].reshape(-1)[0].item())
                s_sem, s_syn = _sigma_sem_syn(info["sigma_band"][0], sem_bands)
                scores_str = ",".join(f"{s:.2f}" for s in top_scores[0].tolist())
                step_lines.append(
                    f"      c{c_step}: stop={stop_prob:.3f} σ_sem={s_sem:.3f} σ_syn={s_syn:.3f} "
                    f"sel=[{scores_str}] | {pred_str}"
                )

                gates = info.get("gates")
                if gates is not None and gates.dim() == 2 and gates.shape[-1] >= 1:
                    g = gates[0].float()
                    gate_goal_acc.append(float(g[0].item()))
                    if g.shape[0] > 1:
                        g_c = g[1:]
                        pos_m = (cand_pos[0] >= 0.5) & cand_valid[0].bool()
                        neg_m = (cand_pos[0] < 0.5) & cand_valid[0].bool()
                        if g_c.shape[0] == pos_m.shape[0]:
                            if pos_m.any():
                                gate_pos_acc.append(float(g_c[pos_m].mean().item()))
                            if neg_m.any():
                                gate_neg_acc.append(float(g_c[neg_m].mean().item()))

                if args.closed_loop_reencode:
                    x_gen_norm = unet.normalize(x_clause_gen.float())
                    z_reencoded = hippo_oracle(
                        x_gen_norm,
                        return_sequence=False,
                        valid_mask=gen_valid_mask,
                        subspace_proj=unet.subspace_proj,
                    )
                    z_curr = z_reencoded.to(dtype=z_disc.dtype)
                    if dual_mode and s_disc is not None:
                        if translator is not None:
                            w_re = torch.argmax(translator(x_clause_gen.float()), dim=-1)
                            w_re = torch.where(gen_valid_mask.bool(), w_re, torch.full_like(w_re, pad_id))
                            if gen_valid_mask.any():
                                dual_re = encoder.encode_dual(w_re, gen_valid_mask.float(), return_hidden=False)
                                s_curr = (
                                    dual_re["z_god"] if args.planning_space == "z_god" else dual_re["s_pool"]
                                ).to(dtype=s_disc.dtype)
                            else:
                                s_curr = info["s_pred"].to(dtype=s_disc.dtype)
                        else:
                            s_curr = info["s_pred"].to(dtype=s_disc.dtype)
                else:
                    z_curr = z_thought_c.to(dtype=z_disc.dtype)
                    if dual_mode and s_disc is not None:
                        s_curr = info["s_pred"].to(dtype=s_disc.dtype)

                if stop_prob > float(args.stop_threshold):
                    stopped_at = c_step
                    break

            stop_desc = f"stop@c{stopped_at}" if stopped_at is not None else f"max {args.num_rollout_clauses}"
            print(
                f"   [Rollout | {cond:<9}]:    {' | '.join(rollout_strs)} "
                f"(Clause-1 CosSim={clause1_cos:.4f}, {stop_desc})"
            )
            for line in step_lines:
                print(line)
            if cond == "full":
                def _mean(xs: List[float]) -> Optional[float]:
                    return (sum(xs) / len(xs)) if xs else None
                print(
                    f"      gates(full): goal={_fmt_opt(_mean(gate_goal_acc))} | "
                    f"K+={_fmt_opt(_mean(gate_pos_acc))} | K-={_fmt_opt(_mean(gate_neg_acc))}"
                )

        if first_sample_cache is None:
            first_sample_cache = dict(
                dom_name=dom_name, sample_idx=sample_idx, z_disc=z_disc, s_disc=s_disc, z_cand=z_cand, w_cand=w_cand,
                w_goal=w_goal, cand_valid=cand_valid, aux_full=aux_full, s_aux_full=s_aux_full,
                aux_valid_full=aux_valid_full,
                gt1=gt_clause_strs[0] if gt_clause_strs else "",
                has_gt1=num_valid_disc >= 2,
            )
        print("-" * 92)

    # 4. Diversity demo: clause 1 of the first benchmark sample, sampled N times under full / none
    n_div = int(getattr(args, "eval_num_samples", 0) or 0)
    if first_sample_cache is not None and n_div > 0:
        fc = first_sample_cache
        demo_conds = ["none"] if discourse_only else ["full", "none"]
        print(
            f"🎲 Diversity demo — Domain [{fc['dom_name']}] (Sample #{fc['sample_idx']}) clause 1 x{n_div} "
            f"(T={args.temperature}) | GT: {fc['gt1']}"
        )
        for cond in demo_conds:
            group = torch.full((1,), STREAM_GROUPS[cond], device=device, dtype=torch.long)
            aux_valid_c = apply_stream_group_mask(fc["aux_valid_full"], group)
            z_samples: List[torch.Tensor] = []
            for s_i in range(n_div):
                with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                    z_s, _h, _st, info = right_brain.step_clause(
                        z_curr=fc["z_disc"][:, 0, :],
                        states=None,
                        aux=fc["aux_full"],
                        aux_valid=aux_valid_c,
                        ode_steps=args.right_ode_steps,
                        temperature=args.temperature,
                        return_all=True,
                        s_curr=fc["s_disc"][:, 0, :] if dual_mode and fc["s_disc"] is not None else None,
                        s_aux=fc["s_aux_full"] if dual_mode else None,
                    )
                z_s = z_s.float()
                z_samples.append(z_s[0])
                x_gen, _ = select_and_crystallize_clause(
                    z_s, knowledge_selector, unet, encoder, fc["z_cand"], fc["w_cand"], fc["w_goal"],
                    fc["cand_valid"], args, pad_id, device, amp_enabled, amp_dtype,
                )
                txt, _ = decode_clause_vector(x_gen, translator, tokenizer, args.seq_len, device)
                s_sem, s_syn = _sigma_sem_syn(info["sigma_band"][0], sem_bands)
                gt_cos = (
                    F.cosine_similarity(z_s, fc["z_disc"][:, 1, :].float(), dim=-1).mean().item()
                    if fc["has_gt1"] else float("nan")
                )
                print(
                    f"   [{cond:<4} #{s_i + 1}] σ_sem={s_sem:.3f} σ_syn={s_syn:.3f} cos(GT)={gt_cos:.4f} | {txt}"
                )
            if len(z_samples) >= 2:
                zs = F.normalize(torch.stack(z_samples, dim=0), dim=-1)
                sim = zs @ zs.t()
                n = sim.shape[0]
                off = sim[~torch.eye(n, dtype=torch.bool, device=sim.device)]
                print(f"   [{cond:<4}] mean pairwise cos between samples = {off.mean().item():.4f}")

    print("=" * 92 + "\n")


def train():
    default_data_path = "data/Basic_ZH/moss_dialogues.jsonl,data/Knowledge_ZH/bicameral_knowledge_qa.jsonl"
    if os.path.exists("/content/omni_knowledge_corpus.jsonl") and not os.path.exists("data/Basic_ZH/moss_dialogues.jsonl"):
        default_data_path = "/content/omni_knowledge_corpus.jsonl"

    parser = get_training_parser(
        "Stage 2 & Stage 3 Modular Bicameral Right-Brain + KnowledgeSelector Training & Dual-Brain Rollout (CUDA)"
    )
    parser.add_argument("--data_path", type=str, default=default_data_path, help="Comma-separated dataset path(s)")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to load (useful for smoke tests)")
    parser.add_argument("--max_steps", type=int, default=50000, help="Maximum training steps horizon")
    parser.add_argument("--min_lr_ratio", type=float, default=0.1, help="Minimum cosine LR decay ratio")
    parser.add_argument(
        "--encoder_ckpt",
        type=str,
        default="checkpoints/basencoder_step_115000/student.pt",
        help="Path to frozen Phase 1 TinyCharEncoderCUDA checkpoint",
    )
    parser.add_argument(
        "--left_brain_ckpt",
        "--unet_ckpt",
        dest="left_brain_ckpt",
        type=str,
        default=None,
        help="Path to frozen LeftBrainCrystallizer checkpoint (auto-discovered if omitted)",
    )
    parser.add_argument(
        "--left_brain_ckpt_prefix",
        type=str,
        default="left_brain_s32_fullrank",
        help="Preferred checkpoint prefix when auto-discovering LeftBrainCrystallizer",
    )
    parser.add_argument(
        "--translator_ckpt",
        type=str,
        default=None,
        help="Path to frozen ContinuousToTextTranslatorCUDA checkpoint (auto-discovered if omitted)",
    )
    parser.add_argument("--num_subbands", type=int, default=32, help="Number of Legendre harmonic modes K (default 32)")
    parser.add_argument("--d_mamba", type=int, default=1024, help="Internal hidden dimension of ClauseMambaStream")
    # #029: 1 main + N companion streams. Legacy #027 per-stream layer flags map onto main / aux.
    parser.add_argument("--n_layers_main", type=int, default=None, help="Mamba layers in the main (discourse) stream (default 4)")
    parser.add_argument("--n_layers_aux", type=int, default=None, help="Mamba layers in the shared companion stream (default 2)")
    parser.add_argument("--n_layers_discourse", type=int, default=None, help="Legacy alias for --n_layers_main")
    parser.add_argument("--n_layers_knowledge", type=int, default=None, help="Legacy alias for --n_layers_aux")
    parser.add_argument("--n_layers_goal", type=int, default=None, help="Legacy alias for --n_layers_aux")
    parser.add_argument(
        "--active_streams",
        type=str,
        default="discourse,knowledge,goal",
        help="Comma-separated active Right-Brain streams: 'discourse' (single-stream warmup, all samples "
             "use stream group 'none') or anything else (e.g. 'discourse,knowledge,goal' / 'discourse,aux') "
             "to enable companion streams with stream dropout",
    )
    parser.add_argument("--max_goal_clauses", type=int, default=4, help="Max question clauses in Goal Stream")
    parser.add_argument("--max_cand_clauses", type=int, default=8, help="Max candidate knowledge clauses in pool")
    parser.add_argument("--max_discourse_clauses", type=int, default=8, help="Max progressive clauses in Discourse Stream")
    parser.add_argument("--min_clause_chars", type=int, default=6, help="Min characters per punctuation clause")
    parser.add_argument("--max_clause_chars", type=int, default=28, help="Max characters per punctuation clause")
    parser.add_argument(
        "--selector_weight", type=float, default=None,
        help="Legacy alias for --sample_selector_weight (sample-level goal-probe selector loss)",
    )
    parser.add_argument("--top_m_clauses", type=int, default=3, help="Top-M candidate clauses selected by KnowledgeSelector")
    parser.add_argument("--min_selector_score", type=float, default=0.20, help="Minimum gate score threshold for candidate clauses")
    parser.add_argument("--eval_interval", type=int, default=200, help="Run Dual-Brain rollout evaluation every N steps")
    parser.add_argument("--log_interval", type=int, default=20, help="Print training metrics every N steps")
    parser.add_argument("--right_ode_steps", type=int, default=10, help="Euler ODE steps for Right-Brain ThoughtFlowMatcher")
    parser.add_argument(
        "--left_ode_steps",
        "--euler_steps",
        dest="left_ode_steps",
        type=int,
        default=10,
        help="Euler ODE steps for LeftBrainCrystallizer",
    )
    parser.add_argument("--num_rollout_clauses", type=int, default=6, help="Max number of clauses to rollout during evaluation")
    parser.add_argument("--temperature", type=float, default=0.75, help="Prior noise temperature for ThoughtFlowMatcher sampling")
    # ---------------- #029 Right-Brain v2 (adaptive prior / K-branch flow / stop head) ----------------
    parser.add_argument("--num_branches", type=int, default=4, help="K blind flow branches per clause step (soft competition)")
    parser.add_argument("--train_ode_steps", type=int, default=3, help="Euler steps of each blind training branch / self-feed sampling")
    parser.add_argument("--win_weight", type=float, default=0.65, help="Weight of the winner branch loss (others get 1 - win_weight)")
    parser.add_argument("--sem_bands", type=int, default=6, help="Number of low Legendre bands treated as semantic (hypothesis)")
    parser.add_argument("--sigma_min", type=float, default=0.05, help="Adaptive prior per-band sigma floor (normalized space)")
    parser.add_argument("--sigma_max", type=float, default=3.0, help="Adaptive prior per-band sigma ceiling (normalized space)")
    parser.add_argument("--prior_weight", type=float, default=1.0, help="Weight of the band Gaussian NLL prior loss")
    parser.add_argument("--stop_weight", type=float, default=0.2, help="Weight of the stop-head BCE loss")
    parser.add_argument("--stop_threshold", type=float, default=0.5, help="Rollout stops once sigmoid(stop) exceeds this")
    parser.add_argument("--p_full", type=float, default=0.6, help="Stream dropout: P(all companion streams)")
    parser.add_argument("--p_goal_only", type=float, default=0.2, help="Stream dropout: P(goal stream only); none = remainder")
    parser.add_argument("--p_band_jitter", type=float, default=0.3, help="P(random orthogonal rotation of high bands) per input clause")
    parser.add_argument("--p_self_max", type=float, default=0.25, help="Max scheduled self-feeding probability")
    parser.add_argument("--self_feed_start", type=int, default=2000, help="Global step where the self-feeding ramp starts")
    parser.add_argument("--self_feed_ramp", type=int, default=8000, help="Steps to ramp p_self from 0 to --p_self_max")
    parser.add_argument("--clause_selector_weight", type=float, default=0.5, help="Weight of the per-clause selector loss")
    parser.add_argument("--sample_selector_weight", type=float, default=0.25, help="Weight of the sample-level (goal-probe) selector loss")
    parser.add_argument("--probe_noise_std", type=float, default=0.05, help="Relative Gaussian noise on GT clause probes (x per-row std)")
    parser.add_argument("--rel_overlap_threshold", type=float, default=0.15, help="Min bigram overlap for cand_clause_rel pseudo-labels")
    parser.add_argument("--eval_conditions", type=str, default="full,none", help="Comma-separated rollout conditions: full,goal_only,none")
    parser.add_argument("--eval_num_samples", type=int, default=3, help="Clause-1 samples per condition in the diversity demo (0=off)")
    parser.add_argument(
        "--planning_space",
        type=str,
        default="z_god",
        choices=["z_god", "s_pool", "z_clause"],
        help="Right-Brain Mamba planning space: 'z_god' (default), 's_pool', or legacy 'z_clause'",
    )
    parser.add_argument(
        "--sem_self_cond_prob",
        type=float,
        default=0.35,
        help="Probability of conditioning Z_clause bridge on predicted semantic sample instead of noisy GT",
    )
    parser.add_argument(
        "--auto_resume", dest="auto_resume", action="store_true",
        help="Alias of --continue: resume from the latest compatible checkpoint with the current --ckpt_prefix",
    )
    parser.add_argument(
        "--closed_loop_reencode",
        dest="closed_loop_reencode",
        action="store_true",
        default=True,
        help="Re-encode crystallized Left-Brain output back into Right-Brain Mamba during rollout (default True)",
    )
    parser.add_argument(
        "--no_closed_loop_reencode",
        dest="closed_loop_reencode",
        action="store_false",
        help="Feed latent z_thought directly without Left-Brain re-encoding during rollout",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "no"],
        help="Mixed precision mode (bf16, fp16, no)",
    )
    parser.add_argument("--accum_steps", type=int, default=1, help="Gradient accumulation steps")
    # Legacy CLI compatibility aliases (so old Phase 5 notebook invocations never crash)
    parser.add_argument("--max_know_streams", type=int, default=None, help="Legacy alias for --top_m_clauses")
    parser.add_argument("--know_seq_len", type=int, default=None, help="Legacy ignored flag (uses clause seq_len=32)")
    parser.add_argument("--num_beats", type=int, default=None, help="Legacy ignored flag")
    parser.add_argument("--mamba_layers", type=int, default=None, help="Legacy alias for --n_layers_main")

    parser.set_defaults(
        seq_len=32,
        batch_size=32,
        warmup_steps=500,
        save_steps=1000,
        ckpt_prefix="bicameral_right_brain_v2",
    )

    args = parser.parse_args()
    if args.seq_len > 64:
        print(f"ℹ️ Overriding legacy --seq_len {args.seq_len} -> 32 to match Punctuation-Clause LeftBrainCrystallizer.")
        args.seq_len = 32
    # Layer-count resolution: explicit v2 flags > legacy #027 flags > v2 defaults (main=4, aux=2).
    if args.n_layers_main is None:
        if args.mamba_layers is not None:
            args.n_layers_main = int(args.mamba_layers)
        elif args.n_layers_discourse is not None:
            args.n_layers_main = int(args.n_layers_discourse)
        else:
            args.n_layers_main = 4
    if args.n_layers_aux is None:
        legacy_aux = [v for v in (args.n_layers_goal, args.n_layers_knowledge) if v is not None]
        if legacy_aux:
            args.n_layers_aux = int(max(legacy_aux))
            if len(set(legacy_aux)) > 1:
                print(
                    f"ℹ️ Legacy --n_layers_goal/--n_layers_knowledge differ {legacy_aux}; "
                    f"using --n_layers_aux={args.n_layers_aux} for the shared companion stream."
                )
        else:
            args.n_layers_aux = 2
    if args.selector_weight is not None:
        print(f"ℹ️ Legacy --selector_weight {args.selector_weight} -> --sample_selector_weight.")
        args.sample_selector_weight = float(args.selector_weight)
    if args.max_know_streams is not None:
        args.top_m_clauses = int(args.max_know_streams)
    args.eval_conditions_list = [c.strip() for c in str(args.eval_conditions).split(",") if c.strip()]
    bad_conds = [c for c in args.eval_conditions_list if c not in STREAM_GROUPS]
    if bad_conds:
        print(f"⚠️ Ignoring unknown --eval_conditions {bad_conds} (valid: {list(STREAM_GROUPS)}).")
        args.eval_conditions_list = [c for c in args.eval_conditions_list if c in STREAM_GROUPS]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    active_streams = tuple(s.strip() for s in args.active_streams.split(",") if s.strip())
    if not active_streams:
        active_streams = ("discourse",)
    discourse_only = all(s == "discourse" for s in active_streams)

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

    # 1. Clause-Chain Dataset & DataLoader
    dataset, dataloader = create_clause_chain_dataloader(
        data_path=args.data_path,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        shuffle=True,
        seq_len=args.seq_len,
        max_goal_clauses=args.max_goal_clauses,
        max_cand_clauses=args.max_cand_clauses,
        max_discourse_clauses=args.max_discourse_clauses,
        min_clause_chars=args.min_clause_chars,
        max_clause_chars=args.max_clause_chars,
        max_samples=args.max_samples,
        seed=args.seed if args.seed is not None else 42,
        verbose=True,
        rel_overlap_threshold=args.rel_overlap_threshold,
    )

    benchmark_samples = select_fixed_benchmark_indices(dataset, seed=42, max_domains=6)
    if benchmark_samples:
        bench_desc = ", ".join(f"{dom}:#{idx}" for dom, idx in benchmark_samples)
        print(f"📌 Fixed Cross-Domain Rollout Benchmark Samples: {bench_desc}")

    # 2. Frozen Left-Brain & Perception Modules
    encoder = TinyCharEncoderCUDA(vocab_size=tokenizer.vocab_size).to(device)
    if os.path.exists(args.encoder_ckpt):
        if args.encoder_ckpt.endswith(".safetensors"):
            load_mlx_safetensors_into_torch(encoder, args.encoder_ckpt)
            encoder.to(device)
            print(f"✅ Loaded frozen TinyCharEncoderCUDA (safetensors) from {args.encoder_ckpt}")
        else:
            raw_enc_sd = torch.load(args.encoder_ckpt, map_location=device, weights_only=True)
            missing_k, unexpected_k = encoder.load_state_dict(raw_enc_sd, strict=False)
            print(f"✅ Loaded frozen TinyCharEncoderCUDA from {args.encoder_ckpt}")
            if missing_k:
                print(f"   ℹ️ Legacy encoder checkpoint missing keys (initialized fresh): {missing_k}")
            if unexpected_k:
                print(f"   ℹ️ Encoder checkpoint unexpected keys (ignored): {unexpected_k}")
    else:
        print(f"⚠️ WARNING: Encoder checkpoint '{args.encoder_ckpt}' not found! Using random weights (for testing only).")
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    hippo_oracle = HiPPOOracleCUDA(hidden_dim=1024, N=64, num_subbands=args.num_subbands).to(device)
    hippo_oracle.eval()

    unet = LeftBrainCrystallizer(
        hidden_dim=1024,
        cond_dim=1024,
        num_layers=6,
        num_subbands=args.num_subbands,
    ).to(device)

    left_ckpt_path = args.left_brain_ckpt
    if left_ckpt_path and ("*" in left_ckpt_path or "?" in left_ckpt_path):
        matched = sorted(glob.glob(left_ckpt_path))
        left_ckpt_path = matched[-1] if matched else None
    if left_ckpt_path and os.path.isdir(left_ckpt_path):
        left_ckpt_path = os.path.join(left_ckpt_path, "unet.pt")

    loaded_left_ok = False
    discovered_left = None
    if left_ckpt_path and os.path.exists(left_ckpt_path):
        try:
            unet.load_state_dict(torch.load(left_ckpt_path, map_location=device, weights_only=True), strict=True)
            print(
                f"✅ Loaded frozen LeftBrainCrystallizer from {left_ckpt_path} "
                f"(Mean={unet.manifold_mean.item():.4f}, Std={unet.manifold_std.item():.4f})"
            )
            loaded_left_ok = True
        except Exception as e:
            print(f"⚠️ Specified --left_brain_ckpt '{left_ckpt_path}' is incompatible ({e}); falling back to auto-discovery...")

    if not loaded_left_ok:
        preferred_prefixes = [args.left_brain_ckpt_prefix, "left_brain_s32_fullrank", "phase4_sentvec_flow", "hippo_flow_moss"]
        dedup_prefixes = list(dict.fromkeys([p for p in preferred_prefixes if p]))
        discovered_left = find_latest_left_brain_ckpt(
            unet=unet,
            base_dirs=[args.out_dir, "checkpoints/run", "checkpoints"],
            preferred_prefixes=dedup_prefixes,
            device=device,
        )
        if discovered_left:
            print(
                f"✅ Auto-discovered frozen LeftBrainCrystallizer from {discovered_left} "
                f"(Mean={unet.manifold_mean.item():.4f}, Std={unet.manifold_std.item():.4f})"
            )
        else:
            locked_ok = unet.bind_encoder_subspace(encoder)
            print(
                f"⚠️ WARNING: No compatible LeftBrainCrystallizer checkpoint found; "
                f"bound encoder subspace_proj={locked_ok}, manifold mean/std will auto-calibrate on Batch 0."
            )
    unet.eval()
    for p in unet.parameters():
        p.requires_grad = False

    translator_path = args.translator_ckpt
    if translator_path and ("*" in translator_path or "?" in translator_path):
        matched = sorted(glob.glob(translator_path))
        translator_path = matched[-1] if matched else None
    elif translator_path and os.path.isdir(translator_path):
        translator_path = os.path.join(translator_path, "translator.pt")
    if not translator_path:
        translator_path = find_latest_translator_ckpt(base_dirs=[args.out_dir, "checkpoints/run", "checkpoints"])

    translator = ContinuousToTextTranslatorCUDA(d_model=1024, vocab_size=tokenizer.vocab_size).to(device)
    if translator_path and os.path.exists(translator_path):
        translator.load_state_dict(torch.load(translator_path, map_location=device, weights_only=True))
        print(f"✅ Loaded frozen ContinuousToTextTranslatorCUDA from {translator_path}")
    else:
        print(f"⚠️ WARNING: No translator checkpoint found at '{translator_path}'; using untrained translator lens for testing.")
    translator.eval()
    for p in translator.parameters():
        p.requires_grad = False

    # 3. Trainable Right-Brain Modules (`MultiStreamRightBrain` v2 + `KnowledgeSelector`)
    right_brain = MultiStreamRightBrain(
        d_clause=1024,
        d_mamba=args.d_mamba,
        n_layers_main=args.n_layers_main,
        n_layers_aux=args.n_layers_aux,
        active_streams=active_streams,
        num_bands=args.num_subbands,
        sem_bands=args.sem_bands,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        num_branches=args.num_branches,
        train_ode_steps=args.train_ode_steps,
        win_weight=args.win_weight,
        prior_weight=args.prior_weight,
        stop_weight=args.stop_weight,
    ).to(device)

    knowledge_selector = KnowledgeSelector(
        d_clause=1024,
        d_proj=256,
    ).to(device)

    trainable_params = list(right_brain.parameters()) + list(knowledge_selector.parameters())
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

    total_horizon_steps = args.max_steps if args.max_steps > 0 else max(1, args.epochs * len(dataloader))
    scheduler = get_warmup_cosine_scheduler(
        optimizer=optimizer,
        warmup_steps=args.warmup_steps,
        max_steps=total_horizon_steps,
        min_lr_ratio=args.min_lr_ratio,
    )

    checkpointer = Checkpointer(out_dir=args.out_dir, prefix=args.ckpt_prefix, keep_last_k=args.keep_last_k)
    checkpointer.register_model("right_brain", right_brain)
    checkpointer.register_model("knowledge_selector", knowledge_selector)
    checkpointer.register_optimizer("engine_opt", optimizer)
    checkpointer.register_optimizer("lr_sched", scheduler)
    checkpointer.register_args(args)

    global_step = 0
    if args.auto_resume or args.resume_from:
        if args.resume_from and os.path.isdir(args.resume_from):
            compat_path, compat_step = args.resume_from, 0
        else:
            compat_path, compat_step = find_compatible_right_brain_ckpt(
                right_brain=right_brain,
                knowledge_selector=knowledge_selector,
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
                print(f"✅ Resumed Right-Brain training from step {global_step} ({compat_path})")
            except Exception as e:
                print(f"⚠️ Failed to resume from '{compat_path}' ({e}); starting fresh at step 0.")
                global_step = 0
                checkpointer.set_step(0)
        else:
            print(f"ℹ️ No compatible Right-Brain checkpoints found in '{args.out_dir}'. Starting fresh from step 0.")
            checkpointer.set_step(0)
    else:
        checkpointer.set_step(0)

    start_epoch = global_step // len(dataloader) if len(dataloader) > 0 else 0
    pad_id = tokenizer.pad_token_id
    n_rb_params = sum(p.numel() for p in right_brain.parameters())
    n_ks_params = sum(p.numel() for p in knowledge_selector.parameters())

    print(
        f"🚀 Starting Stage 2/3 Right-Brain v2 (#029) + KnowledgeSelector Training "
        f"(ActiveStreams={active_streams}{' -> single-stream warmup (group none)' if discourse_only else ''}, "
        f"d_mamba={args.d_mamba}, Layers=[main:{args.n_layers_main}, aux:{args.n_layers_aux}], "
        f"Params[RB={n_rb_params / 1e6:.1f}M, KS={n_ks_params / 1e6:.2f}M])"
    )
    print(
        f"   Flow: K={args.num_branches} branches x {args.train_ode_steps} ODE steps, win_weight={args.win_weight} | "
        f"Prior: bands={args.num_subbands}, sem_bands={args.sem_bands}, sigma=[{args.sigma_min}, {args.sigma_max}], "
        f"prior_w={args.prior_weight} | Stop: w={args.stop_weight}, thr={args.stop_threshold}"
    )
    print(
        f"   StreamDropout: full={args.p_full}, goal_only={args.p_goal_only}, "
        f"none={max(0.0, 1.0 - args.p_full - args.p_goal_only):.2f} | Jitter p={args.p_band_jitter} | "
        f"SelfFeed: p_max={args.p_self_max}, start={args.self_feed_start}, ramp={args.self_feed_ramp} | "
        f"Selector: clause_w={args.clause_selector_weight}, sample_w={args.sample_selector_weight}, "
        f"probe_noise={args.probe_noise_std} | ckpt_prefix={args.ckpt_prefix}"
    )

    dual_mode = args.planning_space in ("z_god", "s_pool")
    optimizer.zero_grad(set_to_none=True)
    last_log_time = time.time()
    last_log_step = global_step
    excluded_time = 0.0  # eval / checkpoint wall time excluded from s/step
    warned_missing_labels = False

    for epoch in range(start_epoch, args.epochs):
        resumed_loader = checkpointer.wrap_dataloader(dataloader, epoch, global_step)
        right_brain.train()
        knowledge_selector.train()

        for batch_idx, batch in enumerate(resumed_loader):
            w_goal = batch["w_goal"].to(device)                      # (B, G, L)
            goal_valid_mask = batch["goal_valid_mask"].to(device)    # (B, G)
            w_cand = batch["w_cand"].to(device)                      # (B, C, L)
            cand_pos_mask = batch["cand_pos_mask"].to(device)        # (B, C)
            cand_valid_mask = batch["cand_valid_mask"].to(device)    # (B, C)
            w_discourse = batch["w_discourse"].to(device)            # (B, T, L)
            discourse_valid_mask = batch["discourse_valid_mask"].to(device)  # (B, T)
            # #029 labels (absent only if an old dataset module is imported -> those losses are skipped)
            stop_mask = batch["discourse_stop_mask"].to(device) if "discourse_stop_mask" in batch else None  # (B, T)
            cand_clause_rel = batch["cand_clause_rel"].to(device) if "cand_clause_rel" in batch else None    # (B, T, C)
            cand_clause_rel_valid = (
                batch["cand_clause_rel_valid"].to(device) if "cand_clause_rel_valid" in batch else None      # (B, T)
            )
            if (stop_mask is None or cand_clause_rel is None) and not warned_missing_labels:
                print("⚠️ WARNING: batch lacks #029 labels (discourse_stop_mask / cand_clause_rel); "
                      "stop and per-clause selector losses are disabled. Is the #029 dataset merged?")
                warned_missing_labels = True

            B_cur = w_discourse.shape[0]
            with torch.no_grad():
                if dual_mode:
                    z_discourse, s_discourse = encode_clause_bank_dual(
                        encoder, unet, hippo_oracle, w_discourse, discourse_valid_mask, pad_id,
                        semantic_source=args.planning_space,
                    )
                    z_goal, s_goal = encode_clause_bank_dual(
                        encoder, unet, hippo_oracle, w_goal, goal_valid_mask, pad_id,
                        semantic_source=args.planning_space,
                    )
                    z_cand, s_cand = encode_clause_bank_dual(
                        encoder, unet, hippo_oracle, w_cand, cand_valid_mask, pad_id,
                        semantic_source=args.planning_space,
                    )
                    s_aux, _ = right_brain.pack_aux_streams(s_goal, goal_valid_mask, s_cand, cand_valid_mask)
                else:
                    z_discourse = encode_clause_bank_to_2d(
                        encoder, unet, hippo_oracle, w_discourse, discourse_valid_mask, pad_id
                    )  # (B, T, 1024)
                    z_goal = encode_clause_bank_to_2d(
                        encoder, unet, hippo_oracle, w_goal, goal_valid_mask, pad_id
                    )  # (B, G, 1024)
                    z_cand = encode_clause_bank_to_2d(
                        encoder, unet, hippo_oracle, w_cand, cand_valid_mask, pad_id
                    )  # (B, C, 1024)
                    s_discourse = None
                    s_aux = None

                # Masked mean of z_goal and lead question clause z_discourse[:, 0] (sample-level selector probe)
                goal_f = goal_valid_mask.unsqueeze(-1).to(dtype=z_goal.dtype)
                z_goal_mean = ((z_goal * goal_f).sum(dim=1) + z_discourse[:, 0, :]) / (
                    goal_f.sum(dim=1).clamp(min=0.0) + 1.0
                )

                # #029 companion streams (stream 0 = goal sequence, 1..C = each candidate clause, K+ and K-)
                # + per-sample stream dropout (0=full / 1=goal_only / 2=none).
                aux, aux_valid = right_brain.pack_aux_streams(z_goal, goal_valid_mask, z_cand, cand_valid_mask)
                group = sample_stream_groups(B_cur, args.p_full, args.p_goal_only, discourse_only, device)
                aux_valid = apply_stream_group_mask(aux_valid, group)

                # #029 exposure-bias robustness: high-band jitter + scheduled self-feeding of the main input.
                z_in = right_brain.jitter_high_bands(
                    z_discourse[:, :-1, :].clone(),
                    p=args.p_band_jitter,
                    sem_bands=right_brain.sem_bands,
                    num_bands=right_brain.num_bands,
                )
                s_in = s_discourse[:, :-1, :].clone() if dual_mode and s_discourse is not None else None
                p_self = compute_p_self(global_step, args.self_feed_start, args.self_feed_ramp, args.p_self_max)
                n_self_fed = 0
                calibrated = int(right_brain.flow_matcher.calib_steps.item()) > 0
                if dual_mode:
                    calibrated = calibrated and int(right_brain.s_calib_steps.item()) > 0
                if p_self > 0.0 and calibrated and z_in.shape[1] >= 2:
                    clause_valid = discourse_valid_mask[:, 1:-1].bool()  # clauses 1..T-2
                    replace = (torch.rand(clause_valid.shape, device=device) < p_self) & clause_valid
                    if dual_mode and s_discourse is not None and s_in is not None:
                        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                            z_samples, s_samples = right_brain.sample_next_clauses(
                                z_discourse[:, :-1, :],
                                aux,
                                aux_valid,
                                temperature=1.0,
                                ode_steps=args.train_ode_steps,
                                s_in=s_discourse[:, :-1, :],
                                s_aux=s_aux,
                                return_dual=True,
                            )
                        z_in[:, 1:, :] = torch.where(
                            replace.unsqueeze(-1), z_samples[:, :-1, :].to(dtype=z_in.dtype), z_in[:, 1:, :]
                        )
                        s_in[:, 1:, :] = torch.where(
                            replace.unsqueeze(-1), s_samples[:, :-1, :].to(dtype=s_in.dtype), s_in[:, 1:, :]
                        )
                    else:
                        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                            samples = right_brain.sample_next_clauses(
                                z_discourse[:, :-1, :],
                                aux,
                                aux_valid,
                                temperature=1.0,
                                ode_steps=args.train_ode_steps,
                            )  # samples[:, j] = predicted clause j+1 (denormalized z_clause space)
                        z_in[:, 1:, :] = torch.where(
                            replace.unsqueeze(-1), samples[:, :-1, :].to(dtype=z_in.dtype), z_in[:, 1:, :]
                        )
                    n_self_fed = int(replace.sum().item())

                # Per-clause selector probes: GT target clause + relative Gaussian noise (x per-row std).
                z_tgt = z_discourse[:, 1:, :]
                row_std = z_tgt.float().std(dim=-1, keepdim=True)
                z_probe = z_tgt + float(args.probe_noise_std) * row_std.to(dtype=z_tgt.dtype) * torch.randn_like(z_tgt)

            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                loss_rb, m = right_brain.compute_loss(
                    z_discourse_seq=z_discourse,
                    target_valid_mask=discourse_valid_mask[:, 1:],
                    aux=aux,
                    aux_valid=aux_valid,
                    stop_labels=stop_mask[:, 1:] if stop_mask is not None else None,
                    z_in_override=z_in,
                    sample_group=group,
                    return_metrics=True,
                    s_discourse_seq=s_discourse,
                    s_in_override=s_in,
                    s_aux=s_aux,
                    sem_self_cond_prob=args.sem_self_cond_prob,
                )
                if cand_clause_rel is not None:
                    row_valid = discourse_valid_mask[:, 1:].bool()
                    if cand_clause_rel_valid is not None:
                        row_valid = row_valid & cand_clause_rel_valid[:, 1:].bool()
                    loss_clause = knowledge_selector.compute_clause_selection_loss(
                        z_probe=z_probe,
                        z_candidates=z_cand,
                        rel_labels=cand_clause_rel[:, 1:, :],
                        cand_valid_mask=cand_valid_mask,
                        row_valid=row_valid,
                    )
                else:
                    loss_clause = loss_rb.new_zeros(())
                loss_sample = knowledge_selector.compute_selection_loss(
                    z_query=z_goal_mean,
                    z_candidates=z_cand,
                    pos_mask=cand_pos_mask,
                    cand_valid_mask=cand_valid_mask,
                )
                loss = (
                    loss_rb
                    + float(args.clause_selector_weight) * loss_clause
                    + float(args.sample_selector_weight) * loss_sample
                )
                loss_scaled = loss / max(1, args.accum_steps)

            if scaler is not None:
                scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()

            if (batch_idx + 1) % args.accum_steps == 0 or (batch_idx + 1) == len(resumed_loader):
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            global_step += 1
            checkpointer.set_step(global_step)

            is_eval_step = args.eval_interval > 0 and global_step % args.eval_interval == 0
            if global_step == 1 or global_step % max(1, args.log_interval) == 0 or is_eval_step:
                now = time.time()
                sec_per_step = (now - last_log_time - excluded_time) / max(1, global_step - last_log_step)
                last_log_time, last_log_step, excluded_time = now, global_step, 0.0
                cur_lr = optimizer.param_groups[0]["lr"]
                z_std_cal = right_brain.flow_matcher.z_std.mean().item()
                n_groups = [int((group == g).sum().item()) for g in (0, 1, 2)]

                def _grp(key: str) -> str:
                    return " / ".join(_fmt_opt(m.get(f"{key}_{name}")) for name in ("full", "goal_only", "none"))

                print(
                    f"Epoch {epoch + 1} | Step {global_step} | LR: {cur_lr:.2e} | {sec_per_step:.2f}s/step | "
                    f"Total: {loss.item():.4f} | Flow: {m['flow_loss']:.4f} "
                    f"(win {m['win_loss']:.4f}, other {m['other_loss']:.4f}) | PriorNLL: {m['prior_nll']:.4f} | "
                    f"Stop: {m['stop_loss']:.4f} (acc {_fmt_opt(m.get('stop_acc'))}) | "
                    f"ThoughtCos: {m['thought_cos']:.4f}"
                )
                print(
                    f"    σ_sem {m['sigma_sem']:.3f} σ_syn {m['sigma_syn']:.3f} | "
                    f"σ_sem[full/goal/none] {_grp('sigma_sem')} | σ_syn[full/goal/none] {_grp('sigma_syn')} | "
                    f"groups(n) {n_groups[0]}/{n_groups[1]}/{n_groups[2]}"
                )
                print(
                    f"    BranchDiv {m['branch_diversity']:.4f} | Winners {m['winner_hist']} | "
                    f"Gates mean {m['gate_mean']:.3f} valid {_fmt_opt(m.get('gate_valid_mean'))} "
                    f"(N={m['n_streams']}) | tanh(α_aux) {m['alpha_aux']:.3f} | "
                    f"L_clause {loss_clause.item():.4f} | L_sample {loss_sample.item():.4f} | "
                    f"p_self {p_self:.3f} (fed {n_self_fed}) | z_std {z_std_cal:.3f}"
                )
                check_low_band_saturation(m.get("sigma_band", []), right_brain.sem_bands, args.sigma_max, tag="train")
                if is_eval_step:
                    band_str = " ".join(f"{s:.2f}" for s in m.get("sigma_band", []))
                    print(f"    σ_band[0..{len(m.get('sigma_band', [])) - 1}]: {band_str}")

            if is_eval_step:
                t_eval0 = time.time()
                evaluate_dual_brain_rollout(
                    step=global_step,
                    right_brain=right_brain,
                    knowledge_selector=knowledge_selector,
                    unet=unet,
                    encoder=encoder,
                    hippo_oracle=hippo_oracle,
                    translator=translator,
                    tokenizer=tokenizer,
                    dataset=dataset,
                    benchmark_samples=benchmark_samples,
                    active_streams=active_streams,
                    args=args,
                    device=device,
                    amp_enabled=amp_enabled,
                    amp_dtype=amp_dtype,
                )
                right_brain.train()
                knowledge_selector.train()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # Exclude eval time from the next s/step measurement
                excluded_time += time.time() - t_eval0

            if args.save_steps and global_step % args.save_steps == 0:
                t_save0 = time.time()
                checkpointer.save(global_step)
                # Exclude checkpoint I/O from the next s/step measurement
                excluded_time += time.time() - t_save0

            if args.max_steps > 0 and global_step >= args.max_steps:
                break

        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    # Final save (skipped if the periodic save already wrote this exact step)
    if not (args.save_steps and global_step > 0 and global_step % args.save_steps == 0):
        checkpointer.save(global_step)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    train()
