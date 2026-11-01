import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import sys
import re
import math
import glob
import json
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

from model.phase1_charencoder_cuda import TinyCharEncoderCUDA
from model.continuous_translator_cuda import ContinuousToTextTranslatorCUDA
from model.hippo_oracle import HiPPOOracleCUDA
from model.flow_unet_cuda import LeftBrainCrystallizer

try:
    from model.bicameral_right_brain import MultiStreamRightBrain, KnowledgeSelector
    if not hasattr(MultiStreamRightBrain, "step_clause"):
        from model.bicameral_right_brain_027 import MultiStreamRightBrain, KnowledgeSelector
except ImportError:
    from model.bicameral_right_brain_027 import MultiStreamRightBrain, KnowledgeSelector

try:
    from training.data.knowledge_dataset import (
        BicameralClauseChainDataset,
        create_clause_chain_dataloader,
        CANONICAL_DOMAINS,
    )
except ImportError:
    from training.data.knowledge_dataset_027 import (
        BicameralClauseChainDataset,
        create_clause_chain_dataloader,
        CANONICAL_DOMAINS,
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
    prefix: str = "bicameral_right_brain",
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
    """Encodes token IDs `(N, L)` through the frozen Phase 1 `TinyCharEncoderCUDA` into `(N, L, 1024)`."""
    h = encoder.get_normalized_embeddings(token_ids)
    for layer in encoder.layers:
        h = layer(h, None)
    return encoder.final_ln(h)


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

    x_active = encode_tokens(encoder, flat_w[active_idx])  # (N_active, L, D)
    active_tok_mask = flat_tok_valid[active_idx]

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
    End-to-End Dual-Brain Autoregressive Multi-Clause Rollout Evaluation (#027):
      1. Encodes user question `w_goal` -> `z_goal` and candidate pool `w_cand` -> `z_cand`.
      2. Uses `knowledge_selector.select_active_clause_attractors(...)` to select top active
         knowledge clauses and construct `x_attractors = encode_tokens(encoder, w_attractors)` + `k_mask`,
         and builds `z_knowledge_active` for the Right Brain's Knowledge Stream.
      3. Steps `right_brain.step_clause(...)` autoregressively for `num_rollout_clauses` steps
         starting from `z_curr = z_discourse[:, 0]` (lead question clause):
           - Right Brain predicts next 2D clause vector `z_thought^(c) in R^1024`.
           - Frozen `LeftBrainCrystallizer` integrates `--left_ode_steps` Euler ODE steps conditioned on
             `h_potential = z_thought^(c)` and `k_attractors = unet.normalize(x_attractors)` to crystallize
             `x_clause_gen: (1, 32, 1024)`.
           - Frozen `Translator` decodes `x_clause_gen` into text clause `C_c`.
           - Closed-Loop Recrystallization Feedback (`--closed_loop_reencode`): re-encodes `x_clause_gen`
             via `hippo_oracle` (or feeds `z_thought^(c)`) as the next step's `z_curr`.
      4. Prints a diagnostic table per benchmark domain showing `[Question]`, `[Target Clauses]`,
         `[KnowledgeSelector]`, `[Oracle Left-Brain Decode]`, and `[Dual-Brain Rollout]`.
    """
    right_brain.eval()
    knowledge_selector.eval()
    unet.eval()

    pad_id = tokenizer.pad_token_id
    mode_tag = "Closed-Loop Re-Encode" if args.closed_loop_reencode else "Open-Loop Latent"
    print("\n" + "=" * 92)
    print(
        f"🧠 [Step {step}] End-to-End Dual-Brain Autoregressive Multi-Clause Rollout "
        f"(Streams={','.join(active_streams)} | Mode={mode_tag} | "
        f"RightODE={args.right_ode_steps} | LeftODE={args.left_ode_steps} | RolloutClauses={args.num_rollout_clauses})"
    )
    print("=" * 92)

    for dom_name, sample_idx in benchmark_samples:
        item = dataset[sample_idx]
        w_goal = item["w_goal"].unsqueeze(0).to(device)                  # (1, G, L)
        goal_valid = item["goal_valid_mask"].unsqueeze(0).to(device)     # (1, G)
        w_cand = item["w_cand"].unsqueeze(0).to(device)                  # (1, C, L)
        cand_pos = item["cand_pos_mask"].unsqueeze(0).to(device)         # (1, C)
        cand_valid = item["cand_valid_mask"].unsqueeze(0).to(device)     # (1, C)
        w_disc = item["w_discourse"].unsqueeze(0).to(device)             # (1, T, L)
        disc_valid = item["discourse_valid_mask"].unsqueeze(0).to(device)  # (1, T)

        z_goal = encode_clause_bank_to_2d(encoder, unet, hippo_oracle, w_goal, goal_valid, pad_id)
        z_cand = encode_clause_bank_to_2d(encoder, unet, hippo_oracle, w_cand, cand_valid, pad_id)
        z_disc = encode_clause_bank_to_2d(encoder, unet, hippo_oracle, w_disc, disc_valid, pad_id)

        goal_f = goal_valid.unsqueeze(-1).to(dtype=z_goal.dtype)
        z_query = ((z_goal * goal_f).sum(dim=1) + z_disc[:, 0, :]) / (goal_f.sum(dim=1).clamp(min=0.0) + 1.0)

        # 1. KnowledgeSelector evaluation & span-preserving attractor construction
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

        w_attractors, k_mask, top_scores = knowledge_selector.select_active_clause_attractors(
            z_query=z_query,
            z_candidates=z_cand,
            w_candidates=w_cand,
            w_goal=w_goal,
            cand_valid_mask=cand_valid,
            top_m_clauses=args.top_m_clauses,
            min_score=args.min_selector_score,
            pad_token_id=pad_id,
        )
        z_knowledge_active, _ = knowledge_selector.select_active_knowledge_vectors(
            z_query=z_query,
            z_candidates=z_cand,
            cand_valid_mask=cand_valid,
            top_m_clauses=args.top_m_clauses,
            min_score=args.min_selector_score,
        )

        x_attractors = encode_tokens(encoder, w_attractors)
        x_attractors_norm = unet.normalize(x_attractors)

        # 2. Ground-truth target clauses & Oracle Left-Brain Decode
        num_valid_disc = int(disc_valid[0].sum().item())
        num_gt_ans = max(1, min(num_valid_disc - 1, args.num_rollout_clauses))
        gt_clause_strs: List[str] = []
        oracle_clause_strs: List[str] = []

        for c_i in range(1, num_gt_ans + 1):
            gt_ids = w_disc[0, c_i].tolist()
            gt_clause_strs.append(decode_clause_clean(tokenizer, gt_ids))

            z_gt_c = z_disc[:, c_i, :]
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                x_oracle_c = crystallize_clause_euler_ode(
                    unet=unet,
                    z_clause=z_gt_c,
                    x_attractors_norm=x_attractors_norm,
                    k_mask=k_mask,
                    steps=args.left_ode_steps,
                    seq_len=args.seq_len,
                )
            if translator is not None:
                ora_ids = translator.decode_greedy(x_oracle_c.float())[0].tolist()
                oracle_clause_strs.append(decode_clause_clean(tokenizer, ora_ids))
            else:
                oracle_clause_strs.append("<no-translator>")

        # 3. Autoregressive Dual-Brain Rollout (`right_brain.step_clause` -> `LeftBrainCrystallizer` -> `Translator`)
        rb_states: Optional[Dict[str, Any]] = None
        z_curr = z_disc[:, 0, :]  # Clause 0: lead question departure clause
        rollout_clause_strs: List[str] = []
        clause1_cos = 0.0

        for c_step in range(1, args.num_rollout_clauses + 1):
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                z_thought_c, rb_states = right_brain.step_clause(
                    z_curr=z_curr,
                    states=rb_states,
                    z_knowledge=z_knowledge_active,
                    z_goal=z_goal,
                    active_streams=active_streams,
                    ode_steps=args.right_ode_steps,
                    temperature=args.temperature,
                )
                x_clause_gen = crystallize_clause_euler_ode(
                    unet=unet,
                    z_clause=z_thought_c,
                    x_attractors_norm=x_attractors_norm,
                    k_mask=k_mask,
                    steps=args.left_ode_steps,
                    seq_len=args.seq_len,
                )

            if c_step == 1 and num_valid_disc >= 2:
                clause1_cos = F.cosine_similarity(
                    z_thought_c.float(), z_disc[:, 1, :].float(), dim=-1
                ).mean().item()

            if translator is not None:
                pred_ids = translator.decode_greedy(x_clause_gen.float())[0].tolist()
                pred_str = decode_clause_clean(tokenizer, pred_ids)
                rollout_clause_strs.append(pred_str)
                gen_valid_mask = (
                    torch.tensor(pred_ids, device=device, dtype=torch.long).unsqueeze(0) != pad_id
                )
            else:
                rollout_clause_strs.append(f"<z_norm={z_thought_c.norm().item():.2f}>")
                gen_valid_mask = torch.ones((1, args.seq_len), device=device, dtype=torch.bool)

            if args.closed_loop_reencode:
                x_gen_norm = unet.normalize(x_clause_gen.float())
                z_reencoded = hippo_oracle(
                    x_gen_norm,
                    return_sequence=False,
                    valid_mask=gen_valid_mask,
                    subspace_proj=unet.subspace_proj,
                )
                z_curr = z_reencoded.to(dtype=z_disc.dtype)
            else:
                z_curr = z_thought_c

        top_scores_str = ", ".join(f"{s:.2f}" for s in top_scores[0].tolist())
        active_tokens_cnt = int(k_mask.sum().item())

        print(f"📌 Domain [{dom_name}] (Sample #{sample_idx})")
        print(f"   [Question]:                {item['question_text']}")
        print(f"   [Target Clauses]:          {' | '.join(gt_clause_strs)}")
        print(
            f"   [KnowledgeSelector]:       Top-{top_scores.shape[1]} Scores=[{top_scores_str}] | "
            f"K+/K- Acc={sel_acc:.1f}% | Active Attractor Tokens={active_tokens_cnt}"
        )
        print(f"   [Oracle Left-Brain Decode]: {' | '.join(oracle_clause_strs)}")
        print(
            f"   [Dual-Brain Rollout]:      {' | '.join(rollout_clause_strs)} "
            f"(Clause-1 CosSim={clause1_cos:.4f})"
        )
        print("-" * 92)

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
    parser.add_argument("--n_layers_discourse", type=int, default=4, help="Number of Mamba layers in Discourse Stream")
    parser.add_argument("--n_layers_knowledge", type=int, default=2, help="Number of Mamba layers in Knowledge Stream")
    parser.add_argument("--n_layers_goal", type=int, default=2, help="Number of Mamba layers in Goal Stream")
    parser.add_argument(
        "--active_streams",
        type=str,
        default="discourse,knowledge,goal",
        help="Comma-separated active Right-Brain streams: 'discourse' or 'discourse,knowledge,goal'",
    )
    parser.add_argument("--max_goal_clauses", type=int, default=4, help="Max question clauses in Goal Stream")
    parser.add_argument("--max_cand_clauses", type=int, default=8, help="Max candidate knowledge clauses in pool")
    parser.add_argument("--max_discourse_clauses", type=int, default=8, help="Max progressive clauses in Discourse Stream")
    parser.add_argument("--min_clause_chars", type=int, default=6, help="Min characters per punctuation clause")
    parser.add_argument("--max_clause_chars", type=int, default=28, help="Max characters per punctuation clause")
    parser.add_argument("--selector_weight", type=float, default=0.5, help="Loss weight for KnowledgeSelector BCE+margin loss")
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
    parser.add_argument("--num_rollout_clauses", type=int, default=4, help="Number of clauses to rollout during evaluation")
    parser.add_argument("--temperature", type=float, default=0.75, help="Prior noise temperature for ThoughtFlowMatcher sampling")
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
    parser.add_argument("--mamba_layers", type=int, default=None, help="Legacy alias for --n_layers_discourse")

    parser.set_defaults(
        seq_len=32,
        batch_size=32,
        warmup_steps=500,
        save_steps=1000,
        ckpt_prefix="bicameral_right_brain",
    )

    args = parser.parse_args()
    if args.seq_len > 64:
        print(f"ℹ️ Overriding legacy --seq_len {args.seq_len} -> 32 to match Punctuation-Clause LeftBrainCrystallizer.")
        args.seq_len = 32
    if args.mamba_layers is not None:
        args.n_layers_discourse = int(args.mamba_layers)
    if args.max_know_streams is not None:
        args.top_m_clauses = int(args.max_know_streams)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    active_streams = tuple(s.strip() for s in args.active_streams.split(",") if s.strip())
    if not active_streams:
        active_streams = ("discourse",)

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
    )

    benchmark_samples = select_fixed_benchmark_indices(dataset, seed=42, max_domains=6)
    if benchmark_samples:
        bench_desc = ", ".join(f"{dom}:#{idx}" for dom, idx in benchmark_samples)
        print(f"📌 Fixed Cross-Domain Rollout Benchmark Samples: {bench_desc}")

    # 2. Frozen Left-Brain & Perception Modules
    encoder = TinyCharEncoderCUDA(vocab_size=tokenizer.vocab_size).to(device)
    if os.path.exists(args.encoder_ckpt):
        encoder.load_state_dict(torch.load(args.encoder_ckpt, map_location=device, weights_only=True))
        print(f"✅ Loaded frozen TinyCharEncoderCUDA from {args.encoder_ckpt}")
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
            print("⚠️ WARNING: No compatible LeftBrainCrystallizer checkpoint found; manifold will auto-calibrate on Batch 0.")
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

    # 3. Trainable Right-Brain Modules (`MultiStreamRightBrain` + `KnowledgeSelector`)
    right_brain = MultiStreamRightBrain(
        d_clause=1024,
        d_mamba=args.d_mamba,
        n_layers_discourse=args.n_layers_discourse,
        n_layers_knowledge=args.n_layers_knowledge,
        n_layers_goal=args.n_layers_goal,
        active_streams=active_streams,
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

    print(
        f"🚀 Starting Stage 2/3 Modular Bicameral Right-Brain + KnowledgeSelector Training "
        f"(ActiveStreams={active_streams}, d_mamba={args.d_mamba}, "
        f"Layers=[D:{args.n_layers_discourse}, K:{args.n_layers_knowledge}, G:{args.n_layers_goal}], "
        f"SelectorWeight={args.selector_weight:.2f})..."
    )

    optimizer.zero_grad(set_to_none=True)

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

            with torch.no_grad():
                z_discourse = encode_clause_bank_to_2d(
                    encoder, unet, hippo_oracle, w_discourse, discourse_valid_mask, pad_id
                )  # (B, T, 1024)
                z_goal = encode_clause_bank_to_2d(
                    encoder, unet, hippo_oracle, w_goal, goal_valid_mask, pad_id
                )  # (B, G, 1024)
                z_cand = encode_clause_bank_to_2d(
                    encoder, unet, hippo_oracle, w_cand, cand_valid_mask, pad_id
                )  # (B, C, 1024)

                # Masked mean of z_goal and lead question clause z_discourse[:, 0]
                goal_f = goal_valid_mask.unsqueeze(-1).to(dtype=z_goal.dtype)
                z_goal_mean = ((z_goal * goal_f).sum(dim=1) + z_discourse[:, 0, :]) / (
                    goal_f.sum(dim=1).clamp(min=0.0) + 1.0
                )

                # Build ordered active knowledge clause sequence `z_knowledge_seq` for Knowledge Stream Mamba
                B_cur, C_cur, D_cur = z_cand.shape
                M_know = min(max(1, args.top_m_clauses), C_cur)
                pos_priority = torch.where(
                    cand_valid_mask.bool(),
                    cand_pos_mask.float(),
                    torch.full_like(cand_pos_mask, -1e4),
                )
                _, top_pos_idx = torch.topk(pos_priority, k=M_know, dim=-1)
                top_pos_idx_sorted, _ = torch.sort(top_pos_idx, dim=-1)
                gather_k = top_pos_idx_sorted.unsqueeze(-1).expand(B_cur, M_know, D_cur)
                z_knowledge_seq = torch.gather(z_cand, dim=1, index=gather_k)
                gathered_valid = torch.gather(cand_valid_mask.bool(), dim=1, index=top_pos_idx_sorted)
                z_knowledge_seq = z_knowledge_seq * gathered_valid.unsqueeze(-1).to(dtype=z_knowledge_seq.dtype)

            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                loss_flow, flow_metrics = right_brain.compute_loss(
                    z_discourse_seq=z_discourse,
                    z_knowledge_seq=z_knowledge_seq,
                    z_goal_seq=z_goal,
                    target_valid_mask=discourse_valid_mask[:, 1:],
                    return_metrics=True,
                )
                loss_sel = knowledge_selector.compute_selection_loss(
                    z_query=z_goal_mean,
                    z_candidates=z_cand,
                    pos_mask=cand_pos_mask,
                    cand_valid_mask=cand_valid_mask,
                )
                loss = loss_flow + float(args.selector_weight) * loss_sel
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

            if global_step == 1 or global_step % max(1, args.log_interval) == 0:
                cur_lr = optimizer.param_groups[0]["lr"]
                z_std_cal = right_brain.flow_matcher.z_std.item()
                ak = torch.tanh(right_brain.alpha_k).item()
                ag = torch.tanh(right_brain.alpha_g).item()
                print(
                    f"Epoch {epoch + 1} | Step {global_step} | LR: {cur_lr:.2e} | "
                    f"Total: {loss.item():.4f} | Flow: {loss_flow.item():.4f} "
                    f"(Vel MSE: {flow_metrics['vel_loss']:.4f}, Thought Cos: {flow_metrics['thought_cos']:.4f}) | "
                    f"Sel BCE: {loss_sel.item():.4f} | Gates(k={ak:.3f}, g={ag:.3f}) | z_std: {z_std_cal:.3f}"
                )

            if args.eval_interval > 0 and global_step % args.eval_interval == 0:
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

            if args.save_steps and global_step % args.save_steps == 0:
                checkpointer.save(global_step)

            if args.max_steps > 0 and global_step >= args.max_steps:
                break

        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    checkpointer.save(global_step)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    train()
