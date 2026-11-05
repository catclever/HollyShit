# =====================================================================================================
# [Task #033: Phase 5 Stage-1 Single-Stream Right-Brain Trainer]
# 1. What: trains `MultiStreamRightBrain` (#029 v2 architecture: K-branch soft-competition flow,
#    32-band adaptive NLL prior, stop head) as a PURE SINGLE-STREAM model (aux=None, N=0) on the
#    clause-stream data of #032 (`training.data.clause_stream_dataset`):
#      main stream = context prefix clauses + target clauses (T = --max_clauses, default 32; L = 32 chars);
#      losses only on target rows (`target_mask`), stop BCE only on `stop_valid` rows (QA answers;
#      continuation rows have no stop label). QA / continuation are mixed by --cont_ratio.
#    The KnowledgeSelector is NOT trained here (see training/train_phase5_selector.py).
# 2. Why: staged plan (design/tentative/2026-11-02_phase5_staged_training_plan.md, user-approved):
#    certainty must come from the main stream alone first; style is an IMPLICIT condition carried by the
#    longer context (no style/source label is ever a model input — `source_id` is used for grouped
#    logging only). Companion streams (Stage 3) attach later as a zero-initialised identity branch
#    (`alpha_aux`=0 + zero PriorHead aux columns, see model/bicameral_right_brain.py #033 notes).
#    Kept from #029 (design/tentative/2026-11-02_phase5_right_brain_v2_adaptive_prior.md): K-way flow
#    65/35, band NLL prior (sigma_max 3.0), high-band jitter, scheduled self-feeding.
# 3. Observability: sigma_sem / sigma_syn overall + per source (qa / cont), winner histogram, branch
#    diversity, stop acc / recall, p_self, s/step (eval + checkpoint I/O excluded), low-band saturation
#    warning, optional --profile (CUDA-synchronised per-module timing: data / encode / self_feed /
#    mamba_prior / flow / backward, and eval wall time).
#    Eval on a fixed deterministic benchmark (3 QA + 3 continuation): rollout from the context prefix
#    with stop head (attractors none | context via a trained selector), diversity demo, and the
#    "sigma vs context length" curve (implicit-condition evidence: sigma should shrink as k grows).
# 4. Caveats for subsequent development:
#    - Imports use FINAL module names (`model.bicameral_right_brain`, `training.data.clause_stream_dataset`).
#      Shared helpers (encode / ckpt discovery / scheduler / saturation warning) are imported from the
#      #029 trainer `training.train_phase5_mamba_flow_cuda`; keep that module importable (it imports
#      `training.data.knowledge_dataset`).
#    - Self-feeding replaces ONLY target rows of the main-stream input (context rows stay ground truth),
#      and starts only after the flow matcher's z calibration ran (calib_steps > 0).
#    - Checkpoints contain right_brain (+ optimizer / scheduler) only, prefix `right_brain_stage1`.
#      `--continue` / `--auto_resume` strict-loads the newest compatible `<prefix>_step_*` directory.
#    - `--eval_attractors context` needs `--selector_ckpt` (a #033 selector with sink head); otherwise
#      the eval silently falls back to `none`.
#    - Pool clauses generated during rollout are added to the attractor pool only when a translator is
#      loaded (attractors need token ids).
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
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.core.char_tokenizer import CharTokenizer
from training.core.args import get_training_parser
from training.core.checkpoint import Checkpointer

from model.phase1_charencoder_cuda import TinyCharEncoderCUDA
from model.continuous_translator_cuda import ContinuousToTextTranslatorCUDA
from model.hippo_oracle import HiPPOOracleCUDA
from model.flow_unet_cuda import LeftBrainCrystallizer
from model.bicameral_right_brain import (
    MultiStreamRightBrain,
    KnowledgeSelector,
    band_gaussian_nll,
    crystallize_with_optional_attractors,
)
from training.data.clause_stream_dataset import create_clause_stream_dataloader
from training.train_phase5_mamba_flow_cuda import (
    compute_p_self,
    _fmt_opt,
    check_low_band_saturation,
    get_warmup_cosine_scheduler,
    find_latest_left_brain_ckpt,
    find_latest_translator_ckpt,
    decode_clause_clean,
    encode_tokens,
    encode_clause_bank_to_2d,
    _sigma_sem_syn,
)

SOURCE_NAMES: Dict[int, str] = {0: "qa", 1: "cont"}


# ============================================================================ generic helpers
def split_paths(p: Optional[Union[str, Sequence[str]]]) -> List[str]:
    """'a.jsonl,b.jsonl' / list / None -> list of non-empty paths."""
    if p is None:
        return []
    if isinstance(p, (list, tuple)):
        return [str(x).strip() for x in p if str(x).strip()]
    return [x.strip() for x in str(p).split(",") if x.strip()]


def setup_amp(mixed_precision: str, device: torch.device) -> Tuple[bool, torch.dtype, Optional[Any]]:
    """Same policy as #029: bf16 if supported, else fp16 + GradScaler; CPU -> fp32."""
    if device.type == "cuda" and mixed_precision != "no":
        if mixed_precision == "bf16" and torch.cuda.is_bf16_supported():
            print("⚡ Mixed Precision: Enabled (Native BFloat16)")
            return True, torch.bfloat16, None
        print("⚡ Mixed Precision: Enabled (Float16 + GradScaler)")
        return True, torch.float16, torch.cuda.amp.GradScaler()
    print("⚡ Mixed Precision: Disabled (FP32)")
    return False, torch.float32, None


def resolve_ckpt_file(path: Optional[str], filename: str) -> Optional[str]:
    """Resolves a glob / directory / file argument to an existing checkpoint file (or None)."""
    if not path:
        return None
    if "*" in path or "?" in path:
        matched = sorted(glob.glob(path))
        path = matched[-1] if matched else None
        if not path:
            return None
    if os.path.isdir(path):
        path = os.path.join(path, filename)
    return path if os.path.isfile(path) else None


class Profiler:
    """CUDA-synchronised section timer. `mark(name)` attributes the time since the previous mark to `name`."""

    def __init__(self, enabled: bool, device: torch.device):
        self.enabled = bool(enabled)
        self.device = device
        self.acc: Dict[str, float] = {}
        self.n_steps = 0
        self._t = None

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def start(self) -> None:
        if self.enabled:
            self._sync()
            self._t = time.perf_counter()

    def mark(self, name: str) -> None:
        if not self.enabled or self._t is None:
            return
        self._sync()
        now = time.perf_counter()
        self.acc[name] = self.acc.get(name, 0.0) + (now - self._t)
        self._t = now

    def end_step(self) -> None:
        if self.enabled:
            self.n_steps += 1

    def add(self, name: str, seconds: float) -> None:
        if self.enabled:
            self.acc[name] = self.acc.get(name, 0.0) + float(seconds)

    def report_and_reset(self) -> str:
        if not self.enabled or self.n_steps == 0:
            return ""
        n = self.n_steps
        parts = [f"{k}={1000.0 * v / n:.1f}ms" for k, v in self.acc.items()]
        total = sum(v for k, v in self.acc.items() if k != "eval") / n
        self.acc, self.n_steps = {}, 0
        return f"    ⏱ profile (avg/step over {n}): " + " | ".join(parts) + f" | sum(train)={1000.0 * total:.1f}ms"


def find_compatible_ckpt(
    models: Dict[str, nn.Module],
    out_dir: str,
    prefix: str,
    device: Union[str, torch.device] = "cpu",
) -> Tuple[Optional[str], int]:
    """
    Newest `<prefix>_step_*` / `<prefix>_latest_emergency` directory in `out_dir` whose `<name>.pt` files
    strictly match every registered model (the FIRST model's file is required, others optional).
    """
    if not out_dir or not os.path.isdir(out_dir):
        return None, 0
    names = list(models.keys())
    cands: List[Tuple[int, str]] = []
    for d in os.listdir(out_dir):
        p = os.path.join(out_dir, d)
        if not os.path.isdir(p):
            continue
        if prefix and not (d.startswith(f"{prefix}_step_") or d == f"{prefix}_latest_emergency"):
            continue
        m = re.search(r"step_(\d+)", d)
        step = int(m.group(1)) if m else 0
        if os.path.isfile(os.path.join(p, f"{names[0]}.pt")):
            cands.append((step, p))
    cands.sort(key=lambda x: x[0], reverse=True)
    for step, p in cands:
        try:
            for name, mod in models.items():
                f = os.path.join(p, f"{name}.pt")
                if os.path.isfile(f):
                    mod.load_state_dict(torch.load(f, map_location=device, weights_only=True), strict=True)
                elif name == names[0]:
                    raise FileNotFoundError(f)
            print(f"✅ Verified compatible checkpoint '{os.path.basename(p)}' (step {step})")
            return p, step
        except Exception as e:
            print(f"ℹ️ Skipping incompatible checkpoint '{os.path.basename(p)}' (step {step}): {e}")
    return None, 0


def resume_training(
    args, checkpointer: Checkpointer, models: Dict[str, nn.Module], scheduler, device: torch.device,
) -> int:
    """#029-style resume: explicit --resume_from dir, else strict-compat discovery with the current prefix."""
    if not (args.auto_resume or args.resume_from):
        checkpointer.set_step(0)
        return 0
    if args.resume_from and os.path.isdir(args.resume_from):
        path, step = args.resume_from, 0
    else:
        path, step = find_compatible_ckpt(models, args.out_dir, args.ckpt_prefix, device)
    if not path:
        print(f"ℹ️ No compatible '{args.ckpt_prefix}' checkpoint in '{args.out_dir}'. Starting fresh from step 0.")
        checkpointer.set_step(0)
        return 0
    try:
        loaded = checkpointer.load(path)
        global_step = loaded if loaded > 0 else step
        rb = models.get("right_brain")
        fm = getattr(rb, "flow_matcher", None) if rb is not None else None
        if fm is not None and getattr(fm, "_upgraded_from_v1", False):
            import shutil
            for opt in checkpointer._optimizers.values():
                if hasattr(opt, "state") and isinstance(opt.state, dict):
                    opt.state.clear()
            base_lr = float(getattr(args, "lr", 2e-4))
            for opt in checkpointer._optimizers.values():
                if hasattr(opt, "param_groups"):
                    for pg in opt.param_groups:
                        pg["lr"] = base_lr
                        pg["initial_lr"] = base_lr
            if scheduler is not None:
                scheduler.base_lrs = [base_lr for _ in scheduler.base_lrs]
                scheduler.last_epoch = 0
                if hasattr(scheduler, "lr_lambdas"):
                    for pg, lmbda in zip(scheduler.optimizer.param_groups, scheduler.lr_lambdas):
                        pg["lr"] = base_lr * float(lmbda(0))
                    scheduler._last_lr = [pg["lr"] for pg in scheduler.optimizer.param_groups]
            if args.out_dir and os.path.isdir(args.out_dir) and args.ckpt_prefix:
                archive_dir = os.path.join(args.out_dir, f"{args.ckpt_prefix}_v1_archive_step_{global_step}")
                try:
                    if os.path.isdir(path) and not os.path.exists(archive_dir):
                        shutil.move(path, archive_dir)
                        print(f"📦 Archived v1 checkpoint '{os.path.basename(path)}' -> '{os.path.basename(archive_dir)}'")
                    for d in os.listdir(args.out_dir):
                        dp = os.path.join(args.out_dir, d)
                        if os.path.isdir(dp) and d.startswith(f"{args.ckpt_prefix}_step_"):
                            shutil.rmtree(dp, ignore_errors=True)
                except Exception as arch_err:
                    print(f"⚠️ Could not archive old v1 checkpoints: {arch_err}")
            fm._upgraded_from_v1 = False
            checkpointer.set_step(0)
            checkpointer.save(0)
            print(
                f"🔄 Upgraded v1 checkpoint from step {global_step} ({path}) to Flow-v2: "
                f"preserved Mamba/PriorHead/StopHead + z_mean/z_std, re-initialized ThoughtFlowMatcher, "
                f"and reset step counter -> 0 with fresh LR schedule."
            )
            return 0
        if getattr(rb, "_upgraded_context_lookback", False):
            # #038: When resuming a pre-#038 checkpoint that lacked `context_lookback.*`, restore the saved
            # AdamW state for all existing parameters (indices 0..P_old-1) and leave `context_lookback.*` fresh.
            opt_file = os.path.join(path, "engine_opt.pt")
            opt = checkpointer._optimizers.get("engine_opt")
            if opt is not None and os.path.isfile(opt_file):
                try:
                    opt_sd = torch.load(opt_file, map_location=device, weights_only=True)
                    if (
                        isinstance(opt_sd, dict)
                        and len(opt_sd.get("param_groups", [])) == 1
                        and len(opt.param_groups) == 1
                    ):
                        n_old = len(opt_sd["param_groups"][0].get("params", []))
                        n_new = len(opt.param_groups[0].get("params", []))
                        if 0 < n_old < n_new:
                            opt_sd["param_groups"][0]["params"] = list(range(n_new))
                            opt.load_state_dict(opt_sd)
                            print(
                                f"🔄 Restored AdamW state for {n_old}/{n_new} parameters "
                                f"(initialized fresh state for {n_new - n_old} CausalContextLookback params)."
                            )
                except Exception as opt_err:
                    print(f"⚠️ Could not partially restore optimizer state: {opt_err}")
            rb._upgraded_context_lookback = False
        checkpointer.set_step(global_step)
        if not os.path.exists(os.path.join(path, "lr_sched.pt")) and global_step > 0:
            scheduler.step(global_step)
        print(f"✅ Resumed from step {global_step} ({path})")
        return global_step
    except Exception as e:
        print(f"⚠️ Failed to resume from '{path}' ({e}); starting fresh at step 0.")
        checkpointer.set_step(0)
        return 0


def load_frozen_modules(
    args, tokenizer: CharTokenizer, device: torch.device, need_translator: bool = True,
) -> Tuple[TinyCharEncoderCUDA, HiPPOOracleCUDA, LeftBrainCrystallizer, Optional[ContinuousToTextTranslatorCUDA]]:
    """Frozen encoder / HiPPO / left brain (+ translator) with #029-compatible discovery and fallbacks."""
    encoder = TinyCharEncoderCUDA(vocab_size=tokenizer.vocab_size).to(device)
    if args.encoder_ckpt and os.path.exists(args.encoder_ckpt):
        encoder.load_state_dict(torch.load(args.encoder_ckpt, map_location=device, weights_only=True))
        print(f"✅ Loaded frozen TinyCharEncoderCUDA from {args.encoder_ckpt}")
    else:
        print(f"⚠️ WARNING: Encoder checkpoint '{args.encoder_ckpt}' not found! Using random weights (testing only).")
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    hippo = HiPPOOracleCUDA(hidden_dim=1024, N=64, num_subbands=args.num_subbands).to(device)
    hippo.eval()

    unet = LeftBrainCrystallizer(hidden_dim=1024, cond_dim=1024, num_layers=6, num_subbands=args.num_subbands).to(device)
    left_path = resolve_ckpt_file(args.left_brain_ckpt, "unet.pt")
    loaded = False
    if left_path:
        try:
            unet.load_state_dict(torch.load(left_path, map_location=device, weights_only=True), strict=True)
            print(f"✅ Loaded frozen LeftBrainCrystallizer from {left_path} "
                  f"(Mean={unet.manifold_mean.item():.4f}, Std={unet.manifold_std.item():.4f})")
            loaded = True
        except Exception as e:
            print(f"⚠️ --left_brain_ckpt '{left_path}' incompatible ({e}); falling back to auto-discovery...")
    if not loaded:
        prefs = list(dict.fromkeys([p for p in [args.left_brain_ckpt_prefix, "left_brain_s32_fullrank",
                                                "phase4_sentvec_flow", "hippo_flow_moss"] if p]))
        found = find_latest_left_brain_ckpt(unet, base_dirs=[args.out_dir, "checkpoints/run", "checkpoints"],
                                            preferred_prefixes=prefs, device=device)
        if found:
            print(f"✅ Auto-discovered frozen LeftBrainCrystallizer from {found}")
        else:
            print("⚠️ WARNING: No compatible LeftBrainCrystallizer checkpoint; manifold will auto-calibrate on batch 0.")
    unet.eval()
    for p in unet.parameters():
        p.requires_grad = False

    translator = None
    if need_translator:
        tr_path = resolve_ckpt_file(args.translator_ckpt, "translator.pt")
        if not tr_path:
            tr_path = find_latest_translator_ckpt(base_dirs=[args.out_dir, "checkpoints/run", "checkpoints"])
        if tr_path and os.path.exists(tr_path):
            translator = ContinuousToTextTranslatorCUDA(d_model=1024, vocab_size=tokenizer.vocab_size).to(device)
            translator.load_state_dict(torch.load(tr_path, map_location=device, weights_only=True))
            translator.eval()
            for p in translator.parameters():
                p.requires_grad = False
            print(f"✅ Loaded frozen ContinuousToTextTranslatorCUDA from {tr_path}")
        else:
            print("⚠️ WARNING: No translator checkpoint found; eval prints vector stats instead of text.")
    return encoder, hippo, unet, translator


def load_selector(path: Optional[str], device: torch.device) -> Optional[KnowledgeSelector]:
    f = resolve_ckpt_file(path, "knowledge_selector.pt")
    if not f:
        if path:
            print(f"⚠️ --selector_ckpt '{path}' not found; context attractors disabled.")
        return None
    sel = KnowledgeSelector(d_clause=1024, d_proj=256).to(device)
    sel.load_state_dict(torch.load(f, map_location=device, weights_only=True), strict=True)
    sel.eval()
    for p in sel.parameters():
        p.requires_grad = False
    print(f"✅ Loaded frozen KnowledgeSelector from {f}")
    return sel


def build_right_brain_from_args(a: Dict[str, Any]) -> MultiStreamRightBrain:
    """Constructs the Stage-1 right brain from a dict of (saved) training args (missing keys -> defaults)."""
    def g(k, d):
        v = a.get(k, d)
        return d if v is None else v
    return MultiStreamRightBrain(
        d_clause=1024,
        d_mamba=int(g("d_mamba", 1024)),
        n_layers_main=int(g("n_layers_main", 4)),
        n_layers_aux=int(g("n_layers_aux", 2)),
        active_streams=("discourse",),
        num_bands=int(g("num_subbands", 32)),
        sem_bands=int(g("sem_bands", 6)),
        sigma_min=float(g("sigma_min", 0.05)),
        sigma_max=float(g("sigma_max", 3.0)),
        num_branches=int(g("num_branches", 4)),
        train_ode_steps=int(g("train_ode_steps", 3)),
        win_weight=float(g("win_weight", 0.65)),
        prior_weight=float(g("prior_weight", 1.0)),
        stop_weight=float(g("stop_weight", 0.2)),
    )


def load_right_brain(path: Optional[str], device: torch.device) -> Optional[MultiStreamRightBrain]:
    """Loads a frozen Stage-1 right brain; architecture is read from the ckpt dir's training_args.json."""
    f = resolve_ckpt_file(path, "right_brain.pt")
    if not f:
        if path:
            print(f"⚠️ --right_brain_ckpt '{path}' not found; right-brain probes disabled.")
        return None
    saved: Dict[str, Any] = {}
    args_file = os.path.join(os.path.dirname(f), "training_args.json")
    if os.path.isfile(args_file):
        with open(args_file, "r") as fh:
            saved = json.load(fh)
    rb = build_right_brain_from_args(saved).to(device)
    rb.load_state_dict(torch.load(f, map_location=device, weights_only=True), strict=True)
    rb.eval()
    for p in rb.parameters():
        p.requires_grad = False
    print(f"✅ Loaded frozen Right-Brain from {f} (d_mamba={rb.d_mamba})")
    return rb


def _scalar(v: Any, default: int = 0) -> int:
    if v is None:
        return default
    if torch.is_tensor(v):
        return int(v.reshape(-1)[0].item())
    return int(v)


def select_benchmark_items(
    dataset, n_qa: int = 3, n_cont: int = 3, seed: int = 42, max_scan: int = 4000,
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Deterministically picks `n_qa` QA + `n_cont` continuation samples (with >= 1 context row and >= 1 target
    row) and CACHES the item dicts (the dataset may crop continuation windows randomly per __getitem__;
    RNG state is seeded during selection and restored afterwards).
    """
    py_state, torch_state = random.getstate(), torch.get_rng_state()
    try:
        import numpy as np
        np_state = np.random.get_state()
        np.random.seed(seed)
    except Exception:
        np, np_state = None, None
    random.seed(seed)
    torch.manual_seed(seed)
    out: Dict[int, List[Tuple[str, Dict[str, Any]]]] = {0: [], 1: []}
    want = {0: n_qa, 1: n_cont}
    try:
        n = len(dataset)
        order = list(range(n))
        random.Random(seed).shuffle(order)
        for idx in order[:max_scan]:
            if all(len(out[s]) >= want[s] for s in (0, 1)):
                break
            item = dataset[idx]
            src = _scalar(item.get("source_id"), 0)
            if src not in out or len(out[src]) >= want[src]:
                continue
            mv = item["main_valid"].bool()
            tm = item["target_mask"].bool() & mv
            if not bool(tm.any()) or int(mv.sum().item()) < 2:
                continue
            first_t = int(torch.nonzero(tm, as_tuple=False)[0].item())
            if first_t < 1:
                continue
            out[src].append((f"{SOURCE_NAMES.get(src, src)}#{idx}", item))
    finally:
        random.setstate(py_state)
        torch.set_rng_state(torch_state)
        if np is not None and np_state is not None:
            np.random.set_state(np_state)
    return out[0] + out[1]


@torch.no_grad()
def decode_x(
    x_gen: torch.Tensor, translator: Optional[ContinuousToTextTranslatorCUDA], tokenizer: CharTokenizer,
) -> Tuple[List[str], List[List[int]], torch.Tensor]:
    """Translator greedy decode of `x_gen (B, L, D)` -> (texts, token id lists, valid mask (B, L))."""
    B, L, _ = x_gen.shape
    if translator is None:
        texts = [f"<no-translator |x|={x_gen[i].float().norm().item():.1f}>" for i in range(B)]
        return texts, [[] for _ in range(B)], torch.ones((B, L), device=x_gen.device, dtype=torch.bool)
    pred = translator.decode_greedy(x_gen.float())  # (B, L)
    ids = [pred[i].tolist() for i in range(B)]
    valid = pred != tokenizer.pad_token_id
    return [decode_clause_clean(tokenizer, r) for r in ids], ids, valid


def ids_to_clause_tokens(ids: List[int], tokenizer: CharTokenizer, seq_len: int, device: torch.device) -> torch.Tensor:
    """Generated token ids -> (seq_len,) clause token row (truncated at EOS, PAD-filled)."""
    if tokenizer.eos_token_id in ids:
        ids = ids[: ids.index(tokenizer.eos_token_id)]
    ids = [i for i in ids if i != tokenizer.pad_token_id][:seq_len]
    row = torch.full((seq_len,), tokenizer.pad_token_id, device=device, dtype=torch.long)
    if ids:
        row[: len(ids)] = torch.tensor(ids, device=device, dtype=torch.long)
    return row


def _ctx_last(ctx: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    return {
        k: v[:, -1]
        for k, v in ctx.items()
        if k in ("h", "h0", "F_aux", "gates", "gate_max", "gate_sum", "z_lookback", "gate_lb") and v is not None
    }


def _short(s: str, n: int = 160) -> str:
    s = str(s).replace("\n", " ")
    return s if len(s) <= n else s[: n - 3] + "..."


# ============================================================================ evaluation
@torch.no_grad()
def evaluate_stage1(
    step: int,
    right_brain: MultiStreamRightBrain,
    selector: Optional[KnowledgeSelector],
    unet: LeftBrainCrystallizer,
    encoder: TinyCharEncoderCUDA,
    hippo: HiPPOOracleCUDA,
    translator: Optional[ContinuousToTextTranslatorCUDA],
    tokenizer: CharTokenizer,
    bench: List[Tuple[str, Dict[str, Any]]],
    args,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> None:
    right_brain.eval()
    pad = tokenizer.pad_token_id
    nb_sem = int(right_brain.sem_bands)
    fm = right_brain.flow_matcher
    # =====================================================================================
    # [Feature Addition & Bug Fix - Task #038: CausalContextLookback History & Auto Context Attractors]
    # 1. Problem fixed:
    #    - `evaluate_stage1` previously initialized `states = {"main": ctx["main_states"]}` without passing
    #      `history_h` and `history_z` from `forward_context` into `right_brain.step(...)`.
    #    - Without a trained `selector_ckpt`, `--eval_attractors` defaulted to `none`, leaving the Left Brain
    #      without context attractors during Stage-1 evaluation.
    # 2. Method used:
    #    - Pass `history_h` and `history_z` in `states` so `right_brain.step` attends over the full context prefix.
    #    - Support `--eval_attractors auto`: uses `selector.select_context_attractors` if `selector` is loaded,
    #      otherwise falls back to Whitened Clause Cosine Top-M Attractor Selection (`0.6 * sim_sem + 0.4 * sim_full`).
    # 3. Caveats for subsequent development:
    #    - Keep `return_state=True` in `forward_context` before sequential `step` loops so `history_h`/`history_z` are populated.
    # =====================================================================================
    use_sel_attr = args.eval_attractors in ("auto", "context") and selector is not None
    use_wht_attr = args.eval_attractors in ("auto", "context") and selector is None
    if use_sel_attr:
        attr_desc = "context(selector)"
    elif use_wht_attr:
        attr_desc = "auto(wht_cos)"
    else:
        attr_desc = "none"
    print("\n" + "=" * 96)
    print(f"🧠 [Step {step}] Stage-1 single-stream rollout | attractors={attr_desc} | "
          f"RightODE={args.right_ode_steps} LeftODE={args.left_ode_steps} T={args.temperature} "
          f"MaxClauses={args.num_rollout_clauses} StopThr={args.stop_threshold} "
          f"ClosedLoop={args.closed_loop_reencode}")
    print("=" * 96)

    def autocast():
        return torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled)

    def prepare(item: Dict[str, Any]):
        w = item["w_main"].unsqueeze(0).to(device)
        mv = item["main_valid"].unsqueeze(0).to(device).bool()
        tm = item["target_mask"].unsqueeze(0).to(device).bool() & mv
        z = encode_clause_bank_to_2d(encoder, unet, hippo, w, mv, pad).float()
        n_valid = int(mv[0].sum().item())
        first_t = int(torch.nonzero(tm[0], as_tuple=False)[0].item()) if bool(tm.any()) else n_valid
        return w, mv, tm, z, n_valid, first_t

    def crystallize(z_th: torch.Tensor, pool_z: List[torch.Tensor], pool_w: List[torch.Tensor]):
        x_attr, k_mask, use_attr, info = None, None, None, None
        if use_sel_attr and pool_z:
            zc = torch.stack(pool_z, dim=0).unsqueeze(0)
            wc = torch.stack(pool_w, dim=0).unsqueeze(0)
            w_attr, k_mask, info = selector.select_context_attractors(
                z_th.float(), zc, wc, torch.ones(zc.shape[:2], device=device, dtype=torch.bool),
                top_m=args.top_m, min_score=args.min_selector_score, sink_threshold=args.sink_threshold,
                pad_token_id=pad,
            )
            use_attr = info["use_attr"]
            if w_attr is not None:
                x_attr = unet.normalize(encode_tokens(encoder, w_attr))
        elif use_wht_attr and pool_z:
            zc = torch.stack(pool_z, dim=0)  # (N_pool, D)
            wc = torch.stack(pool_w, dim=0)  # (N_pool, L)
            z_th_n = fm.normalize(z_th.float())  # (1, D)
            zc_n = fm.normalize(zc.float())      # (N_pool, D)
            sem_dim = int(right_brain.sem_bands) * int(fm.d_sub)
            sim_sem = F.cosine_similarity(z_th_n[:, :sem_dim], zc_n[:, :sem_dim], dim=-1)
            sim_full = F.cosine_similarity(z_th_n, zc_n, dim=-1)
            scores = 0.6 * sim_sem + 0.4 * sim_full  # (N_pool,)
            thr = max(float(args.min_selector_score), 0.25)
            k = min(int(args.top_m), int((scores >= thr).sum().item()))
            max_cos_val = float(scores.max().item()) if scores.numel() > 0 else 0.0
            if k > 0:
                top_idx = scores.topk(k).indices
                top_idx, _ = torch.sort(top_idx)
                w_attr = wc[top_idx].reshape(1, -1)  # (1, k * L)
                k_mask = (w_attr != pad)
                use_attr = torch.ones((1,), device=device, dtype=torch.bool)
                x_attr = unet.normalize(encode_tokens(encoder, w_attr))
                info = {"use_attr": use_attr, "max_cos": max_cos_val}
            else:
                use_attr = torch.zeros((1,), device=device, dtype=torch.bool)
                info = {"use_attr": use_attr, "max_cos": max_cos_val}
        with autocast():
            x_gen = crystallize_with_optional_attractors(
                unet, z_th, x_attr, k_mask, use_attr, steps=args.left_ode_steps, seq_len=args.seq_len,
            )
        return x_gen.float(), info

    if int(fm.calib_steps.item()) == 0 and bench:
        calib_rows = []
        for _, b_item in bench:
            _, b_mv, _, b_z, b_n_valid, _ = prepare(b_item)
            if b_n_valid > 0:
                calib_rows.append(b_z[0, :b_n_valid])
        if calib_rows:
            fm.calibrate_z(torch.cat(calib_rows, dim=0))

    first_of: Dict[int, Tuple[str, Dict[str, Any]]] = {}
    for tag, item in bench:
        src = _scalar(item.get("source_id"), 0)
        first_of.setdefault(src, (tag, item))
        w, mv, tm, z, n_valid, first_t = prepare(item)
        ctx_len = max(1, first_t)
        gt_rows = list(range(first_t, n_valid))
        gt_strs = [decode_clause_clean(tokenizer, w[0, r].tolist()) for r in gt_rows[: args.num_rollout_clauses]]
        ctx_strs = [decode_clause_clean(tokenizer, w[0, r].tolist()) for r in range(ctx_len)]
        print(f"📌 [{tag}] domain={item.get('domain', '?')} | ctx_rows={ctx_len} target_rows={len(gt_rows)}")
        ctx_show = ctx_strs if len(ctx_strs) <= 6 else (ctx_strs[:2] + ["…"] + ctx_strs[-3:])
        print(f"   [Context]: {_short(' | '.join(ctx_show), 220)}")
        print(f"   [GT]:      {_short(' | '.join(gt_strs), 220)}")

        with autocast():
            ctx = right_brain.forward_context(z[:, :ctx_len], return_state=True)
        states = {"main": ctx["main_states"], "history_h": ctx.get("history_h"), "history_z": ctx.get("history_z")}
        last = _ctx_last(ctx)
        pool_z = [z[0, r] for r in range(ctx_len)]
        pool_w = [w[0, r] for r in range(ctx_len)]
        gen_strs: List[str] = []
        lines: List[str] = []
        stopped = None
        for c in range(1, args.num_rollout_clauses + 1):
            with autocast():
                mu, sig = right_brain.predict_prior(last)
                stop_p = float(torch.sigmoid(right_brain.stop_head(last["h"]).float()).reshape(-1)[0].item())
                z_th = fm.sample(last["h"], mu=mu, sigma_band=sig, steps=args.right_ode_steps,
                                 temperature=args.temperature).float()
            x_gen, info = crystallize(z_th, pool_z, pool_w)
            texts, ids, gvalid = decode_x(x_gen, translator, tokenizer)
            s_sem, s_syn = _sigma_sem_syn(sig[0], nb_sem)
            gt_r = first_t + c - 1
            cos_gt = (F.cosine_similarity(z_th, z[:, gt_r], dim=-1).item() if gt_r < n_valid else float("nan"))
            wht_cos = (
                F.cosine_similarity(fm.normalize(z_th.float()), fm.normalize(z[:, gt_r].float()), dim=-1).item()
                if gt_r < n_valid else float("nan")
            )
            attr_s = ""
            if info is not None:
                if "max_cos" in info:
                    attr_s = f" attr={'Y' if bool(info['use_attr'][0]) else 'N'}(cos={float(info['max_cos']):.2f})"
                elif "sink_prob" in info:
                    attr_s = (f" attr={'Y' if bool(info['use_attr'][0]) else 'N'}"
                              f"(sink={float(info['sink_prob'][0]):.2f})")
            lines.append(
                f"      c{c}: stop={stop_p:.3f} σ_sem={s_sem:.3f} σ_syn={s_syn:.3f} "
                f"cos(GT)={cos_gt:.3f} wht_cos={wht_cos:.3f}{attr_s} | {texts[0]}"
            )
            gen_strs.append(texts[0])
            if args.closed_loop_reencode:
                z_next = hippo(unet.normalize(x_gen), return_sequence=False, valid_mask=gvalid,
                               subspace_proj=unet.subspace_proj).float()
            else:
                z_next = z_th
            if stop_p > float(args.stop_threshold):
                stopped = c
                break
            with autocast():
                last, states = right_brain.step(z_next, states)
        stop_desc = f"stop@c{stopped}" if stopped is not None else f"max {args.num_rollout_clauses}"
        print(f"   [Rollout] ({stop_desc}): {_short(' | '.join(gen_strs), 220)}")
        for ln in lines:
            print(ln)
        print("-" * 96)

    # ---- diversity demo: next clause sampled N times for the first QA and first continuation sample
    n_div = int(args.eval_num_samples or 0)
    if n_div > 0:
        for src in (0, 1):
            if src not in first_of:
                continue
            tag, item = first_of[src]
            w, mv, tm, z, n_valid, first_t = prepare(item)
            ctx_len = max(1, first_t)
            gt = decode_clause_clean(tokenizer, w[0, first_t].tolist()) if first_t < n_valid else ""
            with autocast():
                ctx = right_brain.forward_context(z[:, :ctx_len])
                last = _ctx_last(ctx)
                mu, sig = right_brain.predict_prior(last)
            s_sem, s_syn = _sigma_sem_syn(sig[0], nb_sem)
            print(f"🎲 Diversity [{tag}] next clause x{n_div} (T={args.temperature}, σ_sem={s_sem:.3f}, σ_syn={s_syn:.3f}) | GT: {gt}")
            samples = []
            pool_z = [z[0, r] for r in range(ctx_len)]
            pool_w = [w[0, r] for r in range(ctx_len)]
            for i in range(n_div):
                with autocast():
                    z_s = fm.sample(last["h"], mu=mu, sigma_band=sig, steps=args.right_ode_steps,
                                    temperature=args.temperature).float()
                samples.append(z_s[0])
                x_gen, _ = crystallize(z_s, pool_z, pool_w)
                texts, _, _ = decode_x(x_gen, translator, tokenizer)
                cos_gt = F.cosine_similarity(z_s, z[:, first_t], dim=-1).item() if first_t < n_valid else float("nan")
                wht_cos = (
                    F.cosine_similarity(fm.normalize(z_s.float()), fm.normalize(z[:, first_t].float()), dim=-1).item()
                    if first_t < n_valid else float("nan")
                )
                print(f"   #{i + 1} cos(GT)={cos_gt:.3f} wht_cos={wht_cos:.3f} | {texts[0]}")
            if len(samples) >= 2:
                zs = F.normalize(torch.stack(samples), dim=-1)
                sim = zs @ zs.t()
                off = sim[~torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)]
                print(f"   mean pairwise cos between samples = {off.mean().item():.4f}")

    # ---- sigma vs context length (implicit-condition evidence) on up to 2 continuation samples
    ks = [int(k) for k in str(args.context_lengths).split(",") if k.strip()]
    cont_items = [(t, it) for t, it in bench if _scalar(it.get("source_id"), 0) == 1][:2]
    if ks and cont_items:
        print(f"📉 σ vs context length (k = rows given; predicting row k) — expect σ_sem to DECREASE with k")
        for tag, item in cont_items:
            w, mv, tm, z, n_valid, first_t = prepare(item)
            rows = []
            for k in ks:
                if k < 1 or k >= n_valid:
                    rows.append(f"k={k}: n/a")
                    continue
                with autocast():
                    ctx = right_brain.forward_context(z[:, :k])
                    last = _ctx_last(ctx)
                    mu, sig = right_brain.predict_prior(last)
                s_sem, s_syn = _sigma_sem_syn(sig[0], nb_sem)
                z1n = fm.normalize(z[:, k].float())
                nll = float(band_gaussian_nll(z1n, mu.float(), sig.float(), right_brain.num_bands).item())
                mu_unit = mu.float() / mu.float().std(dim=-1, keepdim=True).clamp(min=1e-5)
                cos_mu = F.cosine_similarity(fm.denormalize(mu_unit), z[:, k].float(), dim=-1).item()
                wht_cos_mu = F.cosine_similarity(mu.float(), z1n, dim=-1).item()
                rows.append(
                    f"k={k}: σ_sem={s_sem:.3f} σ_syn={s_syn:.3f} nll={nll:.3f} "
                    f"cos(μ,GT)={cos_mu:.3f} wht_cos(μ,GT)={wht_cos_mu:.3f}"
                )
            print(f"   [{tag}] " + " | ".join(rows))
    print("=" * 96 + "\n")


# ============================================================================ CLI
def add_frozen_ckpt_args(parser) -> None:
    parser.add_argument("--encoder_ckpt", type=str, default="checkpoints/basencoder_step_115000/student.pt",
                        help="Frozen Phase 1 TinyCharEncoderCUDA checkpoint")
    parser.add_argument("--left_brain_ckpt", "--unet_ckpt", dest="left_brain_ckpt", type=str, default=None,
                        help="Frozen LeftBrainCrystallizer checkpoint (file / dir / glob; auto-discovered if omitted)")
    parser.add_argument("--left_brain_ckpt_prefix", type=str, default="left_brain_s32_fullrank",
                        help="Preferred prefix for LeftBrainCrystallizer auto-discovery")
    parser.add_argument("--translator_ckpt", type=str, default=None,
                        help="Frozen ContinuousToTextTranslatorCUDA checkpoint (auto-discovered if omitted)")
    parser.add_argument("--num_subbands", type=int, default=32, help="Legendre modes K (clause-vector bands)")


def add_data_args(parser) -> None:
    default_qa = "/content/omni_knowledge_corpus.jsonl" if os.path.exists("/content/omni_knowledge_corpus.jsonl") \
        else "data/Knowledge_ZH/bicameral_knowledge_qa.jsonl"
    parser.add_argument("--qa_paths", type=str, default=default_qa, help="Comma-separated QA corpora (JSONL)")
    parser.add_argument("--cont_paths", type=str, default="", help="Comma-separated long-text corpora ({'text': ...} JSONL)")
    parser.add_argument("--cont_ratio", type=float, default=0.3, help="Fraction of continuation samples in the mix")
    parser.add_argument("--max_clauses", type=int, default=32, help="Clauses per main-stream window (T)")
    parser.add_argument("--max_qa_samples", type=int, default=None, help="Cap on loaded QA samples")
    parser.add_argument("--max_cont_docs", type=int, default=None, help="Cap on loaded continuation documents")
    parser.add_argument("--num_workers", type=int, default=None, help="DataLoader workers (passed only if set)")
    # #032 dataset pass-through (defaults per #032 delivery: spec 0.5 / 0.6 gave sink=1 on ~91% of target rows)
    parser.add_argument("--rel_threshold", type=float, default=0.3,
                        help="Rare-span relevance threshold: sink_label=1 if no earlier row reaches it")
    parser.add_argument("--rel_min_idf_quantile", type=float, default=0.3,
                        help="df-weighted IDF quantile above which a char bigram counts as rare")
    parser.add_argument("--cont_min_context", type=int, default=None, help="Min context rows of continuation windows")
    parser.add_argument("--cont_max_context", type=int, default=None, help="Max context rows of continuation windows")
    parser.add_argument("--epoch_size", type=int, default=None, help="Override dataset length (samples per epoch)")


def build_dataloader(args, tokenizer: CharTokenizer, shuffle: bool = True):
    kwargs = dict(
        qa_paths=split_paths(args.qa_paths),
        cont_paths=split_paths(args.cont_paths),
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        shuffle=shuffle,
        seq_len=args.seq_len,
        max_clauses=args.max_clauses,
        cont_ratio=args.cont_ratio,
        max_qa_samples=args.max_qa_samples,
        max_cont_docs=args.max_cont_docs,
        seed=args.seed if args.seed is not None else 42,
        verbose=True,
        rel_threshold=args.rel_threshold,
        rel_min_idf_quantile=args.rel_min_idf_quantile,
    )
    for key in ("num_workers", "cont_min_context", "cont_max_context", "epoch_size"):
        val = getattr(args, key, None)
        if val is not None:
            kwargs[key] = int(val)
    return create_clause_stream_dataloader(**kwargs)


def build_parser():
    parser = get_training_parser("Phase 5 Stage-1 single-stream Right-Brain training (#033, CUDA)")
    add_data_args(parser)
    add_frozen_ckpt_args(parser)
    parser.add_argument("--max_steps", type=int, default=50000, help="Training horizon (steps)")
    parser.add_argument("--min_lr_ratio", type=float, default=0.1, help="Cosine LR floor ratio")
    parser.add_argument("--accum_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["bf16", "fp16", "no"])
    parser.add_argument("--auto_resume", dest="auto_resume", action="store_true", help="Alias of --continue")
    # right-brain hyper-parameters (#029 defaults)
    parser.add_argument("--d_mamba", type=int, default=1024, help="Internal Mamba width")
    parser.add_argument("--n_layers_main", type=int, default=4, help="Main-stream Mamba layers")
    parser.add_argument("--n_layers_aux", type=int, default=2, help="Companion-stream layers (unused in Stage 1, kept for Stage 3 compat)")
    parser.add_argument("--num_branches", type=int, default=4, help="K blind flow branches per target row")
    parser.add_argument("--train_ode_steps", type=int, default=3, help="Euler steps per blind branch / self-feed sample")
    parser.add_argument("--win_weight", type=float, default=0.65, help="Winner-branch loss weight")
    parser.add_argument("--sem_bands", type=int, default=6, help="Low bands treated as semantic (hypothesis)")
    parser.add_argument("--sigma_min", type=float, default=0.05, help="Prior per-band sigma floor (normalized space)")
    parser.add_argument("--sigma_max", type=float, default=3.0, help="Prior per-band sigma ceiling (normalized space)")
    parser.add_argument("--prior_weight", type=float, default=1.0, help="Band NLL prior weight")
    parser.add_argument("--stop_weight", type=float, default=0.2, help="Stop-head BCE weight")
    parser.add_argument("--p_band_jitter", type=float, default=0.3, help="P(high-band orthogonal jitter) per input row")
    parser.add_argument("--p_self_max", type=float, default=0.25, help="Max scheduled self-feeding probability")
    parser.add_argument("--self_feed_start", type=int, default=2000, help="Step where the self-feed ramp starts")
    parser.add_argument("--self_feed_ramp", type=int, default=8000, help="Steps to ramp p_self to --p_self_max")
    # eval
    parser.add_argument("--eval_interval", type=int, default=500, help="Eval every N steps (0 = off)")
    parser.add_argument("--eval_at_start", action="store_true", help="Run one eval before training")
    parser.add_argument("--log_interval", type=int, default=20, help="Log every N steps")
    parser.add_argument("--num_rollout_clauses", type=int, default=8, help="Max clauses per eval rollout")
    parser.add_argument("--right_ode_steps", type=int, default=3, help="Right-brain ODE steps at eval")
    parser.add_argument("--left_ode_steps", "--euler_steps", dest="left_ode_steps", type=int, default=10,
                        help="Left-brain ODE steps at eval")
    parser.add_argument("--temperature", type=float, default=0.75, help="Prior temperature at eval")
    parser.add_argument("--stop_threshold", type=float, default=0.5, help="Rollout stops when stop_prob exceeds this")
    parser.add_argument("--eval_num_samples", type=int, default=3, help="Diversity-demo samples (0 = off)")
    parser.add_argument("--context_lengths", type=str, default="2,4,8,16", help="k values for the σ-vs-context curve")
    parser.add_argument("--bench_qa", type=int, default=3, help="QA samples in the fixed benchmark")
    parser.add_argument("--bench_cont", type=int, default=3, help="Continuation samples in the fixed benchmark")
    parser.add_argument("--eval_attractors", type=str, default="auto", choices=["auto", "context", "none"],
                        help="Left-brain attractors at eval: auto (selector if loaded, else whitened-cosine context top-M) | context | none")
    parser.add_argument("--selector_ckpt", type=str, default=None, help="Trained #033 KnowledgeSelector (file / dir)")
    parser.add_argument("--top_m", type=int, default=3, help="Max attractor clauses (context mode)")
    parser.add_argument("--min_selector_score", type=float, default=0.25, help="Min sigmoid score of a selected clause")
    parser.add_argument("--sink_threshold", type=float, default=0.5, help="sink_prob above this -> no attractors")
    parser.add_argument("--closed_loop_reencode", dest="closed_loop_reencode", action="store_true", default=True,
                        help="Re-encode the crystallized clause into the right brain during rollout (default)")
    parser.add_argument("--no_closed_loop_reencode", dest="closed_loop_reencode", action="store_false",
                        help="Feed z_thought directly during rollout")
    parser.add_argument("--profile", action="store_true", help="CUDA-synchronised per-module timing at every log step")
    parser.set_defaults(seq_len=32, batch_size=16, warmup_steps=500, save_steps=1000, ckpt_prefix="right_brain_stage1")
    return parser


# ============================================================================ training
def train():
    args = build_parser().parse_args()
    if args.seq_len > 64:
        print(f"ℹ️ Overriding --seq_len {args.seq_len} -> 32 (punctuation-clause left brain).")
        args.seq_len = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    amp_enabled, amp_dtype, scaler = setup_amp(args.mixed_precision, device)
    tokenizer = CharTokenizer()
    pad_id = tokenizer.pad_token_id

    dataset, dataloader = build_dataloader(args, tokenizer, shuffle=True)
    bench = select_benchmark_items(dataset, n_qa=args.bench_qa, n_cont=args.bench_cont, seed=42)
    print(f"📌 Fixed benchmark: {', '.join(t for t, _ in bench) if bench else '(none found)'}")

    encoder, hippo, unet, translator = load_frozen_modules(args, tokenizer, device, need_translator=True)
    selector = load_selector(args.selector_ckpt, device) if args.eval_attractors in ("auto", "context") else None

    right_brain = build_right_brain_from_args(vars(args)).to(device)
    # Stage 1 never feeds companions: only parameters on the main path are optimised (aux_stream / gates /
    # aux_v would get no gradient anyway; excluding them keeps AdamW weight decay from shrinking them).
    aux_prefixes = ("aux_stream.", "gate_q.", "gate_k.", "gate_bias", "aux_v.", "alpha_aux")
    trainable = [p for n, p in right_brain.named_parameters() if not n.startswith(aux_prefixes)]
    optimizer = optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    horizon = args.max_steps if args.max_steps > 0 else max(1, args.epochs * len(dataloader))
    scheduler = get_warmup_cosine_scheduler(optimizer, args.warmup_steps, horizon, args.min_lr_ratio)

    checkpointer = Checkpointer(out_dir=args.out_dir, prefix=args.ckpt_prefix, keep_last_k=args.keep_last_k)
    checkpointer.register_model("right_brain", right_brain)
    checkpointer.register_optimizer("engine_opt", optimizer)
    checkpointer.register_optimizer("lr_sched", scheduler)
    checkpointer.register_args(args)
    global_step = resume_training(args, checkpointer, {"right_brain": right_brain}, scheduler, device)

    n_params = sum(p.numel() for p in trainable)
    print(f"🚀 Stage-1 single-stream Right-Brain (#033) | trainable {n_params / 1e6:.1f}M | d_mamba={args.d_mamba} "
          f"main_layers={args.n_layers_main} | T={args.max_clauses} cont_ratio={args.cont_ratio}")
    print(f"   Flow: K={args.num_branches} x {args.train_ode_steps} ODE, win={args.win_weight} | Prior: sem_bands="
          f"{args.sem_bands} σ∈[{args.sigma_min},{args.sigma_max}] w={args.prior_weight} | Stop w={args.stop_weight} | "
          f"Jitter p={args.p_band_jitter} | SelfFeed p_max={args.p_self_max} start={args.self_feed_start} "
          f"ramp={args.self_feed_ramp} | ckpt_prefix={args.ckpt_prefix}")

    prof = Profiler(args.profile, device)

    def run_eval(step: int) -> float:
        t0 = time.time()
        evaluate_stage1(step, right_brain, selector, unet, encoder, hippo, translator, tokenizer, bench, args,
                        device, amp_enabled, amp_dtype)
        right_brain.train()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        dt = time.time() - t0
        if args.profile:
            print(f"    ⏱ eval wall time {dt:.2f}s")
        return dt

    if args.eval_at_start and bench:
        run_eval(global_step)

    start_epoch = global_step // len(dataloader) if len(dataloader) > 0 else 0
    optimizer.zero_grad(set_to_none=True)
    last_log_time, last_log_step, excluded = time.time(), global_step, 0.0
    done = False
    for epoch in range(start_epoch, args.epochs):
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)  # #032: rotates index -> QA record / continuation slot mapping
        loader = checkpointer.wrap_dataloader(dataloader, epoch, global_step)
        right_brain.train()
        for batch_idx, batch in enumerate(loader):
            prof.start()
            w_main = batch["w_main"].to(device)                       # (B, T, L)
            main_valid = batch["main_valid"].to(device).bool()        # (B, T)
            target_mask = batch["target_mask"].to(device).bool()      # (B, T)
            stop_labels = batch["stop_labels"].to(device).float()     # (B, T)
            stop_valid = batch["stop_valid"].to(device).bool()        # (B, T)
            source_id = batch["source_id"].to(device).long()          # (B,) logging only
            prof.mark("data")

            with torch.no_grad():
                z_main = encode_clause_bank_to_2d(encoder, unet, hippo, w_main, main_valid, pad_id)  # (B, T, D)
                prof.mark("encode")
                target_valid = target_mask[:, 1:] & main_valid[:, 1:]  # (B, T-1)
                if int(right_brain.flow_matcher.calib_steps.item()) == 0 and bool(main_valid.any()):
                    right_brain.flow_matcher.calibrate_z(z_main[main_valid])
                z_in = right_brain.jitter_high_bands(
                    z_main[:, :-1].clone(), p=args.p_band_jitter,
                    sem_bands=right_brain.sem_bands, num_bands=right_brain.num_bands,
                )
                p_self = compute_p_self(global_step, args.self_feed_start, args.self_feed_ramp, args.p_self_max)
                n_fed = 0
                if p_self > 0.0 and int(right_brain.flow_matcher.calib_steps.item()) > 0 and z_in.shape[1] >= 2:
                    with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                        samples = right_brain.sample_next_clauses(
                            z_main[:, :-1], temperature=1.0, ode_steps=args.train_ode_steps,
                        )  # samples[:, j] = predicted row j+1
                    # z_in[:, t] (t >= 1) is row t; only TARGET rows are replaced (context stays ground truth).
                    tgt_rows = (target_mask[:, 1:-1] & main_valid[:, 1:-1])
                    replace = (torch.rand(tgt_rows.shape, device=device) < p_self) & tgt_rows
                    z_in[:, 1:] = torch.where(replace.unsqueeze(-1), samples[:, :-1].to(dtype=z_in.dtype), z_in[:, 1:])
                    n_fed = int(replace.sum().item())
                prof.mark("self_feed")

            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                loss, m = right_brain.compute_loss(
                    z_main,
                    target_valid,
                    aux=None,
                    stop_labels=stop_labels[:, 1:],
                    stop_valid_mask=stop_valid[:, 1:],
                    z_in_override=z_in,
                    sample_group=source_id,
                    group_names=SOURCE_NAMES,
                    return_metrics=True,
                    profile_cb=prof.mark if args.profile else None,
                )
                loss_scaled = loss / max(1, args.accum_steps)
            if scaler is not None:
                scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()
            if (batch_idx + 1) % args.accum_steps == 0 or (batch_idx + 1) == len(loader):
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            prof.mark("backward")
            prof.end_step()

            global_step += 1
            checkpointer.set_step(global_step)
            is_eval = args.eval_interval > 0 and global_step % args.eval_interval == 0
            if global_step == 1 or global_step % max(1, args.log_interval) == 0 or is_eval:
                now = time.time()
                sps = (now - last_log_time - excluded) / max(1, global_step - last_log_step)
                last_log_time, last_log_step, excluded = now, global_step, 0.0
                n_qa = int((source_id == 0).sum().item())
                n_ct = int((source_id == 1).sum().item())
                print(
                    f"Epoch {epoch + 1} | Step {global_step} | LR {optimizer.param_groups[0]['lr']:.2e} | {sps:.2f}s/step | "
                    f"Total {m['total_loss']:.4f} | Flow {m['flow_loss']:.4f} (win {m['win_loss']:.4f}, slot {m.get('slot_loss', 0.0):.4f}, "
                    f"other {m['other_loss']:.4f}) | PriorNLL {m['prior_nll']:.4f} | Stop {m['stop_loss']:.4f} "
                    f"(acc {_fmt_opt(m.get('stop_acc'))}, rec {_fmt_opt(m.get('stop_recall'))}, rows {m.get('n_stop_rows', 0)}) | "
                    f"ThoughtCos {m['thought_cos']:.4f}"
                )
                print(
                    f"    σ_sem {m['sigma_sem']:.3f} σ_syn {m['sigma_syn']:.3f} | σ_sem[qa/cont] "
                    f"{_fmt_opt(m.get('sigma_sem_qa'))} / {_fmt_opt(m.get('sigma_sem_cont'))} | σ_syn[qa/cont] "
                    f"{_fmt_opt(m.get('sigma_syn_qa'))} / {_fmt_opt(m.get('sigma_syn_cont'))} | samples qa/cont {n_qa}/{n_ct} "
                    f"| target rows qa/cont {m.get('n_rows_qa', 0)}/{m.get('n_rows_cont', 0)}"
                )
                z_std_t = right_brain.flow_matcher.z_std
                print(
                    f"    BranchDiv {m['branch_diversity']:.4f} | Winners {m['winner_hist']} | p_self {p_self:.3f} "
                    f"(fed {n_fed}) | z_std[min/mean/max] {z_std_t.min().item():.3f}/{z_std_t.mean().item():.3f}/{z_std_t.max().item():.3f}"
                )
                check_low_band_saturation(m.get("sigma_band", []), right_brain.sem_bands, args.sigma_max, tag="train")
                if is_eval:
                    print("    σ_band: " + " ".join(f"{s:.2f}" for s in m.get("sigma_band", [])))
                rep = prof.report_and_reset()
                if rep:
                    print(rep)
            if is_eval and bench:
                excluded += run_eval(global_step)
            if args.save_steps and global_step % args.save_steps == 0:
                t0 = time.time()
                checkpointer.save(global_step)
                excluded += time.time() - t0
            if args.max_steps > 0 and global_step >= args.max_steps:
                done = True
                break
        if done:
            break

    if not (args.save_steps and global_step > 0 and global_step % args.save_steps == 0):
        checkpointer.save(global_step)
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    train()
