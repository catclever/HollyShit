import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import sys
import math
import glob
import json
import re
import random
import time
from collections import deque
from typing import List, Tuple, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir in sys.path and os.path.exists(os.path.join(script_dir, "model.py")):
    sys.path = [p for p in sys.path if p != script_dir] + [script_dir]

from training.core.char_tokenizer import CharTokenizer

from model.hippo_oracle import HiPPOOracleCUDA
from model.flow_unet_cuda import (
    FlowMatching1DUnet,
    LeftBrainCrystallizer,
)
from model.loss_hippo_cfm import hippo_cfm_loss
from training.data.dialogue_dataset import GenericPairDataset

from model.phase1_charencoder_cuda import TinyCharEncoderCUDA, load_mlx_safetensors_into_torch
from model.continuous_translator_cuda import ContinuousToTextTranslatorCUDA
from training.core.args import get_training_parser
from training.core.checkpoint import Checkpointer
from torch.utils.data import DataLoader, Subset

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


def find_compatible_unet_ckpt(unet, out_dir, prefix="phase4_sentvec_flow", device="cpu"):
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
    """Encodes token IDs `(B, L)` via `encoder.encode_tokens` with mandatory `<PAD>` mask & zeroing."""
    return encoder.encode_tokens(token_ids)


@torch.no_grad()
def sample_euler_ode(unet, x_prompt, h_potential, steps=10, seq_len=32, prompt_mask=None, device="cpu", x_init=None):
    """
    Standard Stable Euler ODE Integrator conditioned on 2D Clause Vector `h_potential: (B, 1024)` + `k_attractors`:
    dX_t = v(X_t, t, h_potential=z_sent, k_attractors=x_prompt_norm, k_mask=prompt_mask) * dt
    Integrates from white noise X_0 ~ N(0, I) -> X_1_norm, then auto-denormalizes back to encoder manifold.
    `x_init` (optional, #053): fixed initial noise X_0 of shape (B, seq_len, D) for step-to-step comparable evals.
    """
    B = x_prompt.shape[0]
    D = x_prompt.shape[-1]
    x_prompt_norm = unet.normalize(x_prompt)
    if x_init is not None:
        x = x_init.to(device=device, dtype=x_prompt.dtype)
    else:
        x = torch.randn((B, seq_len, D), device=device, dtype=x_prompt.dtype)
    dt = 1.0 / steps
    for i in range(steps):
        t_val = i / float(steps)
        t_tensor = torch.full((B,), t_val, device=device, dtype=torch.float32)
        v_pred = unet(x, t_tensor, h_potential=h_potential, k_attractors=x_prompt_norm, k_mask=prompt_mask)
        x = x + v_pred * dt
    return unet.denormalize(x)


# ============================================================================
# #053 Fixed Held-Out Quantitative Per-Mode Health Evaluation
# ============================================================================
QUANT_EVAL_MODES = [
    # (key, label, kind, param)
    ("exact", "Exact    ", "exact", None),
    ("cav_055", "Cav α=.55", "cavity", 0.55),
    ("cav_040", "Cav α=.40", "cavity", 0.40),
    ("low_k7", "Low k≤7  ", "lowfreq", 7),
    ("low_k5", "Low k≤5  ", "lowfreq", 5),
]


def select_quant_eval_indices(n_total: int, size: int, seed: int) -> np.ndarray:
    """Deterministically picks `size` distinct dataset indices (sorted) with a seeded numpy Generator. O(size) memory."""
    size = int(max(0, min(int(size), int(n_total))))
    if size == 0:
        return np.zeros((0,), dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(int(n_total), size=size, replace=False)
    return np.sort(idx.astype(np.int64))


def build_train_subset(dataset, held_out_idx: np.ndarray):
    """Returns a `Subset` over all indices except `held_out_idx` (boolean mask, O(N) & memory-light), or the dataset itself."""
    if held_out_idx is None or len(held_out_idx) == 0:
        return dataset
    n_total = len(dataset)
    keep = np.ones((n_total,), dtype=bool)
    keep[held_out_idx] = False
    idx_dtype = np.int32 if n_total < 2**31 - 1 else np.int64
    train_idx = np.flatnonzero(keep).astype(idx_dtype)
    del keep
    return Subset(dataset, train_idx)


def build_token_class_lut(tokenizer, vocab_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-token-id boolean lookup tables: (is_cjk U+4E00–U+9FFF, is_printable_ascii 0x20–0x7E), via tokenizer.decode."""
    is_zh = torch.zeros((vocab_size,), dtype=torch.bool)
    is_ascii = torch.zeros((vocab_size,), dtype=torch.bool)
    special = {
        getattr(tokenizer, "pad_token_id", None),
        getattr(tokenizer, "unk_token_id", None),
        getattr(tokenizer, "bos_token_id", None),
        getattr(tokenizer, "eos_token_id", None),
    }
    for tid in range(vocab_size):
        if tid in special:
            continue
        try:
            s = tokenizer.decode([tid], skip_special_tokens=True)
        except Exception:
            continue
        if len(s) != 1:
            continue
        cp = ord(s)
        if 0x4E00 <= cp <= 0x9FFF:
            is_zh[tid] = True
        elif 0x20 <= cp <= 0x7E:
            is_ascii[tid] = True
    return is_zh, is_ascii


def _first_eos_pos(ids: torch.Tensor, eos_id: int) -> torch.Tensor:
    """First <EOS> position per row (same rule as `decode_with_eos_marker`: list.index(eos)), or -1 if absent."""
    is_eos = ids == eos_id
    has = is_eos.any(dim=1)
    pos = is_eos.to(torch.int8).argmax(dim=1)
    return torch.where(has, pos, torch.full_like(pos, -1))


def _renorm_std(z: torch.Tensor) -> torch.Tensor:
    """Same re-normalization convention as `hippo_cfm_loss` cavity/spectral modes: z / std(z) (no mean shift)."""
    return z / z.float().std(dim=-1, keepdim=True).clamp(min=1e-5).to(z.dtype)


class QuantEvalBenchmark:
    """
    Fixed held-out benchmark: caches token ids (CPU), fixed distractor indices, fixed initial ODE noise x_0, and
    (lazily, after manifold calibration) the clean `z_sent` for every item. Everything is seeded => identical across evals.
    """

    def __init__(self, dataset, indices: np.ndarray, tokenizer, seq_len: int, hidden_dim: int = 1024,
                 seed: int = 1234, num_distractors: int = 8):
        self.indices = [int(i) for i in indices]
        self.n = len(self.indices)
        self.seq_len = int(seq_len)
        self.hidden_dim = int(hidden_dim)
        self.seed = int(seed)
        self.num_distractors = int(num_distractors)
        self.tokenizer = tokenizer

        prompts, targets = [], []
        for ds_idx in self.indices:
            item = dataset[ds_idx]
            w_p, w_t = (item[0], item[-1]) if len(item) != 2 else item
            prompts.append(w_p)
            targets.append(w_t)
        self.w_prompt = torch.stack(prompts, dim=0) if prompts else torch.zeros((0, 1), dtype=torch.long)
        self.w_target = torch.stack(targets, dim=0) if targets else torch.zeros((0, self.seq_len), dtype=torch.long)

        rng = np.random.default_rng(self.seed)
        dist = np.zeros((self.n, max(1, self.num_distractors)), dtype=np.int64)
        for i in range(self.n):
            others = np.concatenate([np.arange(0, i), np.arange(i + 1, self.n)])
            if len(others) == 0:
                dist[i, :] = i  # degenerate N=1 fallback
            else:
                dist[i, :] = rng.choice(others, size=dist.shape[1], replace=len(others) < dist.shape[1])
        self.dist_idx = torch.from_numpy(dist)

        gen = torch.Generator(device="cpu")
        gen.manual_seed(self.seed)
        self.x_init = torch.randn((self.n, self.seq_len, self.hidden_dim), generator=gen, dtype=torch.float32)

        self.z_clean = None  # (N, D) float32 CPU, filled by `prepare`
        self._cache_key = None

    @staticmethod
    def _unet_key(unet):
        return (
            int(unet.calib_steps.item()),
            float(unet.manifold_mean.item()),
            float(unet.manifold_std.item()),
            float(unet.subspace_proj.float().abs().sum().item()),
        )

    @torch.no_grad()
    def prepare(self, encoder, hippo_oracle, unet, device, amp_dtype=torch.float32, amp_enabled=False, chunk: int = 64):
        """(Re-)computes clean z_sent only if the manifold calibration / subspace_proj changed since the last cache."""
        key = self._unet_key(unet)
        if self.z_clean is not None and key == self._cache_key:
            return False
        pad_id = self.tokenizer.pad_token_id
        zs = []
        for s in range(0, self.n, chunk):
            w_t = self.w_target[s:s + chunk].to(device)
            vm = w_t != pad_id
            x_t = encode_tokens(encoder, w_t)
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                z = hippo_oracle(unet.normalize(x_t), return_sequence=False, valid_mask=vm, subspace_proj=unet.subspace_proj)
            zs.append(z.float().cpu())
        self.z_clean = torch.cat(zs, dim=0)
        self._cache_key = key
        return True

    def build_mode_z(self, kind: str, param, sl: slice, num_subbands: int) -> torch.Tensor:
        """Deterministic corrupted z for items in slice `sl` (float32, CPU)."""
        z = self.z_clean[sl]
        if kind == "exact":
            return z.clone()
        if kind == "cavity":
            alpha = float(param)
            z_dist_mean = self.z_clean[self.dist_idx[sl]].mean(dim=1)
            return _renorm_std(alpha * z + (1.0 - alpha) * z_dist_mean)
        if kind == "lowfreq":
            k_max = int(param)
            n, D = z.shape
            K = int(num_subbands)
            z_sub = z.view(n, K, D // K).clone()
            z_sub[:, k_max + 1:, :] = 0.0
            return _renorm_std(z_sub.reshape(n, D))
        raise ValueError(f"Unknown quant-eval mode kind: {kind}")


@torch.no_grad()
def run_quant_eval(bench: QuantEvalBenchmark, unet, encoder, hippo_oracle, translator, tokenizer, device,
                   amp_dtype=torch.float32, amp_enabled=False, euler_steps: int = 10, num_subbands: int = 32,
                   chunk: int = 64, token_luts=None) -> Dict[str, Any]:
    """Runs all QUANT_EVAL_MODES on the fixed benchmark. Caller is responsible for unet.eval()/train() toggling."""
    bench.prepare(encoder, hippo_oracle, unet, device, amp_dtype=amp_dtype, amp_enabled=amp_enabled, chunk=chunk)
    pad_id, eos_id = tokenizer.pad_token_id, tokenizer.eos_token_id
    if token_luts is None:
        token_luts = build_token_class_lut(tokenizer, tokenizer.vocab_size)
    is_zh_lut, is_ascii_lut = token_luts[0].to(device), token_luts[1].to(device)

    stat_keys = ["correct", "valid", "zh_c", "zh_n", "as_c", "as_n", "conf", "cos", "eos", "n"]
    acc = {m[0]: {k: 0.0 for k in stat_keys} for m in QUANT_EVAL_MODES}
    ref = {"correct": 0.0, "valid": 0.0, "eos": 0.0, "n": 0.0}

    for s in range(0, bench.n, chunk):
        sl = slice(s, min(s + chunk, bench.n))
        w_p = bench.w_prompt[sl].to(device)
        w_t = bench.w_target[sl].to(device)
        vm = w_t != pad_id
        pm = w_p != pad_id
        vm_f = vm.float()
        zh_m = is_zh_lut[w_t] & vm
        as_m = is_ascii_lut[w_t] & vm
        tgt_eos = _first_eos_pos(w_t, eos_id)
        x_p = encode_tokens(encoder, w_p)
        x_tgt = encode_tokens(encoder, w_t)
        x0 = bench.x_init[sl].to(device)
        nb = w_t.shape[0]

        # Translator reference ceiling (encoder -> translator), fp32
        ref_pred = translator(x_tgt.float()).argmax(dim=-1)
        ref["correct"] += float(((ref_pred == w_t) & vm).sum().item())
        ref["valid"] += float(vm.sum().item())
        ref["eos"] += float((_first_eos_pos(ref_pred, eos_id) == tgt_eos).sum().item())
        ref["n"] += nb

        for m_key, _m_label, m_kind, m_param in QUANT_EVAL_MODES:
            z_m = bench.build_mode_z(m_kind, m_param, sl, num_subbands).to(device=device, dtype=x_p.dtype)
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                x_gen = sample_euler_ode(
                    unet=unet, x_prompt=x_p, h_potential=z_m, steps=euler_steps, seq_len=bench.seq_len,
                    prompt_mask=pm, device=device, x_init=x0,
                )
            x_gen = x_gen.float()
            logits = translator(x_gen).float()
            probs_max, pred = torch.softmax(logits, dim=-1).max(dim=-1)
            correct = (pred == w_t) & vm
            cos = torch.nn.functional.cosine_similarity(x_gen, x_tgt.float(), dim=-1, eps=1e-6)
            a = acc[m_key]
            a["correct"] += float(correct.sum().item())
            a["valid"] += float(vm.sum().item())
            a["zh_c"] += float((correct & zh_m).sum().item())
            a["zh_n"] += float(zh_m.sum().item())
            a["as_c"] += float((correct & as_m).sum().item())
            a["as_n"] += float(as_m.sum().item())
            a["conf"] += float((probs_max * vm_f).sum().item())
            a["cos"] += float((cos * vm_f).sum().item())
            a["eos"] += float((_first_eos_pos(pred, eos_id) == tgt_eos).sum().item())
            a["n"] += nb

    def _div(x, y):
        return float(x / y) if y > 0 else float("nan")

    result = {
        "n": bench.n,
        "euler_steps": int(euler_steps),
        "ref_char_acc": _div(ref["correct"], ref["valid"]),
        "ref_eos_acc": _div(ref["eos"], ref["n"]),
        "modes": {},
    }
    for m_key, _l, _k, _p in QUANT_EVAL_MODES:
        a = acc[m_key]
        result["modes"][m_key] = {
            "char_acc": _div(a["correct"], a["valid"]),
            "char_acc_zh": _div(a["zh_c"], a["zh_n"]),
            "char_acc_ascii": _div(a["as_c"], a["as_n"]),
            "eos_acc": _div(a["eos"], a["n"]),
            "char_conf": _div(a["conf"], a["valid"]),
            "manifold_cos": _div(a["cos"], a["valid"]),
            "n_zh_tok": int(a["zh_n"]),
            "n_ascii_tok": int(a["as_n"]),
        }
    return result


def _fmt_pct(v) -> str:
    return "-" if (v is None or (isinstance(v, float) and math.isnan(v))) else f"{100.0 * v:.1f}%"


def _fmt_num(v, nd: int = 3) -> str:
    try:
        v = float(v)
    except Exception:
        return "-"
    return "-" if math.isnan(v) else f"{v:.{nd}f}"


def format_quant_eval(step: int, result: Dict[str, Any]) -> str:
    lines = [
        f"📏 [Step {step}] QuantEval (N={result['n']} held-out, {result['euler_steps']} ODE steps) | "
        f"Ref ceiling {_fmt_pct(result['ref_char_acc'])} (EOS {_fmt_pct(result['ref_eos_acc'])})"
    ]
    for m_key, m_label, _k, _p in QUANT_EVAL_MODES:
        r = result["modes"][m_key]
        lines.append(
            f"   {m_label} : CharAcc {_fmt_pct(r['char_acc'])} "
            f"(zh {_fmt_pct(r['char_acc_zh'])} / ascii {_fmt_pct(r['char_acc_ascii'])}) | "
            f"EOS {_fmt_pct(r['eos_acc'])} | Conf {_fmt_num(r['char_conf'])} | ManifoldCos {_fmt_num(r['manifold_cos'])}"
        )
    return "\n".join(lines)


def append_quant_eval_jsonl(path: str, step: int, result: Dict[str, Any]) -> bool:
    """Appends one JSON line; never raises (Drive hiccups must not kill training)."""
    try:
        rec = {"step": int(step), "time": time.time()}
        rec.update(result)
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, allow_nan=True) + "\n")
        return True
    except Exception as e:
        print(f"⚠️ [QuantEval] Failed to append JSONL '{path}': {e}")
        return False


class QuantRedFlagDetector:
    """
    Keeps the last 3 Exact-mode (CharAcc, ManifoldCos) points and flags:
      (a) CharAcc non-decreasing while ManifoldCos dropped by > cos_drop over the window (CE pleasing the translator
          while drifting off the encoder manifold);
      (b) CharAcc decreased by > acc_drop (absolute fraction, 0.03 = 3 points) between two consecutive evals.
    """

    def __init__(self, window: int = 3, cos_drop: float = 0.02, acc_drop: float = 0.03):
        self.window = int(window)
        self.cos_drop = float(cos_drop)
        self.acc_drop = float(acc_drop)
        self.history = deque(maxlen=self.window)

    def seed_from_jsonl(self, path: str, max_step: int = None) -> int:
        try:
            if not os.path.isfile(path):
                return 0
            recs = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue
                    if max_step is not None and int(r.get("step", 0)) > int(max_step):
                        continue
                    ex = r.get("modes", {}).get("exact", {})
                    recs.append((int(r["step"]), float(ex.get("char_acc", float("nan"))), float(ex.get("manifold_cos", float("nan")))))
            recs.sort(key=lambda x: x[0])
            for rec in recs[-self.window:]:
                self.history.append(rec)
            return len(self.history)
        except Exception:
            return 0

    def update(self, step: int, result: Dict[str, Any]) -> List[str]:
        ex = result["modes"]["exact"]
        self.history.append((int(step), float(ex["char_acc"]), float(ex["manifold_cos"])))
        flags = []
        h = list(self.history)
        if len(h) >= 2:
            (s0, a0, _c0), (s1, a1, _c1) = h[-2], h[-1]
            if not (math.isnan(a0) or math.isnan(a1)) and (a0 - a1) > self.acc_drop:
                flags.append(
                    f"⚠️ [RedFlag] Exact CharAcc dropped {100 * (a0 - a1):.1f} pts between consecutive evals "
                    f"(step {s0}: {100 * a0:.1f}% -> step {s1}: {100 * a1:.1f}%)"
                )
        if len(h) >= self.window:
            accs = [x[1] for x in h]
            coss = [x[2] for x in h]
            if not any(math.isnan(v) for v in accs + coss):
                acc_non_dec = all(accs[i + 1] >= accs[i] for i in range(len(accs) - 1))
                cos_drop = coss[0] - coss[-1]
                if acc_non_dec and cos_drop > self.cos_drop:
                    flags.append(
                        f"⚠️ [RedFlag] Exact CharAcc non-decreasing ({' -> '.join(f'{100 * a:.1f}%' for a in accs)}) "
                        f"while ManifoldCos dropped {cos_drop:.3f} ({' -> '.join(f'{c:.3f}' for c in coss)}) over steps "
                        f"{h[0][0]}..{h[-1][0]}: CE may be pleasing the translator while drifting off the encoder manifold"
                    )
        return flags


def load_frozen_encoder_checkpoint(encoder: TinyCharEncoderCUDA, ckpt_path: str, device: torch.device) -> None:
    if not ckpt_path or not os.path.exists(ckpt_path):
        print(f"⚠️ WARNING: Encoder checkpoint {ckpt_path} not found! Using random weights (for testing only).")
        return

    if ckpt_path.endswith(".safetensors"):
        load_mlx_safetensors_into_torch(encoder, ckpt_path)
        print(f"✅ Loaded frozen encoder (.safetensors) from {ckpt_path}")
    else:
        sd = torch.load(ckpt_path, map_location=device, weights_only=True)
        incompatible = encoder.load_state_dict(sd, strict=False)
        if incompatible.missing_keys:
            print(f"ℹ️ Legacy encoder checkpoint missing v2 keys (using defaults): {incompatible.missing_keys}")
        if incompatible.unexpected_keys:
            print(f"ℹ️ Ignoring unexpected keys in encoder checkpoint: {incompatible.unexpected_keys}")
        print(f"✅ Loaded frozen encoder from {ckpt_path}")


def train():
    default_data_path = (
        "/content/omni_knowledge_corpus.jsonl"
        if os.path.exists("/content/omni_knowledge_corpus.jsonl")
        else "data/Basic_ZH/moss_dialogues.jsonl"
    )

    parser = get_training_parser("Phase 4 Punctuation-Clause LeftBrainCrystallizer + HiPPO Flow Engine (CUDA)")
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
    parser.add_argument("--min_target_len", type=int, default=10, help="Minimum clause character length after punctuation split (default 10)")
    parser.add_argument("--max_clause_chars", type=int, default=28, help="Maximum clause character length after punctuation split (default 28)")
    parser.add_argument("--split_by_punctuation", action="store_true", default=True, help="Split multi-sentence targets into progressive 10-28 char punctuation clauses")
    parser.add_argument("--no_split_by_punctuation", dest="split_by_punctuation", action="store_false", help="Disable punctuation splitting")
    parser.add_argument("--attractor_prob", type=float, default=0.90, help="Probability of enabling k_attractors during Phase 4 training (default 0.90)")
    parser.add_argument("--pad_loss_weight", type=float, default=0.05, help="Relative weight for <PAD> tokens in flow matching MSE loss")
    parser.add_argument("--num_subbands", type=int, default=32, help="Number of Legendre harmonic modes K in Z_clause (default 32 for full-rank L=32)")
    parser.add_argument("--cavity_prob", type=float, default=0.70, help="Probability of corrupting Z_clause with spectral coarsening / multi-clause cavity mix (default 0.70)")
    parser.add_argument("--ce_weight", type=float, default=0.35, help="Weight for frozen translator token cross-entropy loss (default 0.35)")
    parser.add_argument("--quant_eval_size", type=int, default=256, help="Held-out quantitative benchmark size (excluded from training; #053)")
    parser.add_argument("--quant_eval_interval", type=int, default=500, help="Run per-mode quantitative eval every N steps (0 disables; also runs once at step 200 and before final save)")
    parser.add_argument("--quant_eval_seed", type=int, default=1234, help="Seed for held-out selection, cavity distractors and fixed ODE noise (#053)")
    parser.set_defaults(
        seq_len=32,
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

    # 1. Punctuation-Split Short-Clause Dataset (10-28 chars, seq_len=32) with Negative Distractor Removal
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

    # #053: Fixed held-out quantitative benchmark (deterministic, excluded from the training DataLoader)
    quant_held_idx = np.zeros((0,), dtype=np.int64)
    if args.quant_eval_interval and args.quant_eval_interval > 0 and args.quant_eval_size and args.quant_eval_size > 0:
        n_q = min(int(args.quant_eval_size), len(dataset) // 5)
        quant_held_idx = select_quant_eval_indices(len(dataset), n_q, args.quant_eval_seed)
    train_dataset = build_train_subset(dataset, quant_held_idx)
    if len(quant_held_idx) > 0:
        print(
            f"📏 QuantEval held-out set: {len(quant_held_idx)} items (seed={args.quant_eval_seed}) excluded from training "
            f"-> {len(train_dataset)} training items"
        )
    dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Cross-domain benchmark: select 1 representative sample per domain (up to 8 domains)
    eval_benchmark_samples = select_fixed_benchmark_indices(dataset, seed=42, max_domains=8)
    bench_domains_str = ", ".join(f"{dom} (#{idx})" for dom, idx in eval_benchmark_samples)
    print(f"📌 Fixed Cross-Domain Benchmark Samples ({len(eval_benchmark_samples)} domains): {bench_domains_str}")

    # 2. Phase 1 Encoder (Frozen Continuous Geometric Mapper)
    encoder = TinyCharEncoderCUDA(vocab_size=tokenizer.vocab_size, pad_token_id=tokenizer.pad_token_id).to(device)
    load_frozen_encoder_checkpoint(encoder, args.encoder_ckpt, device)
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

    # 4. Phase 4 HiPPO Oracle (Frozen 2D Punctuation-Clause Vector Bottleneck)
    hippo_oracle = HiPPOOracleCUDA(hidden_dim=1024, N=64, num_subbands=args.num_subbands).to(device)
    hippo_oracle.eval()

    # 5. Stateless Universal Left Brain Crystallizer v2 (`LeftBrainCrystallizer` / `FlowMatching1DUnet` #025 / #050 / #059)
    unet = LeftBrainCrystallizer(hidden_dim=1024, cond_dim=1024, num_layers=6, num_subbands=args.num_subbands).to(device)
    # Lock fresh runs onto `encoder.char_subspace_proj` before checkpoint resume (resumed runs overwrite with saved `subspace_proj`)
    if unet.bind_encoder_subspace(encoder):
        print("✅ Bound LeftBrainCrystallizer subspace_proj directly to Phase 1 v2 encoder.char_subspace_proj")
    if translator is not None:
        unet.bind_translator_codebook(translator)
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

    # #053: Quantitative per-mode eval state (benchmark tokens/noise built now; z_sent cached lazily at first eval)
    quant_enabled = translator is not None and len(quant_held_idx) > 0
    quant_bench = None
    quant_luts = None
    quant_jsonl_path = os.path.join(args.out_dir, f"{args.ckpt_prefix}_quant_eval.jsonl")
    quant_redflag = QuantRedFlagDetector(window=3, cos_drop=0.02, acc_drop=0.03)
    quant_last_step = -1
    if quant_enabled:
        quant_bench = QuantEvalBenchmark(
            dataset, quant_held_idx, tokenizer, seq_len=args.seq_len, hidden_dim=1024,
            seed=args.quant_eval_seed, num_distractors=8,
        )
        quant_luts = build_token_class_lut(tokenizer, tokenizer.vocab_size)
        n_seeded = quant_redflag.seed_from_jsonl(quant_jsonl_path, max_step=global_step)
        print(
            f"📏 QuantEval enabled: N={quant_bench.n}, interval={args.quant_eval_interval} (+ step 200 + final), "
            f"seed={args.quant_eval_seed}, log -> {quant_jsonl_path}"
            + (f" (red-flag history seeded with {n_seeded} prior evals)" if n_seeded else "")
        )
    elif len(quant_held_idx) > 0:
        print("ℹ️ QuantEval disabled: no translator available (held-out items are still excluded from training).")

    def run_quant_eval_now(step: int):
        nonlocal quant_last_step
        if not quant_enabled or step == quant_last_step:
            return
        was_training = unet.training
        unet.eval()
        try:
            t0 = time.time()
            result = run_quant_eval(
                quant_bench, unet, encoder, hippo_oracle, translator, tokenizer, device,
                amp_dtype=amp_dtype, amp_enabled=amp_enabled, euler_steps=args.euler_steps,
                num_subbands=args.num_subbands, chunk=64, token_luts=quant_luts,
            )
            result["eval_seconds"] = round(time.time() - t0, 2)
            print("\n" + format_quant_eval(step, result) + f"   ({result['eval_seconds']:.1f}s)")
            for flag in quant_redflag.update(step, result):
                print(flag)
            print("")
            append_quant_eval_jsonl(quant_jsonl_path, step, result)
            quant_last_step = step
        except Exception as e:
            print(f"⚠️ [QuantEval] Step {step} evaluation failed (training continues): {e}")
        finally:
            if was_training:
                unet.train()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(
        f"🚀 Starting Phase 4 Punctuation-Clause LeftBrainCrystallizer Training (025) "
        f"(seq_len={args.seq_len}, clause_chars=[{args.min_target_len}, {args.max_clause_chars}], "
        f"prompt_seq_len={args.prompt_seq_len}, attractor_prob={args.attractor_prob:.2f}, warmup={args.warmup_steps})..."
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
                        f"Normalized Valid Std: {norm_std:.4f} | 2D Z_clause Shape: {tuple(z_check.shape)} "
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
                    translator=translator,
                    w_target=w_tgt,
                    cavity_prob=args.cavity_prob,
                    ce_weight=args.ce_weight,
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
                    f"Loss: {loss.item():.4f} (Vel: {metrics['loss_vel'].item():.4f}, "
                    f"Slot32: {metrics['loss_slot32'].item():.4f}, "
                    f"CE: {metrics['loss_ce'].item():.4f}, "
                    f"Cos: {metrics['loss_cos'].item():.4f}) | "
                    f"Z_clause Std: {metrics['z_sent_std'].item():.3f} | "
                    f"Attractor Prob: {p_attractor:.2f} | Cavity Prob: {args.cavity_prob:.2f} | Scale: {scale_factor:.2f}x"
                )
                if "vel_exact" in metrics:
                    def _pm(key, nd=3):
                        v = metrics.get(key)
                        if v is None:
                            return "-"
                        v = float(v.item()) if torch.is_tensor(v) else float(v)
                        return "-" if math.isnan(v) else f"{v:.{nd}f}"

                    print(
                        f"    per-mode | vel exact/hiband/cavity {_pm('vel_exact')}/{_pm('vel_hiband')}/{_pm('vel_cavity')} | "
                        f"ce {_pm('ce_exact', 2)}/{_pm('ce_hiband', 2)}/{_pm('ce_cavity', 2)} | "
                        f"n {_pm('n_exact', 0)}/{_pm('n_hiband', 0)}/{_pm('n_cavity', 0)}"
                    )

            # Upgraded cross-domain evaluation: 1 representative sample per domain (up to 8 domains)
            if translator is not None and (global_step == 50 or global_step % args.eval_interval == 0):
                unet.eval()
                with torch.no_grad():
                    print("\n" + "=" * 70)
                    print(
                        f"🔮 [Step {global_step}] 2D Punctuation-Clause Cross-Domain Euler ODE Evaluation "
                        f"({len(eval_benchmark_samples)} domains, {args.euler_steps} steps)"
                    )
                    # Pre-collect benchmark Z_clause vectors so we can form deterministic multi-clause cavity distractors
                    bench_entries = []
                    for dom_name, ds_idx in eval_benchmark_samples:
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
                        bench_entries.append(
                            (dom_name, ds_idx, w_p_s, w_t_s, tgt_valid_mask, p_valid_mask, p_sample, tgt_sample, z_sent)
                        )

                    z_bench_all = torch.cat([e[8] for e in bench_entries], dim=0)  # (N_bench, 1024)
                    n_bench = z_bench_all.shape[0]

                    for s_idx, (
                        dom_name,
                        ds_idx,
                        w_p_s,
                        w_t_s,
                        tgt_valid_mask,
                        p_valid_mask,
                        p_sample,
                        tgt_sample,
                        z_sent,
                    ) in enumerate(bench_entries):
                        # Build Cavity/Low-Freq (alpha=0.40 & k=0..5) corrupted Z_clause alongside Exact Z_clause
                        if n_bench > 1:
                            other_mask = [i for i in range(n_bench) if i != s_idx][:4]
                            z_dist_mean = z_bench_all[other_mask].mean(dim=0, keepdim=True)
                        else:
                            z_dist_mean = torch.zeros_like(z_sent)
                        z_cav = 0.40 * z_sent + 0.60 * z_dist_mean
                        z_cav = z_cav / z_cav.float().std(dim=-1, keepdim=True).clamp(min=1e-5).to(z_cav.dtype)

                        z_lowf_sub = z_sent.view(1, args.num_subbands, -1).clone()
                        z_lowf_sub[:, 6:, :] = 0.0
                        z_lowf = z_lowf_sub.reshape(1, -1)
                        z_lowf = z_lowf / z_lowf.float().std(dim=-1, keepdim=True).clamp(min=1e-5).to(z_lowf.dtype)

                        z_eval_3 = torch.cat([z_sent, z_cav, z_lowf], dim=0)  # (3, 1024)
                        p_sample_3 = p_sample.expand(3, -1, -1)
                        p_mask_3 = p_valid_mask.expand(3, -1)

                        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                            x_gen_3 = sample_euler_ode(
                                unet=unet,
                                x_prompt=p_sample_3,
                                h_potential=z_eval_3,
                                steps=args.euler_steps,
                                seq_len=args.seq_len,
                                prompt_mask=p_mask_3,
                                device=device,
                            )

                        pred_3 = translator.decode_greedy(x_gen_3.float())
                        gen_text = decode_with_eos_marker(tokenizer, pred_3[0].tolist())
                        cav_text = decode_with_eos_marker(tokenizer, pred_3[1].tolist())
                        lowf_text = decode_with_eos_marker(tokenizer, pred_3[2].tolist())

                        ref_tokens = translator.decode_greedy(tgt_sample.float())[0].tolist()
                        ref_text = decode_with_eos_marker(tokenizer, ref_tokens)

                        prompt_text = tokenizer.decode(w_p_s[0].tolist(), skip_special_tokens=True)
                        valid_len = int(tgt_valid_mask.sum().item())

                        print(f"--- [{dom_name}] Sample #{s_idx + 1} (idx={ds_idx}) ---")
                        print(f"Prompt + Knowledge (k_attractors): {prompt_text}")
                        print(f"Target Clause (len={valid_len}, Z_clause shape: {tuple(z_sent.shape)}): {ref_text}")
                        print(f"Left-Brain Crystallized [Exact Z_clause]          : {gen_text}")
                        print(f"Left-Brain Crystallized [Cavity/Low-Freq (α=0.40 & k=0..5)]: [α=0.40] {cav_text} | [k=0..5] {lowf_text}")
                    print("=" * 70 + "\n")

                unet.train()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # #053: Fixed held-out per-mode quantitative eval (step 200 early sanity + every quant_eval_interval)
            if quant_enabled and (global_step == 200 or global_step % args.quant_eval_interval == 0):
                run_quant_eval_now(global_step)

            if args.save_steps and global_step % args.save_steps == 0:
                checkpointer.save(global_step)

        run_quant_eval_now(global_step)  # #053: right before the final (end-of-epoch) checkpoint save
        checkpointer.save(global_step)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    train()
