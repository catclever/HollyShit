#!/usr/bin/env python3
# =====================================================================================================
# [Diagnostic Script - Task #031: Left-Brain Attractor Robustness & Distractor Contamination Evaluation]
# 1. Purpose:
#    Evaluates the frozen `LeftBrainCrystallizer` (K=32 full-rank Legendre basis) + `ContinuousToTextTranslatorCUDA`
#    when fed ground-truth 2D HiPPO clause vectors `z_clause = HiPPOOracleCUDA(unet.normalize(x_target))`
#    under 4 controlled `k_attractors` conditions (using the EXACT same initial Gaussian noise `x_0`
#    across all 4 conditions per sample):
#      - Condition A [Relevant / Gold]:       k_attractors = true relevant context/knowledge clauses
#      - Condition B [Pure Distractor]:       k_attractors = 3 unrelated cross-domain clauses
#      - Condition C [Empty / None]:          k_attractors = None, k_mask = None (pure 2D z_clause)
#      - Condition D [Mixed: 1 Rel + 3 Dist]: k_attractors = 1 relevant + 3 distractor clauses (shuffled)
# 2. Method:
#    - Auto-discovers `basencoder_step_115000/student.pt`, `left_brain_s32_fullrank*`, and
#      `translator_moss_s32_clean*` across local and Colab Google Drive checkpoint directories.
#    - Always evaluates a Built-in 8-Sample Multi-Domain Canonical Suite (Math Formula, Python Error,
#      Physics Encyclopedia `Kampen`, Advice QA, English phrase, CSS code, SQL query `0 AND rank = "14"`,
#      Sports News `据《密尔沃基哨兵报》报道，`), plus up to `--num_samples` records from
#      `omni_knowledge_corpus.jsonl` via `BicameralClauseChainDataset` if available.
#    - Computes Latent Cosine Similarity, Position-wise Token Accuracy (%), SequenceMatcher Edit
#      Similarity (%), Low-Frequency/Entity Character Recall (%), Distractor Contamination Rate (%),
#      plus internal `OrthogonalAttractor` `MaxSim`, `SnapGate` activation, and trajectory shift `||x - x_C||`.
# 3. Caveats:
#    - Read-only diagnostic tool; never mutates model weights or checkpoints.
#    - When run without trained checkpoints (e.g. CPU smoke test), falls back to random initialization
#      and auto-calibrates `unet.calibrate_manifold` on the target batch so all tensor paths execute cleanly.
# =====================================================================================================

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import sys
import re
import glob
import math
import random
import difflib
import argparse
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from training.core.char_tokenizer import CharTokenizer
from model.phase1_charencoder_cuda import TinyCharEncoderCUDA
from model.continuous_translator_cuda import ContinuousToTextTranslatorCUDA
from model.hippo_oracle import HiPPOOracleCUDA
from model.flow_unet_cuda import LeftBrainCrystallizer
from training.data.knowledge_dataset import BicameralClauseChainDataset

# [Bug Fix - Task #031]
# 1. Problem: `_char_bigrams` only exists in the unpushed #029 version of knowledge_dataset.py, so
#    Colab (which clones from GitHub) raised ImportError.
# 2. Resolution: define a local, identical copy so this diagnostic has no dependency on #029.
# 3. Caveat: keep this script self-contained; do not import #029-only helpers here.
import unicodedata as _unicodedata


def _char_bigrams(text: str) -> set:
    """Character bigrams of `text` with whitespace and Unicode punctuation (category P*) stripped."""
    if not text:
        return set()
    norm = "".join(
        ch for ch in text
        if not ch.isspace() and not _unicodedata.category(ch).startswith("P")
    )
    return {norm[i : i + 2] for i in range(len(norm) - 1)}


# Top-100 common Chinese functional / structural characters excluded from distractor contamination checks
COMMON_FUNCTIONAL_CHARS = set(
    "的了在是有和与或等中为对从到以及而但并把被让将给按于及其"
    "这那你我他她它人们个上下左右大小多少好坏高低长短新旧"
    "一二三四五六七八九十百千万零"
    "不也没很都还就又才只可要会能得着过起出进回来去"
    "说看想知用做工作时年月日天分秒号第次种样点面线体"
    "因为所以如果虽然但是因此或者而且能够可以需要应该通过进行"
    "问题答案方法结果数据信息系统部分情况过程作用关系"
)


@dataclass
class DiagnosticSample:
    sample_id: str
    domain: str
    target_clause: str
    relevant_clauses: List[str]
    distractor_clauses: List[str]
    source: str = "canonical"


BUILTIN_CANONICAL_SUITE: List[DiagnosticSample] = [
    DiagnosticSample(
        sample_id="canon_1",
        domain="math_formula",
        target_clause="圆的面积公式是 S = πr²，",
        relevant_clauses=[
            "已知圆的半径为 r，圆周率记作 π，面积公式是 S = πr²。",
            "求圆的面积计算公式 S = πr²，其中 r 表示半径。",
        ],
        distractor_clauses=[
            "SELECT user_id FROM orders WHERE status = 908;",
            "1973年阿姆斯特丹条约由布鲁因斯玛签署，",
            "error_code = 404 表示远程服务器资源未找到。",
        ],
    ),
    DiagnosticSample(
        sample_id="canon_2",
        domain="python_error",
        target_clause="因此这里会抛出错误。",
        relevant_clauses=[
            "在 Python 中对空值调用方法会触发异常，因此这里会抛出错误。",
            "当变量为 None 时直接访问属性会导致程序抛出错误。",
        ],
        distractor_clauses=[
            "《密尔沃基哨兵报》创刊于1837年威斯康星州，",
            "margin: 0 auto; border-radius: 12px;",
            "欧几里得几何中三角形内角和恒等于 180 度。",
        ],
    ),
    DiagnosticSample(
        sample_id="canon_3",
        domain="physics_kampen",
        target_clause="证明了范坎彭（Kampen）的假设。",
        relevant_clauses=[
            "物理学家范坎彭（Kampen）曾提出著名的量子散射相位假设，",
            "最新激光干涉实验观测结果证明了范坎彭（Kampen）的假设。",
        ],
        distractor_clauses=[
            "UPDATE accounts SET balance = 7500 WHERE id = 99;",
            "东京铁塔高 333 米，于 1958 年正式竣工，",
            "padding: 16px 24px; color: #ff5733;",
        ],
    ),
    DiagnosticSample(
        sample_id="canon_4",
        domain="advice_qa",
        target_clause="希望能帮到你：",
        relevant_clauses=[
            "下面为你整理了三条改善睡眠的实用建议，希望能帮到你：",
            "请参考以下日常作息调整方案，希望能帮到你。",
        ],
        distractor_clauses=[
            "SELECT COUNT(*) FROM logs WHERE http_code = 502;",
            "1889年埃菲尔铁塔在法国巴黎世博会正式落成，",
            "const maxRetryLimit = 64; let timeoutMs = 3000;",
        ],
    ),
    DiagnosticSample(
        sample_id="canon_5",
        domain="english_phrase",
        target_clause="in the middle of",
        relevant_clauses=[
            "Please put the wooden table in the middle of the living room,",
            "The English phrase in the middle of means at the center.",
        ],
        distractor_clauses=[
            "据《密尔沃基哨兵报》报道，雄鹿队以 118 比 104 获胜，",
            "球体体积计算公式为 V = 4/3 πr³，",
            "数据库连接池报错 KeyError: 'session_token_99'。",
        ],
    ),
    DiagnosticSample(
        sample_id="canon_6",
        domain="css_code",
        target_clause="margin-bottom: 30px;",
        relevant_clauses=[
            "设置卡片底部外边距为30像素：margin-bottom: 30px;",
            ".card-box { margin-bottom: 30px; display: flex; }",
        ],
        distractor_clauses=[
            "范坎彭（Kampen）在莱顿大学发表了统计物理论文，",
            "西汉司马迁撰写的《史记》全书共一百三十篇，",
            "WHERE dept_id = 42 AND monthly_salary > 8500",
        ],
    ),
    DiagnosticSample(
        sample_id="canon_7",
        domain="sql_query",
        target_clause='0 AND rank = "14"',
        relevant_clauses=[
            '查询过滤条件要求 status > 0 AND rank = "14"，',
            'SELECT * FROM player_stats WHERE active > 0 AND rank = "14";',
        ],
        distractor_clauses=[
            "margin-top: 25px; background-color: #1a2b3c;",
            "1927年第五届索尔维物理学会议在布鲁塞尔召开，",
            "每天坚持慢跑 45 分钟有助于降低心血管风险。",
        ],
    ),
    DiagnosticSample(
        sample_id="canon_8",
        domain="sports_news",
        target_clause="据《密尔沃基哨兵报》报道，",
        relevant_clauses=[
            "美国威斯康星州《密尔沃基哨兵报》今日发布了体育专刊，",
            "据《密尔沃基哨兵报》报道，球队已正式签下新任主教练。",
        ],
        distractor_clauses=[
            "def compute_hmac_digest(payload_bytes, salt=8192):",
            "圆锥的体积公式为 V = 1/3 πr²h，",
            'WHERE account_flag = 0 AND rank = "99"',
        ],
    ),
]


def build_search_dirs(ckpt_dir: Optional[str] = None) -> List[str]:
    raw_dirs = [
        ckpt_dir,
        "checkpoints/run",
        "checkpoints",
        "/content/drive/MyDrive/hollyshit/checkpoints/run",
        "/content/drive/MyDrive/hollyshit/checkpoints",
        os.path.join(PROJECT_ROOT, "checkpoints/run"),
        os.path.join(PROJECT_ROOT, "checkpoints"),
    ]
    out: List[str] = []
    for d in raw_dirs:
        if d and d not in out and os.path.exists(d):
            out.append(d)
    return out


def discover_encoder_ckpt(explicit_path: Optional[str], search_dirs: Sequence[str]) -> Optional[str]:
    if explicit_path and os.path.isfile(explicit_path):
        return explicit_path
    candidates = [
        explicit_path,
        "checkpoints/basencoder_step_115000/student.pt",
        "/content/drive/MyDrive/hollyshit/checkpoints/basencoder_step_115000/student.pt",
        os.path.join(PROJECT_ROOT, "checkpoints/basencoder_step_115000/student.pt"),
    ]
    for bdir in search_dirs:
        candidates.append(os.path.join(bdir, "basencoder_step_115000", "student.pt"))
        for match in sorted(glob.glob(os.path.join(bdir, "basencoder_step_*", "student.pt"))):
            candidates.append(match)
    for c in candidates:
        if c and os.path.isfile(c):
            return c
    return None


def discover_left_brain_ckpt(
    unet: nn.Module,
    explicit_path: Optional[str],
    search_dirs: Sequence[str],
    device: Union[str, torch.device] = "cpu",
) -> Optional[str]:
    if explicit_path:
        if os.path.isdir(explicit_path):
            explicit_path = os.path.join(explicit_path, "unet.pt")
        if os.path.isfile(explicit_path):
            sd = torch.load(explicit_path, map_location=device, weights_only=True)
            unet.load_state_dict(sd, strict=True)
            return explicit_path

    preferred_prefixes = ("left_brain_s32_fullrank", "phase4_sentvec_flow", "hippo_flow_moss")
    candidates: List[Tuple[int, int, str]] = []
    seen = set()

    for bdir in search_dirs:
        if not os.path.isdir(bdir):
            continue
        for entry in os.listdir(bdir):
            full_dir = os.path.join(bdir, entry)
            if not os.path.isdir(full_dir):
                continue
            pt_file = os.path.join(full_dir, "unet.pt")
            if not os.path.isfile(pt_file) or pt_file in seen:
                continue

            priority = -1
            for p_idx, pref in enumerate(preferred_prefixes):
                if pref in entry:
                    priority = len(preferred_prefixes) - p_idx
                    break
            if priority < 0 and ("left_brain" in entry or "phase4" in entry or "hippo_flow" in entry):
                priority = 0
            if priority < 0:
                continue

            m = re.search(r"step_(\d+)", entry)
            step = int(m.group(1)) if m else 0
            seen.add(pt_file)
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


def discover_translator_ckpt(
    explicit_path: Optional[str],
    search_dirs: Sequence[str],
) -> Optional[str]:
    if explicit_path:
        if os.path.isdir(explicit_path):
            explicit_path = os.path.join(explicit_path, "translator.pt")
        if os.path.isfile(explicit_path):
            return explicit_path

    candidates: List[Tuple[int, int, str]] = []
    seen = set()
    for bdir in search_dirs:
        if not os.path.isdir(bdir):
            continue
        for entry in os.listdir(bdir):
            if "translator" not in entry:
                continue
            full_dir = os.path.join(bdir, entry)
            pt_file = os.path.join(full_dir, "translator.pt")
            if not os.path.isfile(pt_file) or pt_file in seen:
                continue
            seen.add(pt_file)
            m = re.search(r"step_(\d+)", entry)
            step = int(m.group(1)) if m else 0
            if "translator_moss_s32_clean" in entry:
                prio = 3
            elif "s32_clean" in entry:
                prio = 2
            elif "deblur" not in entry:
                prio = 1
            else:
                prio = 0
            candidates.append((prio, step, pt_file))

    if candidates:
        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return candidates[0][2]
    return None


def discover_data_path(explicit_path: Optional[str]) -> Optional[str]:
    candidates = [
        explicit_path,
        "/content/omni_knowledge_corpus.jsonl",
        "/content/drive/MyDrive/hollyshit/data/omni_knowledge_corpus.jsonl",
        "data/omni_knowledge_corpus.jsonl",
        os.path.join(PROJECT_ROOT, "data/omni_knowledge_corpus.jsonl"),
        "data/Basic_ZH/moss_dialogues.jsonl",
        os.path.join(PROJECT_ROOT, "data/Basic_ZH/moss_dialogues.jsonl"),
    ]
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None


def encode_clause_tokens(tokenizer: CharTokenizer, text: str, seq_len: int = 32) -> List[int]:
    ids = list(tokenizer.encode(text, add_special_tokens=True))
    if len(ids) > seq_len:
        ids = ids[:seq_len]
        if tokenizer.eos_token_id is not None:
            ids[-1] = tokenizer.eos_token_id
    elif len(ids) < seq_len:
        ids = ids + [tokenizer.pad_token_id] * (seq_len - len(ids))
    return ids


def decode_clause_clean(tokenizer: CharTokenizer, token_ids: Sequence[int]) -> str:
    ids = list(token_ids)
    if tokenizer.eos_token_id in ids:
        ids = ids[: ids.index(tokenizer.eos_token_id)]
    return tokenizer.decode(ids, skip_special_tokens=True).strip()


@torch.no_grad()
def encode_tokens(encoder: TinyCharEncoderCUDA, token_ids: torch.Tensor) -> torch.Tensor:
    h = encoder.get_normalized_embeddings(token_ids)
    for layer in encoder.layers:
        h = layer(h, None)
    return encoder.final_ln(h)


def pack_clauses_as_attractor_tokens(
    tokenizer: CharTokenizer,
    clauses: Sequence[str],
    seq_len: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Packs a list of clauses into a 2D token tensor `w_attractors: (1, N_clauses * seq_len)`
    and boolean validity mask `k_mask: (1, N_clauses * seq_len)`, matching `KnowledgeSelector.select_active_clause_attractors`.
    """
    flat_ids: List[int] = []
    for c in clauses:
        if c and c.strip():
            flat_ids.extend(encode_clause_tokens(tokenizer, c.strip(), seq_len=seq_len))
    if not flat_ids:
        flat_ids = [tokenizer.pad_token_id] * seq_len
    w_attr = torch.tensor([flat_ids], dtype=torch.long, device=device)
    k_mask = w_attr != tokenizer.pad_token_id
    return w_attr, k_mask


@torch.no_grad()
def compute_attractor_snap_diagnostics(
    unet: LeftBrainCrystallizer,
    x_traj: torch.Tensor,
    x_attractors_norm: Optional[torch.Tensor],
    k_mask: Optional[torch.Tensor],
    valid_mask: torch.Tensor,
) -> Tuple[float, float]:
    """
    Computes the mean `max_sim` and `snap_gate = sigmoid((max_sim - snap_threshold) * 20.0)`
    across all U-Net blocks' `OrthogonalAttractor` modules on valid target positions.
    """
    if x_attractors_norm is None:
        return 0.0, 0.0

    pos_k = unet._get_pos_emb(
        x_attractors_norm.shape[1], x_attractors_norm.device, x_attractors_norm.dtype
    )
    k = x_attractors_norm + 0.02 * pos_k
    x = x_traj
    v_mask = valid_mask[0].bool()

    max_sims: List[float] = []
    snap_gates: List[float] = []

    for block in unet.blocks:
        attr = block.attractor
        mix = torch.tanh(attr.span_mix)
        x_span = x + mix * attr.span_conv(x.transpose(1, 2)).transpose(1, 2)
        k_clean = k * k_mask.unsqueeze(-1).to(k.dtype) if k_mask is not None else k
        k_span = k_clean + mix * attr.span_conv(k_clean.transpose(1, 2)).transpose(1, 2)

        q_unit = F.normalize(attr.norm_x(x_span) * attr.channel_weight, dim=-1, eps=1e-6)
        k_unit = F.normalize(attr.norm_k(k_span) * attr.channel_weight, dim=-1, eps=1e-6)
        sim = torch.bmm(q_unit, k_unit.transpose(1, 2))

        if sim.shape[1] > 1 and sim.shape[2] > 1:
            s_pos = F.relu(sim.masked_fill(~k_mask.unsqueeze(1), 0.0) if k_mask is not None else sim)
            diag_prev = torch.zeros_like(s_pos)
            diag_next = torch.zeros_like(s_pos)
            diag_prev[:, 1:, 1:] = s_pos[:, :-1, :-1]
            diag_next[:, :-1, :-1] = s_pos[:, 1:, 1:]
            diag_relay = 0.5 * (diag_prev + diag_next)

            same_prev = torch.zeros_like(s_pos)
            same_next = torch.zeros_like(s_pos)
            same_prev[:, 1:, :] = s_pos[:, :-1, :]
            same_next[:, :-1, :] = s_pos[:, 1:, :]
            same_key_competitor = torch.maximum(same_prev, same_next)
            sim = sim + 0.25 * diag_relay - 0.30 * same_key_competitor

        if k_mask is not None:
            sim = sim.masked_fill(~k_mask.unsqueeze(1), -1e4)

        max_sim = sim.max(dim=-1).values  # (1, L_x)
        gate = torch.sigmoid((max_sim - attr.snap_threshold) * 20.0)
        if v_mask.any():
            max_sims.append(max_sim[0, v_mask].float().mean().item())
            snap_gates.append(gate[0, v_mask].float().mean().item())

    return (
        sum(max_sims) / max(1, len(max_sims)),
        sum(snap_gates) / max(1, len(snap_gates)),
    )


@torch.no_grad()
def crystallize_with_fixed_noise(
    unet: LeftBrainCrystallizer,
    z_clause: torch.Tensor,
    x_0: torch.Tensor,
    x_attractors_norm: Optional[torch.Tensor],
    k_mask: Optional[torch.Tensor],
    steps: int = 10,
) -> torch.Tensor:
    """
    Integrates `LeftBrainCrystallizer` from the caller-supplied initial Gaussian noise `x_0`
    so that Conditions A, B, C, and D share the exact same starting trajectory `x_0`.
    """
    B = z_clause.shape[0]
    device = z_clause.device
    x = x_0.clone().to(device=device, dtype=z_clause.dtype)
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


def is_content_char(ch: str) -> bool:
    if not ch or ch.isspace():
        return False
    if unicodedata.category(ch).startswith("P"):
        return False
    return True


def extract_entity_chars(target_text: str, relevant_clauses: Sequence[str]) -> List[str]:
    """
    Identifies low-frequency / entity / symbol / alphanumeric characters in `target_text`:
      - Any ASCII digit or letter (`0-9`, `a-z`, `A-Z`), math/symbol char, OR
      - Any CJK content character not in `COMMON_FUNCTIONAL_CHARS` that also appears in `relevant_clauses`.
    """
    rel_joined = "".join(relevant_clauses)
    rel_set = set(rel_joined)
    out: List[str] = []
    for ch in target_text:
        if not is_content_char(ch):
            continue
        is_ascii_alnum = ch.isascii() and ch.isalnum()
        is_math_sym = unicodedata.category(ch).startswith("S")
        is_rare_shared = (ch not in COMMON_FUNCTIONAL_CHARS) and (ch in rel_set)
        if is_ascii_alnum or is_math_sym or is_rare_shared:
            out.append(ch)
    return out


def extract_exclusive_distractor_chars(
    distractor_clauses: Sequence[str],
    target_text: str,
    relevant_clauses: Sequence[str],
) -> set:
    """
    Returns the set of content characters that appear in `distractor_clauses` but NEVER appear in
    `target_text`, `relevant_clauses`, or `COMMON_FUNCTIONAL_CHARS`.
    """
    forbidden = set(target_text) | set("".join(relevant_clauses)) | COMMON_FUNCTIONAL_CHARS
    dist_chars = set()
    for clause in distractor_clauses:
        for ch in clause:
            if is_content_char(ch) and ch not in forbidden:
                dist_chars.add(ch)
    return dist_chars


def compute_sample_metrics(
    x_gen: torch.Tensor,
    x_gen_empty: torch.Tensor,
    x_target: torch.Tensor,
    valid_mask: torch.Tensor,
    pred_ids: List[int],
    tgt_ids: List[int],
    pred_text: str,
    target_text: str,
    relevant_clauses: Sequence[str],
    distractor_clauses: Sequence[str],
    max_sim: float = 0.0,
    snap_gate: float = 0.0,
) -> Dict[str, float]:
    """
    Computes all diagnostic metrics for a single (sample, condition) pair.
    """
    mask_1d = valid_mask[0].bool()
    if mask_1d.any():
        latent_cos = (
            F.cosine_similarity(
                x_gen[0, mask_1d].float(),
                x_target[0, mask_1d].float(),
                dim=-1,
            )
            .mean()
            .item()
        )
        shift_l2 = (
            (x_gen[0, mask_1d].float() - x_gen_empty[0, mask_1d].float())
            .norm(p=2, dim=-1)
            .mean()
            .item()
        )
    else:
        latent_cos = 0.0
        shift_l2 = 0.0

    l_valid = int(mask_1d.sum().item())
    if l_valid > 0:
        matches = sum(1 for p, t in zip(pred_ids[:l_valid], tgt_ids[:l_valid]) if p == t)
        pos_token_acc = (matches / float(l_valid)) * 100.0
    else:
        pos_token_acc = 0.0

    if target_text or pred_text:
        edit_sim = difflib.SequenceMatcher(None, target_text, pred_text).ratio() * 100.0
    else:
        edit_sim = 100.0

    entity_chars = extract_entity_chars(target_text, relevant_clauses)
    if entity_chars:
        pred_char_counts: Dict[str, int] = {}
        for ch in pred_text:
            pred_char_counts[ch] = pred_char_counts.get(ch, 0) + 1
        hits = 0
        for ch in entity_chars:
            if pred_char_counts.get(ch, 0) > 0:
                hits += 1
                pred_char_counts[ch] -= 1
        entity_recall = (hits / float(len(entity_chars))) * 100.0
        has_entity = 1.0
    else:
        entity_recall = 0.0
        has_entity = 0.0

    excl_dist_chars = extract_exclusive_distractor_chars(
        distractor_clauses=distractor_clauses,
        target_text=target_text,
        relevant_clauses=relevant_clauses,
    )
    leaked_chars = [ch for ch in pred_text if ch in excl_dist_chars]
    contaminated = 100.0 if len(leaked_chars) > 0 else 0.0

    return {
        "latent_cos": latent_cos,
        "pos_token_acc": pos_token_acc,
        "edit_sim": edit_sim,
        "entity_recall": entity_recall,
        "has_entity": has_entity,
        "contaminated": contaminated,
        "leaked_char_count": float(len(leaked_chars)),
        "max_sim": max_sim,
        "snap_gate": snap_gate,
        "shift_l2": shift_l2,
    }


def sample_corpus_records(
    data_path: str,
    tokenizer: CharTokenizer,
    num_samples: int,
    seq_len: int = 32,
    seed: int = 42,
) -> List[DiagnosticSample]:
    """
    Loads records from `data_path` using `BicameralClauseChainDataset` and constructs
    `DiagnosticSample` instances with true relevant clauses (K+) and cross-domain distractors (K-).
    """
    if num_samples <= 0 or not data_path or not os.path.exists(data_path):
        return []

    max_load = max(num_samples * 6, 300)
    ds_kwargs = {
        "data_path": data_path,
        "tokenizer": tokenizer,
        "seq_len": seq_len,
        "max_samples": max_load,
        "seed": seed,
        "verbose": False,
    }
    import inspect
    sig = inspect.signature(BicameralClauseChainDataset.__init__)
    if "label_stats_sample" in sig.parameters:
        ds_kwargs["label_stats_sample"] = 0
    ds = BicameralClauseChainDataset(**ds_kwargs)
    if len(ds) == 0:
        return []

    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)

    samples: List[DiagnosticSample] = []
    for idx in indices:
        if len(samples) >= num_samples:
            break
        rec = ds.records[idx]
        a_clauses: List[str] = rec.get("a_clauses", [])
        q_clauses: List[str] = rec.get("q_clauses", [])
        pos_clauses: List[str] = rec.get("pos_clauses", [])
        if not a_clauses:
            continue

        if hasattr(ds, "_build_cand_entries"):
            cand_entries = ds._build_cand_entries(idx)
            pos_cands = [c for c, lbl, val in cand_entries if val and c and lbl >= 0.5]
            neg_cands = [c for c, lbl, val in cand_entries if val and c and lbl < 0.5]
        else:
            pos_cands = list(rec.get("pos_clauses", []))
            if not pos_cands:
                pos_cands = list(rec.get("q_clauses", []))[-2:]
            neg_cands = list(rec.get("neg_clauses", []))

        if not pos_cands:
            pos_cands = (pos_clauses + q_clauses)[-2:]
        if not pos_cands:
            continue

        best_target = a_clauses[0]
        best_overlap = -1.0
        pos_bg = set()
        for pc in pos_cands:
            pos_bg |= _char_bigrams(pc)
        for ac in a_clauses:
            ac_bg = _char_bigrams(ac)
            ov = len(ac_bg & pos_bg) / float(max(1, len(ac_bg))) if ac_bg else 0.0
            if ov > best_overlap:
                best_overlap = ov
                best_target = ac

        pos_cands_sorted = sorted(
            pos_cands,
            key=lambda c: len(_char_bigrams(c) & _char_bigrams(best_target)),
            reverse=True,
        )

        if len(neg_cands) < 3:
            for fb in BUILTIN_CANONICAL_SUITE[(idx % len(BUILTIN_CANONICAL_SUITE))].distractor_clauses:
                if fb not in neg_cands and fb != best_target:
                    neg_cands.append(fb)
                if len(neg_cands) >= 3:
                    break

        samples.append(
            DiagnosticSample(
                sample_id=f"corpus_{idx}",
                domain=str(rec.get("domain", "general") or "general"),
                target_clause=best_target,
                relevant_clauses=pos_cands_sorted[:3],
                distractor_clauses=neg_cands[:3],
                source="corpus",
            )
        )

    return samples


def inspect_unet_attractor_parameters(unet: LeftBrainCrystallizer) -> None:
    print("\n🔍 [LeftBrainCrystallizer Attractor Gate Inspection]")
    for b_idx, block in enumerate(unet.blocks):
        gate_tanh = torch.tanh(block.attractor_gate.detach().float())
        gate_mean = gate_tanh.mean().item()
        gate_max = gate_tanh.abs().max().item()
        scale = block.attractor.logit_scale.detach().float().item()
        span_mix = torch.tanh(block.attractor.span_mix.detach().float()).item()
        struct_gate = torch.tanh(block.struct_gate.detach().float()).mean().item()
        print(
            f"   Block {b_idx}: tanh(attractor_gate) mean={gate_mean:+.4f} (max|g|={gate_max:.4f}) | "
            f"logit_scale={scale:.2f} | tanh(span_mix)={span_mix:+.4f} | tanh(struct_gate) mean={struct_gate:+.4f}"
        )


def summarize_condition_metrics(
    records: List[Dict[str, float]],
) -> Dict[str, float]:
    n = max(1, len(records))
    ent_records = [r["entity_recall"] for r in records if r["has_entity"] > 0.5]
    return {
        "latent_cos": sum(r["latent_cos"] for r in records) / n,
        "pos_token_acc": sum(r["pos_token_acc"] for r in records) / n,
        "edit_sim": sum(r["edit_sim"] for r in records) / n,
        "entity_recall": (sum(ent_records) / len(ent_records)) if ent_records else 0.0,
        "contaminated": sum(r["contaminated"] for r in records) / n,
        "leaked_char_count": sum(r["leaked_char_count"] for r in records) / n,
        "max_sim": sum(r["max_sim"] for r in records) / n,
        "snap_gate": sum(r["snap_gate"] for r in records) / n,
        "shift_l2": sum(r["shift_l2"] for r in records) / n,
    }


def print_summary_table(title: str, cond_summaries: Dict[str, Dict[str, float]], n_samples: int) -> None:
    print("\n" + "=" * 126)
    print(f"📊 {title} (N = {n_samples} samples, identical initial noise x_0 across A/B/C/D)")
    print("=" * 126)
    header = (
        f"{'Condition':<26} | {'Latent Cos':>10} | {'Pos Acc':>8} | "
        f"{'Edit Sim':>9} | {'Entity Rec':>10} | {'Contam Rate':>11} | "
        f"{'MaxSim':>7} | {'SnapGate':>8} | {'ShiftL2':>8}"
    )
    print(header)
    print("-" * 126)
    labels = [
        ("A", "A: Relevant (Gold K+)"),
        ("B", "B: Pure Distractor (K-)"),
        ("C", "C: Empty (None / 2D only)"),
        ("D", "D: Mixed (1 Rel + 3 Dist)"),
    ]
    for key, label in labels:
        m = cond_summaries[key]
        print(
            f"{label:<26} | {m['latent_cos']:>10.4f} | {m['pos_token_acc']:>7.2f}% | "
            f"{m['edit_sim']:>8.2f}% | {m['entity_recall']:>9.2f}% | {m['contaminated']:>10.2f}% | "
            f"{m['max_sim']:>7.3f} | {m['snap_gate']:>8.3f} | {m['shift_l2']:>8.4f}"
        )
    print("=" * 126)


def print_diagnostic_verdict(summary: Dict[str, Dict[str, float]]) -> None:
    acc_a = summary["A"]["edit_sim"]
    acc_b = summary["B"]["edit_sim"]
    acc_c = summary["C"]["edit_sim"]
    acc_d = summary["D"]["edit_sim"]

    pos_a = summary["A"]["pos_token_acc"]
    pos_b = summary["B"]["pos_token_acc"]
    pos_c = summary["C"]["pos_token_acc"]
    pos_d = summary["D"]["pos_token_acc"]

    ent_a = summary["A"]["entity_recall"]
    ent_c = summary["C"]["entity_recall"]
    ent_d = summary["D"]["entity_recall"]

    contam_b = summary["B"]["contaminated"]
    contam_d = summary["D"]["contaminated"]

    gate_a = summary["A"]["snap_gate"]
    gate_b = summary["B"]["snap_gate"]
    sim_a = summary["A"]["max_sim"]
    sim_b = summary["B"]["max_sim"]

    delta_bc_edit = acc_b - acc_c
    delta_bc_pos = pos_b - pos_c
    delta_da_edit = acc_d - acc_a
    delta_ac_ent = ent_a - ent_c

    print("\n" + "🏁" * 42)
    print("🩺 AUTOMATED LEFT-BRAIN ATTRACTOR ROBUSTNESS VERDICT")
    print("🏁" * 42)

    print(
        f"1. [Gold Attractor Lift (A vs C)]:\n"
        f"   - Edit Similarity:  A={acc_a:.2f}% vs C={acc_c:.2f}% (Δ = {acc_a - acc_c:+.2f}%)\n"
        f"   - Pos Token Acc:    A={pos_a:.2f}% vs C={pos_c:.2f}% (Δ = {pos_a - pos_c:+.2f}%)\n"
        f"   - Entity Recall:    A={ent_a:.2f}% vs C={ent_c:.2f}% (Δ = {delta_ac_ent:+.2f}%)"
    )

    print(
        f"\n2. [Pure Distractor Pollution & Snap-Gate Saturation Check (B vs C)]:\n"
        f"   - Edit Similarity:  B={acc_b:.2f}% vs C={acc_c:.2f}% (Δ(B-C) = {delta_bc_edit:+.2f}%)\n"
        f"   - Pos Token Acc:    B={pos_b:.2f}% vs C={pos_c:.2f}% (Δ(B-C) = {delta_bc_pos:+.2f}%)\n"
        f"   - Contamination(B): {contam_b:.2f}% of samples leaked foreign distractor characters\n"
        f"   - Attractor Gate:   MaxSim(A)={sim_a:.3f} (SnapGate={gate_a:.3f}) vs MaxSim(B)={sim_b:.3f} (SnapGate={gate_b:.3f})"
    )
    if gate_b > 0.75:
        print(
            f"   ⚠️ NOTE: Internal `OrthogonalAttractor.snap_threshold=0.08` is saturated open on pure distractors "
            f"(SnapGate(B)={gate_b:.3f} because MaxSim(B)={sim_b:.3f} >> 0.08)."
        )

    if delta_bc_edit < -2.0 or contam_b >= 10.0:
        b_verdict = (
            "❌ VULNERABLE TO PURE DISTRACTORS: Feeding unrelated attractors (B) degrades fidelity below "
            "the attractor-free baseline (C) or leaks foreign entities into decoded text."
        )
        need_sink = True
    else:
        b_verdict = (
            "✅ RESILIENT TO PURE DISTRACTORS: Decoded text under pure distractors (B) matches the "
            "attractor-free baseline (C) with negligible foreign character leakage."
        )
        need_sink = False
    print(f"   -> {b_verdict}")

    print(
        f"\n3. [Mixed Pool Filtering Check (D: 1 Rel + 3 Dist vs A: Gold Only)]:\n"
        f"   - Edit Similarity:  D={acc_d:.2f}% vs A={acc_a:.2f}% (Δ(D-A) = {delta_da_edit:+.2f}%)\n"
        f"   - Entity Recall:    D={ent_d:.2f}% vs A={ent_a:.2f}% (Δ(D-A) = {ent_d - ent_a:+.2f}%)\n"
        f"   - Contamination(D): {contam_d:.2f}% of samples leaked distractor chars when mixed 1:3"
    )
    if delta_da_edit < -3.0 or contam_d >= 10.0:
        d_verdict = (
            "⚠️ DILUTION / INTERFERENCE IN MIXED POOL: Mixing 3 distractors with 1 relevant clause (D) "
            "dilutes softmax attention or introduces distractor interference relative to Gold (A)."
        )
        need_top_k_tight = True
    else:
        d_verdict = (
            "✅ STRONG SPAN LOCKING IN MIXED POOL: Left Brain locks onto the relevant clause even when "
            "surrounded by 3 cross-domain distractors (D ≈ A)."
        )
        need_top_k_tight = False
    print(f"   -> {d_verdict}")

    print("\n4. [Architectural Recommendation for Phase 5 `KnowledgeSelector`]:")
    if need_sink and need_top_k_tight:
        print(
            "   🔧 RECOMMENDATION: BOTH (1) a strict `no-copy / sink` gate threshold (pass `k_attractors=None` "
            "when `max(selector_gates) < tau_sink`, falling back to Condition C) AND (2) tight top-1/top-2 "
            "filtering are REQUIRED. Do NOT force-keep top-1 candidate when all selector scores are low."
        )
    elif need_sink:
        print(
            "   🔧 RECOMMENDATION: Add a `no-copy / sink` threshold in `KnowledgeSelector.select_active_clause_attractors` "
            "(do NOT force `keep_clause[:, 0] = True` when all candidate scores < `min_score`; allow empty `k_mask` "
            "so pure chitchat/connective clauses use Condition C without distractor pollution)."
        )
    elif need_top_k_tight:
        print(
            "   🔧 RECOMMENDATION: Left Brain ignores pure distractors well, but mixed pools dilute copy recall. "
            "Keep `top_m_clauses` small (1~2) and prune low-scoring tail candidates aggressively."
        )
    else:
        print(
            "   🔧 RECOMMENDATION: Left Brain's `OrthogonalAttractor` is naturally resilient to both pure and "
            "mixed distractors (`B ≈ C` and `D ≈ A`). `KnowledgeSelector` only needs coarse top-M retrieval."
        )
    print("=" * 126 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Task #031: Diagnose LeftBrainCrystallizer Attractor Robustness (Conditions A / B / C / D)"
    )
    parser.add_argument("--encoder_ckpt", type=str, default=None, help="Path to frozen TinyCharEncoderCUDA student.pt")
    parser.add_argument("--left_brain_ckpt", type=str, default=None, help="Path to frozen LeftBrainCrystallizer unet.pt")
    parser.add_argument("--translator_ckpt", type=str, default=None, help="Path to frozen ContinuousToTextTranslatorCUDA translator.pt")
    parser.add_argument("--ckpt_dir", type=str, default=None, help="Additional checkpoint root directory to search")
    parser.add_argument("--data_path", type=str, default=None, help="Optional path to omni_knowledge_corpus.jsonl")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of additional dataset samples to evaluate (default: 50)")
    parser.add_argument("--max_print_corpus", type=int, default=8, help="Max corpus samples to print in side-by-side table")
    parser.add_argument("--ode_steps", type=int, default=10, help="Euler ODE integration steps for LeftBrainCrystallizer (default: 10)")
    parser.add_argument("--seq_len", type=int, default=32, help="Clause token sequence length (default: 32)")
    parser.add_argument("--num_subbands", type=int, default=32, help="Number of Legendre modes K (default: 32)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for controlled noise x_0 and shuffling")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Running Left-Brain Attractor Robustness Diagnostic on device: {device}")

    search_dirs = build_search_dirs(args.ckpt_dir)
    print(f"📂 Searching checkpoint directories: {search_dirs}")

    tokenizer = CharTokenizer()
    pad_id = tokenizer.pad_token_id

    # 1. Load Phase 1 TinyCharEncoderCUDA
    encoder = TinyCharEncoderCUDA(vocab_size=tokenizer.vocab_size).to(device)
    enc_path = discover_encoder_ckpt(args.encoder_ckpt, search_dirs)
    if enc_path:
        encoder.load_state_dict(torch.load(enc_path, map_location=device, weights_only=True))
        print(f"✅ Loaded frozen TinyCharEncoderCUDA from: {enc_path}")
    else:
        print("⚠️ WARNING: TinyCharEncoderCUDA checkpoint not found! Using random fallback weights (smoke-test mode).")
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # 2. Load HiPPOOracleCUDA & LeftBrainCrystallizer
    hippo_oracle = HiPPOOracleCUDA(hidden_dim=1024, N=64, num_subbands=args.num_subbands).to(device)
    hippo_oracle.eval()

    unet = LeftBrainCrystallizer(
        hidden_dim=1024,
        cond_dim=1024,
        num_layers=6,
        num_subbands=args.num_subbands,
    ).to(device)
    lb_path = discover_left_brain_ckpt(unet, args.left_brain_ckpt, search_dirs, device=device)
    if lb_path:
        print(
            f"✅ Loaded frozen LeftBrainCrystallizer from: {lb_path} "
            f"(calib_steps={int(unet.calib_steps.item())}, mean={unet.manifold_mean.item():.4f}, std={unet.manifold_std.item():.4f})"
        )
    else:
        print("⚠️ WARNING: LeftBrainCrystallizer checkpoint not found! Using random fallback weights (smoke-test mode).")
    unet.eval()
    for p in unet.parameters():
        p.requires_grad = False

    # 3. Load ContinuousToTextTranslatorCUDA
    translator = ContinuousToTextTranslatorCUDA(d_model=1024, vocab_size=tokenizer.vocab_size).to(device)
    trans_path = discover_translator_ckpt(args.translator_ckpt, search_dirs)
    if trans_path:
        translator.load_state_dict(torch.load(trans_path, map_location=device, weights_only=True))
        print(f"✅ Loaded frozen ContinuousToTextTranslatorCUDA from: {trans_path}")
    else:
        print("⚠️ WARNING: Translator checkpoint not found! Using random fallback weights (smoke-test mode).")
    translator.eval()
    for p in translator.parameters():
        p.requires_grad = False

    inspect_unet_attractor_parameters(unet)

    # 4. Assemble Built-in 8-Sample Canonical Suite + Corpus Samples
    all_samples: List[DiagnosticSample] = list(BUILTIN_CANONICAL_SUITE)
    resolved_data_path = discover_data_path(args.data_path)
    if resolved_data_path and args.num_samples > 0:
        print(f"\n📖 Sampling up to {args.num_samples} additional multi-domain clauses from: {resolved_data_path}")
        corpus_samples = sample_corpus_records(
            data_path=resolved_data_path,
            tokenizer=tokenizer,
            num_samples=args.num_samples,
            seq_len=args.seq_len,
            seed=args.seed,
        )
        print(f"✅ Loaded {len(corpus_samples)} corpus evaluation samples.")
        all_samples.extend(corpus_samples)
    else:
        print("\nℹ️ No external corpus loaded (or --num_samples 0); evaluating Built-in 8-Sample Canonical Suite only.")

    # Auto-calibrate manifold if unet checkpoint is uncalibrated (smoke test fallback)
    if int(unet.calib_steps.item()) == 0:
        with torch.no_grad():
            calib_ids = [
                encode_clause_tokens(tokenizer, s.target_clause, seq_len=args.seq_len)
                for s in all_samples[: min(16, len(all_samples))]
            ]
            w_cal = torch.tensor(calib_ids, dtype=torch.long, device=device)
            m_cal = w_cal != pad_id
            x_cal = encode_tokens(encoder, w_cal)
            unet.calibrate_manifold(x_cal, valid_mask=m_cal)
            print(
                f"⚖️ Auto-calibrated uninitialized LeftBrain manifold: "
                f"mean={unet.manifold_mean.item():.4f}, std={unet.manifold_std.item():.4f}"
            )

    cond_keys = ["A", "B", "C", "D"]
    metrics_all: Dict[str, List[Dict[str, float]]] = {k: [] for k in cond_keys}
    metrics_canon: Dict[str, List[Dict[str, float]]] = {k: [] for k in cond_keys}
    metrics_corpus: Dict[str, List[Dict[str, float]]] = {k: [] for k in cond_keys}

    print("\n" + "=" * 126)
    print(f"🔬 SIDE-BY-SIDE QUALITATIVE COMPARISON (ODE Steps = {args.ode_steps}, SeqLen = {args.seq_len})")
    print("=" * 126)

    printed_corpus = 0
    for s_idx, sample in enumerate(all_samples):
        tgt_ids = encode_clause_tokens(tokenizer, sample.target_clause, seq_len=args.seq_len)
        w_tgt = torch.tensor([tgt_ids], dtype=torch.long, device=device)
        valid_mask = w_tgt != pad_id

        with torch.no_grad():
            x_target = encode_tokens(encoder, w_tgt)  # (1, L, 1024)
            x_target_norm = unet.normalize(x_target)
            z_clause = hippo_oracle(
                x_target_norm,
                return_sequence=False,
                valid_mask=valid_mask,
                subspace_proj=unet.subspace_proj,
            )  # (1, 1024)

            ora_ids = translator.decode_greedy(x_target.float())[0].tolist()
            ora_text = decode_clause_clean(tokenizer, ora_ids)

            # Controlled initial Gaussian noise x_0 (identical across A, B, C, D for this sample)
            gen_cpu = torch.Generator(device="cpu").manual_seed(args.seed + s_idx * 1009)
            x_0 = torch.randn(
                (1, args.seq_len, unet.hidden_dim),
                generator=gen_cpu,
                dtype=torch.float32,
            ).to(device=device, dtype=z_clause.dtype)

            # Condition A: Relevant (Gold)
            w_attr_a, k_mask_a = pack_clauses_as_attractor_tokens(
                tokenizer, sample.relevant_clauses, args.seq_len, device
            )
            x_attr_a = unet.normalize(encode_tokens(encoder, w_attr_a))

            # Condition B: Pure Distractor (Irrelevant)
            w_attr_b, k_mask_b = pack_clauses_as_attractor_tokens(
                tokenizer, sample.distractor_clauses, args.seq_len, device
            )
            x_attr_b = unet.normalize(encode_tokens(encoder, w_attr_b))

            # Condition C: Empty (None)
            x_attr_c, k_mask_c = None, None

            # Condition D: Mixed (1 Relevant + 3 Distractors, shuffled deterministically)
            mixed_clauses = [sample.relevant_clauses[0]] + list(sample.distractor_clauses[:3])
            rng_mix = random.Random(args.seed + s_idx)
            rng_mix.shuffle(mixed_clauses)
            w_attr_d, k_mask_d = pack_clauses_as_attractor_tokens(
                tokenizer, mixed_clauses, args.seq_len, device
            )
            x_attr_d = unet.normalize(encode_tokens(encoder, w_attr_d))

            cond_inputs = {
                "A": (x_attr_a, k_mask_a),
                "B": (x_attr_b, k_mask_b),
                "C": (x_attr_c, k_mask_c),
                "D": (x_attr_d, k_mask_d),
            }

            # Crystallize Condition C first so we can measure ||x_gen(cond) - x_gen(C)||_2
            x_gens: Dict[str, torch.Tensor] = {}
            for c_key in ["C", "A", "B", "D"]:
                x_attr_curr, k_mask_curr = cond_inputs[c_key]
                x_gens[c_key] = crystallize_with_fixed_noise(
                    unet=unet,
                    z_clause=z_clause,
                    x_0=x_0,
                    x_attractors_norm=x_attr_curr,
                    k_mask=k_mask_curr,
                    steps=args.ode_steps,
                )

            decoded_texts: Dict[str, str] = {}
            sample_cond_metrics: Dict[str, Dict[str, float]] = {}

            for c_key in cond_keys:
                x_gen = x_gens[c_key]
                x_attr_curr, k_mask_curr = cond_inputs[c_key]
                pred_ids = translator.decode_greedy(x_gen.float())[0].tolist()
                pred_text = decode_clause_clean(tokenizer, pred_ids)
                decoded_texts[c_key] = pred_text

                max_sim, snap_gate = compute_attractor_snap_diagnostics(
                    unet=unet,
                    x_traj=unet.normalize(x_gen),
                    x_attractors_norm=x_attr_curr,
                    k_mask=k_mask_curr,
                    valid_mask=valid_mask,
                )

                m = compute_sample_metrics(
                    x_gen=x_gen,
                    x_gen_empty=x_gens["C"],
                    x_target=x_target,
                    valid_mask=valid_mask,
                    pred_ids=pred_ids,
                    tgt_ids=tgt_ids,
                    pred_text=pred_text,
                    target_text=sample.target_clause,
                    relevant_clauses=sample.relevant_clauses,
                    distractor_clauses=sample.distractor_clauses,
                    max_sim=max_sim,
                    snap_gate=snap_gate,
                )
                sample_cond_metrics[c_key] = m
                metrics_all[c_key].append(m)
                if sample.source == "canonical":
                    metrics_canon[c_key].append(m)
                else:
                    metrics_corpus[c_key].append(m)

        should_print = (sample.source == "canonical") or (printed_corpus < args.max_print_corpus)
        if should_print:
            if sample.source != "canonical":
                printed_corpus += 1
            print(f"\n📌 [{sample.source.upper()} | {sample.domain}] ({sample.sample_id})")
            print(f"   Target Clause      : {sample.target_clause}")
            print(f"   Encoder->Translator: {ora_text}")
            print(f"   Relevant (K+)      : {' || '.join(sample.relevant_clauses)}")
            print(f"   Distractors (K-)   : {' || '.join(sample.distractor_clauses)}")
            for c_key, c_desc in [
                ("A", "A [Relevant Gold] "),
                ("B", "B [Pure Distract] "),
                ("C", "C [Empty / None]  "),
                ("D", "D [Mixed 1+3]     "),
            ]:
                m = sample_cond_metrics[c_key]
                print(
                    f"   -> {c_desc}: {decoded_texts[c_key]:<28} "
                    f"(Cos={m['latent_cos']:.3f} | PosAcc={m['pos_token_acc']:5.1f}% | "
                    f"Edit={m['edit_sim']:5.1f}% | EntRec={m['entity_recall']:5.1f}% | "
                    f"Contam={m['contaminated']:3.0f}% | MaxSim={m['max_sim']:.2f} | Gate={m['snap_gate']:.2f})"
                )

    canon_summary = {k: summarize_condition_metrics(metrics_canon[k]) for k in cond_keys}
    print_summary_table("BUILT-IN 8-SAMPLE CANONICAL SUITE SUMMARY", canon_summary, len(BUILTIN_CANONICAL_SUITE))

    if any(len(metrics_corpus[k]) > 0 for k in cond_keys):
        corpus_summary = {k: summarize_condition_metrics(metrics_corpus[k]) for k in cond_keys}
        print_summary_table("CORPUS MULTI-DOMAIN SUITE SUMMARY", corpus_summary, len(metrics_corpus["A"]))

    overall_summary = {k: summarize_condition_metrics(metrics_all[k]) for k in cond_keys}
    print_summary_table("OVERALL COMBINED SUMMARY", overall_summary, len(all_samples))
    print_diagnostic_verdict(overall_summary)


if __name__ == "__main__":
    main()
