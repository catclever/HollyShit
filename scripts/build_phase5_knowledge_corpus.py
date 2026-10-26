#!/usr/bin/env python3
"""
scripts/build_phase5_knowledge_corpus.py
========================================
Phase 5 Multi-Stream Knowledge Corpus Builder (`omni_knowledge_corpus.jsonl`).

Produces JSONL records with schema:
  {
    "domain": "...",
    "prompt": "...",
    "knowledge": ["...", "..."],
    "target": "..."
  }

Core Architectural Rules:
  1. NO Artificial Type Tags (`【...】`):
     All knowledge streams are kept completely natural (no `【经典原句】` or `【表结构】` prefixes)
     so the Left-Brain Mamba Planner learns to distinguish poetry, code, equations, encyclopedia
     facts, and narrative lore purely from continuous manifold geometry.
  2. Reading-Comprehension-Friendly & Entity-Overlap Allowed ("No Full-Sentence Cheating, MRC & Entities Preserved"):
     `is_non_cheating_knowledge(k, target)` prevents identical/near-identical full-sentence duplicates
     (`len(target) >= 0.85 * len(k)`) while allowing Reading Comprehension passages `k` that contain
     the concise answer clause `target` (`len(target) < 0.85 * len(k)`) as well as short entities,
     table/column names, equations, numbers, and API signatures for `CopyPointerHead` learning.
  3. Pure Grounded Backbone Default (`--domains grounded_instruct,encyclopedia_rag`, `--samples_per_source 60000`):
     Focuses Phase 5 backbone training on 100% grounded machine reading comprehension (`cmrc2018`,
     `dureader_robust`, `squad`) and grounded instruction QA (`Belle`, `Alpaca-GPT4-CN` with `input`),
     leaving ungrounded open-ended webnovel continuation (`literature_lore`) out of the default backbone.
"""

import os
import re
import sys
import json
import random
import argparse
from typing import Iterator, List, Tuple, Dict, Any, Optional, Union

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


PROMPT_CANDIDATE_KEYS = [
    "prompt", "instruction", "query", "user", "human",
    "question", "question_chinese", "question_zh", "text_a"
]
TARGET_CANDIDATE_KEYS = [
    "target", "output", "response", "assistant", "bot",
    "answer", "answer_zh", "chain_of_thought", "completion", "text_b"
]
SEPARATE_KNOWLEDGE_CANDIDATE_KEYS = [
    "input", "context", "schema", "equation", "source", "reference", "passage"
]

_FALLBACK_DISTRACTORS = [
    "地球大气层主要由氮气和氧气组成，其中氮气约占百分之七十八",
    "计算机操作系统负责管理硬件资源并为应用程序提供运行环境",
    "光合作用是绿色植物利用叶绿素吸收光能将二氧化碳转化为有机物的过程",
    "牛顿第一定律指出物体在不受外力作用时总保持静止或匀速直线运动状态",
]

_STRIP_PUNCT_RE = re.compile(r"^[\s,，.。!！?？;；:：、\"'“”‘’()（）\[\]【】]+|[\s,，.。!！?？;；:：、\"'“”‘’()（）\[\]【】]+$")


def is_non_cheating_knowledge(k: str, target: str) -> bool:
    """
    Reading-Comprehension-Friendly & Entity-Overlap Allowed validation:
      - Strips leading/trailing whitespace and punctuation from `k` and `target`.
      - Rejects empty strings or exact verbatim equality (`k_clean == tgt_clean`).
      - When `tgt_clean in k_clean` (e.g., Reading Comprehension where `k` is a longer reference
        passage and `target` is the concise answer clause contained in `k`):
          * Rejects if `len(tgt_clean) >= 0.85 * len(k_clean)` (near-identical full-sentence copy).
          * ALLOWS reference passages `k` that contain the answer `target` as long as `k` carries
            at least >15% additional surrounding context (`len(tgt_clean) < 0.85 * len(k_clean)`).
      - When `k_clean in tgt_clean`:
          * Rejects ONLY if `len(k_clean) >= 24` AND `len(k_clean) >= 0.5 * len(tgt_clean)`
            (i.e. a long sentence verbatim copy).
          * ALLOWS short entities, table/column names, equations, numbers, API signatures,
            and short clauses (`len(k_clean) < 24` or `< 50%` of target) to naturally overlap
            between `knowledge` (K+) and `target` so `CopyPointerHead` receives rich copy supervision!
    """
    if not k or not target or not isinstance(k, str) or not isinstance(target, str):
        return False

    k_clean = _STRIP_PUNCT_RE.sub("", k.strip())
    tgt_clean = _STRIP_PUNCT_RE.sub("", target.strip())

    if not k_clean or not tgt_clean:
        return False

    if k_clean == tgt_clean:
        return False

    if tgt_clean in k_clean:
        if len(tgt_clean) >= 24 and len(tgt_clean) >= 0.85 * len(k_clean):
            return False
        if len(tgt_clean) >= 0.85 * len(k_clean):
            return False

    if k_clean in tgt_clean:
        if len(k_clean) >= 24 and len(k_clean) >= 0.5 * len(tgt_clean):
            return False

    return True


def clean_text(text: Optional[str]) -> str:
    if not text or not isinstance(text, str):
        return ""
    text = text.strip()
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip any leading artificial bracket tags like `【新闻报道】：` or `【表结构】`
    text = re.sub(r"^\s*【[^】]{1,15}】[：:\s]*", "", text).strip()
    return text


def extract_factual_clauses(
    text: str,
    min_len: int = 12,
    max_len: int = 128,
) -> List[str]:
    """
    Splits text into natural factual/descriptive clauses or sentences by `。！？；\\n`
    with character length in [min_len, max_len]. Never adds artificial tags.
    """
    text = clean_text(text)
    if not text:
        return []

    raw_parts = re.split(r"[。！？；\n]+", text)
    clauses: List[str] = []

    for part in raw_parts:
        part = clean_text(part)
        if not part:
            continue
        if min_len <= len(part) <= max_len:
            clauses.append(part)
        elif len(part) > max_len:
            sub_parts = re.split(r"[，,]+", part)
            acc = ""
            for sp in sub_parts:
                sp = sp.strip()
                if not sp:
                    continue
                candidate = f"{acc}，{sp}" if acc else sp
                if len(candidate) <= max_len:
                    acc = candidate
                else:
                    if min_len <= len(acc) <= max_len:
                        clauses.append(acc)
                    if min_len <= len(sp) <= max_len:
                        acc = sp
                    elif len(sp) > max_len:
                        clauses.append(sp[:max_len])
                        acc = ""
                    else:
                        acc = sp
            if min_len <= len(acc) <= max_len:
                clauses.append(acc)

    return clauses


def _ensure_sentence_end_punct(text: str) -> str:
    text = text.strip()
    if not text:
        return text
    if re.search(r"[。！？.!?;；\"'”’）\)]$", text):
        return text
    has_cjk = bool(re.search(r"[\u4e00-\u9fff]", text))
    text = re.sub(r"[,，、:：]+$", "", text).strip()
    return text + ("。" if has_cjk else ".")


def _extract_mrc_grounded_sample(
    context: str,
    question: str,
    raw_ans: str,
    max_clause_len: int = 126,
) -> Optional[Tuple[str, List[str], str]]:
    """
    Extracts a grounded Reading Comprehension sample `(question, k_plus, target)` from
    `(context, question, raw_ans)`:
      1. Cleans `context`, `question`, `raw_ans`; requires `len(question) >= 4`,
         `len(context) >= 25`, `len(raw_ans) >= 2`.
      2. Locates `raw_ans` in `context` (or the sentence in `context` with highest character overlap)
         and extracts the sentence `ans_sent` containing `raw_ans`.
      3. Constructs `target` (complete, natural answer clause/sentence, 10..95 chars):
         - If `12 <= len(raw_ans) <= max_tgt_len`: uses `raw_ans` (with ending punctuation).
         - Else if `10 <= len(ans_sent) <= max_tgt_len`: uses `ans_sent`.
         - Else: extracts the comma sub-clause window around `raw_ans` within `ans_sent` of length `12..90` chars.
      4. Constructs `k_plus` (1 to 2 reference clauses from `context` up to `max_clause_len = 126`):
         - Extracts a wider context window around `raw_ans` / `target` in `context` of length
           `max(len(target) + 20, 70)` up to `max_clause_len` so `len(target) < 0.80 * len(k_window)`
           and `is_non_cheating_knowledge(k_window, target) == True`, plus optionally 1 adjacent clause.
    """
    ctx = clean_text(context)
    q = clean_text(question)
    ans = clean_text(raw_ans)

    if len(q) < 4 or len(ctx) < 25 or len(ans) < 2:
        return None

    eff_ctx_cap = min(max_clause_len, len(ctx))
    max_tgt_len = min(92, max(14, int(0.74 * eff_ctx_cap)))

    # Split context into sentence spans with their character offsets
    sent_matches = list(re.finditer(r"[^。！？.!?\n]+[。！？.!?]*", ctx))
    if not sent_matches:
        return None

    ans_core = _STRIP_PUNCT_RE.sub("", ans)
    ans_pos = ctx.find(ans)
    if ans_pos < 0 and ans_core:
        ans_pos = ctx.find(ans_core)

    best_sent_idx = 0
    if ans_pos >= 0:
        for i, m in enumerate(sent_matches):
            if m.start() <= ans_pos < m.end():
                best_sent_idx = i
                break
    else:
        ans_chars = set(ans_core or ans)
        best_overlap = -1
        for i, m in enumerate(sent_matches):
            s_txt = m.group(0).strip()
            overlap = sum(1 for ch in ans_chars if ch in s_txt)
            if overlap > best_overlap:
                best_overlap = overlap
                best_sent_idx = i
        ans_pos = sent_matches[best_sent_idx].start()

    ans_sent = sent_matches[best_sent_idx].group(0).strip()

    # Construct natural complete target sentence/clause (10..max_tgt_len chars)
    target = ""
    if 12 <= len(ans) <= max_tgt_len:
        target = _ensure_sentence_end_punct(ans)
    elif 10 <= len(ans_sent) <= max_tgt_len:
        target = _ensure_sentence_end_punct(ans_sent)
    else:
        # Extract comma sub-clause window around raw_ans within ans_sent (or raw_ans if raw_ans > max_tgt_len)
        source_for_sub = ans if len(ans) > max_tgt_len else ans_sent
        comma_spans = [
            s.strip()
            for s in re.findall(r"[^，,;；。！？.!?\n]+[，,;；。！？.!?]*", source_for_sub)
            if s.strip()
        ]
        pivot_idx = 0
        if ans_core:
            for idx_c, sp in enumerate(comma_spans):
                if ans_core in sp or sp in ans_core:
                    pivot_idx = idx_c
                    break
        if comma_spans:
            acc = comma_spans[pivot_idx]
            left_i = pivot_idx - 1
            right_i = pivot_idx + 1
            while len(acc) < 18 and (left_i >= 0 or right_i < len(comma_spans)):
                if right_i < len(comma_spans) and len(acc) + len(comma_spans[right_i]) <= max_tgt_len:
                    acc = acc + comma_spans[right_i]
                    right_i += 1
                elif left_i >= 0 and len(comma_spans[left_i]) + len(acc) <= max_tgt_len:
                    acc = comma_spans[left_i] + acc
                    left_i -= 1
                else:
                    break
            if len(acc) > max_tgt_len:
                acc = acc[:max_tgt_len]
            target = _ensure_sentence_end_punct(acc)
        else:
            target = _ensure_sentence_end_punct(ans_sent[:max_tgt_len])

    if len(target) < 10:
        return None

    tgt_clean = _STRIP_PUNCT_RE.sub("", target)
    t_pos = ctx.find(tgt_clean)
    if t_pos < 0:
        t_pos = max(0, ans_pos)
    t_end = min(len(ctx), t_pos + len(tgt_clean))

    # Construct wider context window k_window around target/raw_ans in context
    min_k_len = max(len(tgt_clean) + 20, int(len(tgt_clean) / 0.75) + 6, 70)
    desired_k_len = min(max_clause_len, max(min_k_len, eff_ctx_cap))

    if len(ctx) < min_k_len:
        # If context itself is shorter than min_k_len, trim target so len(target) < 0.78 * len(ctx)
        trim_t = max(10, int(0.75 * len(_STRIP_PUNCT_RE.sub("", ctx))))
        if len(tgt_clean) >= trim_t:
            target = _ensure_sentence_end_punct(tgt_clean[:trim_t])
            tgt_clean = _STRIP_PUNCT_RE.sub("", target)
        desired_k_len = min(max_clause_len, len(ctx))

    span_len = t_end - t_pos
    extra_room = max(0, desired_k_len - span_len)
    win_start = max(0, t_pos - extra_room // 2)
    win_end = min(len(ctx), win_start + desired_k_len)
    win_start = max(0, win_end - desired_k_len)

    k_window = ctx[win_start:win_end].strip()
    if not is_non_cheating_knowledge(k_window, target):
        return None

    k_plus = [k_window]

    # Optionally add 1 adjacent clause from context outside [win_start, win_end]
    adj_candidates: List[str] = []
    if best_sent_idx + 1 < len(sent_matches):
        adj_candidates.append(sent_matches[best_sent_idx + 1].group(0).strip())
    if best_sent_idx - 1 >= 0:
        adj_candidates.append(sent_matches[best_sent_idx - 1].group(0).strip())

    for adj in adj_candidates:
        adj_c = clean_text(adj)[:max_clause_len]
        if (
            len(adj_c) >= 12
            and adj_c != k_window
            and adj_c not in k_window
            and is_non_cheating_knowledge(adj_c, target)
        ):
            k_plus.append(adj_c)
            break

    return q, k_plus[:2], target


def try_split_structured_prompt(
    prompt: str,
    max_know_streams: int = 4,
    max_clause_len: int = 126,
) -> Optional[Tuple[str, List[str]]]:
    """
    Checks if `prompt` contains structured context/schema/input + query/question, such as:
      - `Schema: <ctx>\\nQuery: <question>`
      - `Context: <ctx>\\nQuestion: <question>`
      - Multi-line `instruction\\ninput`
    Returns `(cleaned_prompt_question, natural_knowledge_list)` with no artificial tags.
    """
    if not prompt or "\n" not in prompt:
        return None

    # Pattern 1: SQL Schema + Query
    sql_match = re.match(
        r"^\s*Schema:\s*(.+?)\s*\n+\s*Query:\s*(.+)\s*$",
        prompt,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if sql_match:
        schema_str = clean_text(sql_match.group(1))
        query_str = clean_text(sql_match.group(2))
        if schema_str and query_str:
            schema_parts = [
                p.strip()[:max_clause_len]
                for p in re.split(r"[;\n]+", schema_str)
                if p.strip()
            ]
            if not schema_parts:
                schema_parts = [schema_str[:max_clause_len]]
            return query_str, schema_parts[:max_know_streams]

    # Pattern 2: Explicit Reference/Context + Question/Query markers
    ref_match = re.match(
        r"^\s*(?:Context|Reference|Passage|Background|Input|参考资料|背景知识|背景材料|已知条件|材料)[:：]\s*(.+?)\s*\n+\s*(?:Question|Query|Instruction|问题|提问|指令|请回答)[:：]\s*(.+)\s*$",
        prompt,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if ref_match:
        ref_str = clean_text(ref_match.group(1))
        q_str = clean_text(ref_match.group(2))
        if ref_str and q_str:
            ref_clauses = extract_factual_clauses(ref_str, min_len=8, max_len=max_clause_len)
            if not ref_clauses:
                ref_clauses = [ref_str[:max_clause_len]]
            return q_str, ref_clauses[:max_know_streams]

    # Pattern 3: Multi-line Instruction + Input block
    lines = [clean_text(ln) for ln in prompt.splitlines() if clean_text(ln)]
    if len(lines) >= 2 and len(lines[0]) >= 5:
        rest = "\n".join(lines[1:]).strip()
        if len(rest) >= 15:
            rest_clauses = extract_factual_clauses(rest, min_len=8, max_len=max_clause_len)
            if not rest_clauses:
                rest_clauses = [rest[:max_clause_len]]
            return lines[0], rest_clauses[:max_know_streams]

    return None


def disjoint_split_target_clauses(
    target_text: str,
    max_pos: int = 2,
    max_clause_len: int = 126,
    min_remaining_len: int = 12,
) -> Tuple[List[str], str]:
    """
    Partitions `target_text` into two disjoint clause spans:
      - `k_plus`: 1-2 leading premise/background clauses extracted as positive knowledge (K+)
      - `new_target`: the remaining continuation clauses with `k_plus` removed.
    Uses `is_non_cheating_knowledge(k, rem_text)` so shared entities/numbers are allowed.
    """
    text = clean_text(target_text)
    if not text:
        return [], text

    # 1. Sentence-level disjoint split
    sentence_spans = [
        s.strip()
        for s in re.findall(r"[^。！？；\n]+[。！？；\n]*", text)
        if s.strip()
    ]
    if len(sentence_spans) >= 2:
        num_take = 1
        if len(sentence_spans) >= 4 and max_pos >= 2:
            rem_if_2 = "".join(sentence_spans[2:]).strip()
            if len(rem_if_2) >= min_remaining_len:
                num_take = 2

        rem_text = "".join(sentence_spans[num_take:]).strip()
        if len(rem_text) >= min_remaining_len:
            raw_k = [
                clean_text(re.sub(r"[。！？；\n]+$", "", s))[:max_clause_len]
                for s in sentence_spans[:num_take]
            ]
            k_plus = [
                k for k in raw_k
                if k and is_non_cheating_knowledge(k, rem_text)
            ]
            if k_plus:
                return k_plus, rem_text

    # 2. Comma/semicolon-level disjoint split for single-sentence targets
    comma_spans = [
        s.strip()
        for s in re.findall(r"[^，,;；]+[,，;；]*", text)
        if s.strip()
    ]
    if len(comma_spans) >= 2:
        num_take = 1
        if len(comma_spans) >= 4 and max_pos >= 2:
            rem_if_2 = "".join(comma_spans[2:]).strip()
            if len(rem_if_2) >= min_remaining_len:
                num_take = 2

        rem_text = "".join(comma_spans[num_take:]).strip()
        if len(rem_text) >= min_remaining_len:
            raw_k = [
                clean_text(re.sub(r"[,，;；。！？\n]+$", "", s))[:max_clause_len]
                for s in comma_spans[:num_take]
            ]
            k_plus = [
                k for k in raw_k
                if len(k) >= 6 and is_non_cheating_knowledge(k, rem_text)
            ]
            if k_plus:
                return k_plus, rem_text

    return [], text


# =========================================================================
# Direct HuggingFace Channel-1 Grounded Streamers
# =========================================================================

def stream_domain1_grounded_instruct(
    max_samples: int, max_clause_len: int = 126
) -> Iterator[Dict[str, Any]]:
    """
    Domain 1 (`grounded_instruct`):
    Sources: `BelleGroup/train_1M_CN` & `silk-road/alpaca-data-gpt4-chinese`
    Filters records where `input` is non-empty (>= 15 chars):
      prompt = instruction, K+ = [input], target = output (entity overlap allowed).
    """
    from datasets import load_dataset
    sources = ["BelleGroup/train_1M_CN", "silk-road/alpaca-data-gpt4-chinese"]
    per_src = max(1, max_samples // len(sources))
    for sname in sources:
        print(f">>> [1/6 grounded_instruct] Streaming {sname}...")
        count = 0
        try:
            ds = load_dataset(sname, split="train", streaming=True)
            for item in ds:
                instr = clean_text(item.get("instruction", ""))
                inp = clean_text(item.get("input", ""))
                out = clean_text(item.get("output", ""))
                if len(out) < 15 or len(instr) < 4:
                    continue
                if len(inp) >= 15:
                    k_clauses = extract_factual_clauses(inp, min_len=8, max_len=max_clause_len)
                    if not k_clauses:
                        k_clauses = [inp[:max_clause_len]]
                    eff_prompt = instr
                else:
                    split_res = try_split_structured_prompt(instr, max_know_streams=2, max_clause_len=max_clause_len)
                    if split_res is None:
                        continue
                    eff_prompt, k_clauses = split_res
                k_plus = [k for k in k_clauses[:2] if is_non_cheating_knowledge(k, out)]
                if k_plus:
                    yield {
                        "domain": "grounded_instruct",
                        "prompt": eff_prompt,
                        "k_plus": k_plus,
                        "target": out[:350],
                    }
                    count += 1
                    if count >= per_src:
                        break
        except Exception as e:
            print(f"⚠️ Warning during {sname} stream: {e}")
        print(f"  -> Extracted {count} samples from {sname}")


def stream_domain2_code_api(
    max_samples: int, max_clause_len: int = 126
) -> Iterator[Dict[str, Any]]:
    """
    Domain 2 (`code_api`):
    Sources: `b-mc2/sql-create-context` & `sahil2801/CodeAlpaca-20k` / `theblackcat102/evol-codealpaca-v1`
    """
    from datasets import load_dataset
    sql_target = max(1, max_samples // 2)
    code_target = max(1, max_samples - sql_target)

    print(">>> [2/6 code_api] Streaming b-mc2/sql-create-context...")
    sql_count = 0
    try:
        ds = load_dataset("b-mc2/sql-create-context", split="train", streaming=True)
        for item in ds:
            q = clean_text(item.get("question", ""))
            ctx = clean_text(item.get("context", ""))
            ans = clean_text(item.get("answer", ""))
            if len(q) >= 5 and len(ctx) >= 10 and len(ans) >= 8:
                parts = [p.strip()[:max_clause_len] for p in re.split(r"[;\n]+", ctx) if p.strip()]
                k_plus = [p for p in (parts or [ctx[:max_clause_len]])[:2] if is_non_cheating_knowledge(p, ans)]
                if k_plus:
                    yield {
                        "domain": "code_api",
                        "prompt": q,
                        "k_plus": k_plus,
                        "target": ans[:350],
                    }
                    sql_count += 1
                    if sql_count >= sql_target:
                        break
    except Exception as e:
        print(f"⚠️ Warning during b-mc2/sql-create-context stream: {e}")
    print(f"  -> Extracted {sql_count} SQL samples")

    code_count = 0
    for sname in ["sahil2801/CodeAlpaca-20k", "theblackcat102/evol-codealpaca-v1"]:
        if code_count >= code_target:
            break
        print(f">>> [2/6 code_api] Streaming {sname}...")
        try:
            ds = load_dataset(sname, split="train", streaming=True)
            for item in ds:
                instr = clean_text(item.get("instruction", ""))
                inp = clean_text(item.get("input", ""))
                out = clean_text(item.get("output", ""))
                if len(instr) < 5 or len(out) < 15:
                    continue
                if len(inp) >= 4 and is_non_cheating_knowledge(inp, out):
                    k_plus = [inp[:max_clause_len]]
                    target_code = out[:350]
                else:
                    lines = [ln for ln in out.splitlines() if ln.strip()]
                    if len(lines) < 2:
                        continue
                    sig_line = lines[0].strip()[:max_clause_len]
                    body_code = "\n".join(lines[1:]).strip()[:350]
                    if len(sig_line) < 6 or len(body_code) < 10 or not is_non_cheating_knowledge(sig_line, body_code):
                        continue
                    k_plus = [sig_line]
                    target_code = body_code
                yield {
                    "domain": "code_api",
                    "prompt": instr,
                    "k_plus": k_plus,
                    "target": target_code,
                }
                code_count += 1
                if code_count >= code_target:
                    break
        except Exception as e:
            print(f"⚠️ Warning during {sname} stream: {e}")
    print(f"  -> Extracted {code_count} CodeAlpaca samples")


def stream_domain3_classical_quote(
    max_samples: int, max_clause_len: int = 126
) -> Iterator[Dict[str, Any]]:
    """
    Domain 3 (`classical_quote`):
    Sources: `gujilab/chinese-classical-corpus` & `Lifan-Z/Chinese-poetries-txt` / `erhwenkuo/poetry-chinese-zhtw`
    """
    from datasets import load_dataset
    half = max(1, max_samples // 2)

    print(">>> [3/6 classical_quote] Streaming gujilab/chinese-classical-corpus...")
    c_count = 0
    try:
        ds = load_dataset("gujilab/chinese-classical-corpus", "translate", split="train", streaming=True)
        for item in ds:
            instr = clean_text(item.get("instruction", "")) or "请阐释并译述该古文名句："
            src = clean_text(item.get("source", ""))
            tgt = clean_text(item.get("target", ""))
            if len(src) >= 6 and len(tgt) >= 10 and is_non_cheating_knowledge(src, tgt):
                yield {
                    "domain": "classical_quote",
                    "prompt": instr,
                    "k_plus": [src[:max_clause_len]],
                    "target": tgt[:300],
                }
                c_count += 1
                if c_count >= half:
                    break
    except Exception as e:
        print(f"⚠️ Warning during gujilab/chinese-classical-corpus stream: {e}")
    print(f"  -> Extracted {c_count} classical translation samples")

    p_count = 0
    p_target = max(1, max_samples - c_count)
    for cname in ["Lifan-Z/Chinese-poetries-txt", "erhwenkuo/poetry-chinese-zhtw"]:
        if p_count >= p_target:
            break
        print(f">>> [3/6 classical_quote] Streaming {cname}...")
        try:
            ds = load_dataset(cname, split="train", streaming=True)
            for item in ds:
                title = clean_text(item.get("title", ""))
                author = clean_text(item.get("author", ""))
                content = item.get("content", item.get("text", item.get("paragraphs", "")))
                if isinstance(content, list):
                    content = "。".join([clean_text(str(x)) for x in content if clean_text(str(x))])
                content = clean_text(str(content))
                if not content or len(content) < 20:
                    continue
                couplets = [s.strip() for s in re.findall(r"[^。！？；\n]+[。！？；\n]*", content) if s.strip()]
                if len(couplets) < 2:
                    continue
                opening = re.sub(r"[。！？；\n]+$", "", couplets[0]).strip()[:max_clause_len]
                continuation = "".join(couplets[1:]).strip()[:300]
                if len(opening) < 6 or len(continuation) < 10 or not is_non_cheating_knowledge(opening, continuation):
                    continue
                prompt_str = f"{title}（{author}）" if (title and author) else (title or "古诗文续写与赏析")
                yield {
                    "domain": "classical_quote",
                    "prompt": prompt_str,
                    "k_plus": [opening],
                    "target": continuation,
                }
                p_count += 1
                if p_count >= p_target:
                    break
        except Exception as e:
            print(f"⚠️ Warning during {cname} stream: {e}")
    print(f"  -> Extracted {p_count} classical poetry samples")


def _extract_raw_answer_from_mrc_item(item: Dict[str, Any]) -> str:
    answers = item.get("answers", item.get("answer", item.get("output", "")))
    if isinstance(answers, dict):
        text_list = answers.get("text", [])
        if isinstance(text_list, list) and text_list:
            return str(text_list[0])
        if isinstance(text_list, str):
            return text_list
    elif isinstance(answers, list) and answers:
        first = answers[0]
        if isinstance(first, dict):
            return str(first.get("text", first.get("answer", "")))
        return str(first)
    elif isinstance(answers, str):
        return answers
    return ""


def stream_domain4_encyclopedia_rag(
    max_samples: int, max_clause_len: int = 126
) -> Iterator[Dict[str, Any]]:
    """
    Domain 4 (`encyclopedia_rag` — Authentic Machine Reading Comprehension):
    Replaces legacy ungrounded Wikipedia/Zhihu disjoint splitting with 100% grounded MRC datasets:
      - Chinese MRC (80% of `max_samples`): `hfl/cmrc2018`, `cmrc2018`, `clue/cmrc2018`,
        `PaddlePaddle/dureader_robust`, `dureader_robust`, `Simulate/cmrc2018`, plus
        `BelleGroup/train_1M_CN` reading-comprehension records as fallback.
      - English MRC (20% of `max_samples`): `rajpurkar/squad`.
    """
    from datasets import load_dataset

    zh_target = max(1, int(0.80 * max_samples))
    en_target = max(1, max_samples - zh_target)

    zh_count = 0
    zh_mrc_sources: List[Tuple[str, Optional[str]]] = [
        ("hfl/cmrc2018", None),
        ("cmrc2018", None),
        ("clue", "cmrc2018"),
        ("PaddlePaddle/dureader_robust", None),
        ("dureader_robust", None),
        ("Simulate/cmrc2018", None),
    ]

    for ds_name, ds_config in zh_mrc_sources:
        if zh_count >= zh_target:
            break
        label_str = f"{ds_name} ({ds_config})" if ds_config else ds_name
        print(f">>> [4/6 encyclopedia_rag] Streaming Chinese MRC {label_str}...")
        src_extracted = 0
        try:
            if ds_config:
                ds = load_dataset(ds_name, ds_config, split="train", streaming=True)
            else:
                ds = load_dataset(ds_name, split="train", streaming=True)
            for item in ds:
                ctx = str(item.get("context", item.get("passage", item.get("input", ""))))
                q = str(item.get("question", item.get("query", item.get("instruction", ""))))
                raw_ans = _extract_raw_answer_from_mrc_item(item)
                res = _extract_mrc_grounded_sample(ctx, q, raw_ans, max_clause_len=max_clause_len)
                if res is not None:
                    q_out, k_plus, tgt_out = res
                    yield {
                        "domain": "encyclopedia_rag",
                        "prompt": q_out,
                        "k_plus": k_plus,
                        "target": tgt_out,
                    }
                    zh_count += 1
                    src_extracted += 1
                    if zh_count >= zh_target:
                        break
        except Exception as e:
            print(f"⚠️ Warning during {label_str} stream: {e}")
        print(f"  -> Extracted {src_extracted} Chinese MRC samples from {label_str}")

    if zh_count < zh_target:
        print(">>> [4/6 encyclopedia_rag] Streaming BelleGroup/train_1M_CN reading-comprehension fallback...")
        fb_extracted = 0
        try:
            ds = load_dataset("BelleGroup/train_1M_CN", split="train", streaming=True)
            for item in ds:
                instr = clean_text(item.get("instruction", ""))
                inp = clean_text(item.get("input", ""))
                out = clean_text(item.get("output", ""))
                if not (4 <= len(out) <= 140 and len(instr) >= 4):
                    continue
                if len(inp) < 40:
                    split_res = try_split_structured_prompt(instr, max_know_streams=2, max_clause_len=max_clause_len)
                    if split_res is not None:
                        instr, k_clauses_fb = split_res
                        inp = "。".join(k_clauses_fb)
                if len(inp) >= 40 and len(instr) >= 4:
                    res = _extract_mrc_grounded_sample(inp, instr, out, max_clause_len=max_clause_len)
                    if res is not None:
                        q_out, k_plus, tgt_out = res
                        yield {
                            "domain": "encyclopedia_rag",
                            "prompt": q_out,
                            "k_plus": k_plus,
                            "target": tgt_out,
                        }
                        zh_count += 1
                        fb_extracted += 1
                        if zh_count >= zh_target:
                            break
        except Exception as e:
            print(f"⚠️ Warning during BelleGroup/train_1M_CN MRC fallback stream: {e}")
        print(f"  -> Extracted {fb_extracted} Chinese MRC fallback samples")

    en_count = 0
    print(">>> [4/6 encyclopedia_rag] Streaming English MRC rajpurkar/squad...")
    try:
        ds = load_dataset("rajpurkar/squad", split="train", streaming=True)
        for item in ds:
            ctx = str(item.get("context", ""))
            q = str(item.get("question", ""))
            raw_ans = _extract_raw_answer_from_mrc_item(item)
            res = _extract_mrc_grounded_sample(ctx, q, raw_ans, max_clause_len=max_clause_len)
            if res is not None:
                q_out, k_plus, tgt_out = res
                yield {
                    "domain": "encyclopedia_rag",
                    "prompt": q_out,
                    "k_plus": k_plus,
                    "target": tgt_out,
                }
                en_count += 1
                if en_count >= en_target:
                    break
    except Exception as e:
        print(f"⚠️ Warning during rajpurkar/squad stream: {e}")
    print(f"  -> Extracted {en_count} SQuAD MRC samples (Total encyclopedia_rag: {zh_count + en_count})")


def stream_domain5_math_rule(
    max_samples: int, max_clause_len: int = 126
) -> Iterator[Dict[str, Any]]:
    """
    Domain 5 (`math_rule`):
    Source: `MU-NLPC/calc-ape210k`
    """
    from datasets import load_dataset
    print(">>> [5/6 math_rule] Streaming MU-NLPC/calc-ape210k...")
    count = 0
    try:
        ds = load_dataset("MU-NLPC/calc-ape210k", split="train", streaming=True)
        for item in ds:
            q = clean_text(item.get("question_chinese", item.get("question", "")))
            eq = clean_text(item.get("equation", ""))
            cot = clean_text(item.get("chain_of_thought", ""))
            res = clean_text(str(item.get("result", item.get("answer", ""))))
            if len(q) < 5 or not eq:
                continue
            if cot and len(cot) >= 15:
                intact_cot = cot.strip(" ，,。；;")
                ans = f"{intact_cot}，最终求解结果为{res}。" if res else f"{intact_cot}。"
            elif res:
                ans = f"依题设等量关系列式 {eq} 逐步推导，可计算得出最终结果为{res}。"
            else:
                continue
            if len(ans) >= 15 and is_non_cheating_knowledge(eq, ans):
                yield {
                    "domain": "math_rule",
                    "prompt": q,
                    "k_plus": [eq[:max_clause_len]],
                    "target": ans[:350],
                }
                count += 1
                if count >= max_samples:
                    break
    except Exception as e:
        print(f"⚠️ Warning during MU-NLPC/calc-ape210k stream: {e}")
    print(f"  -> Extracted {count} math_rule samples")


def stream_domain6_literature_lore(
    max_samples: int, max_clause_len: int = 126
) -> Iterator[Dict[str, Any]]:
    """
    Domain 6 (`literature_lore`):
    Sources: `wdndev/webnovel-chinese` / `jetaudio/chinese_web_novels`
    """
    from datasets import load_dataset
    count = 0
    for cname in ["wdndev/webnovel-chinese", "jetaudio/chinese_web_novels"]:
        if count >= max_samples:
            break
        print(f">>> [6/6 literature_lore] Streaming {cname}...")
        try:
            ds = load_dataset(cname, split="train", streaming=True)
            for item in ds:
                text = clean_text(item.get("text", item.get("content", "")))
                if not text or len(text) < 120:
                    continue
                sents = [
                    s.strip()
                    for s in re.findall(r"[^。！？；\n]+[。！？；\n]*", text[:1200])
                    if len(s.strip()) >= 10
                ]
                if len(sents) < 4:
                    continue
                part1_lore = re.sub(r"[。！？；\n]+$", "", sents[0]).strip()[:max_clause_len]
                prompt_end_idx = min(5, max(3, len(sents) - 2))
                part2_prompt = "".join(sents[1:prompt_end_idx]).strip()[:380]
                part3_target = "".join(sents[prompt_end_idx:]).strip()[:350]
                if (
                    len(part1_lore) >= 10
                    and len(part2_prompt) >= 20
                    and len(part3_target) >= 20
                    and is_non_cheating_knowledge(part1_lore, part3_target)
                ):
                    yield {
                        "domain": "literature_lore",
                        "prompt": part2_prompt,
                        "k_plus": [part1_lore],
                        "target": part3_target,
                    }
                    count += 1
                    if count >= max_samples:
                        break
        except Exception as e:
            print(f"⚠️ Warning during {cname} stream: {e}")
    print(f"  -> Extracted {count} literature_lore samples")


def stream_hf_6domain_knowledge_corpus(
    output_path: str = "/content/omni_knowledge_corpus.jsonl",
    samples_per_source: int = 60000,
    max_know_streams: int = 4,
    know_seq_len: int = 128,
    empty_know_prob: float = 0.15,
    seed: int = 42,
    shuffle_buffer_size: int = 25000,
    domains: Optional[Union[List[str], str]] = "grounded_instruct,encyclopedia_rag",
) -> Dict[str, Any]:
    """
    Streams directly from HuggingFace across selected Channel-1 domains (defaulting to the
    pure grounded backbone: `grounded_instruct` + `encyclopedia_rag`), mixing 1-2 same-domain
    distractors K- from a rolling domain buffer and writing shuffled multi-stream JSONL records.
    """
    rng = random.Random(seed)
    max_clause_len = max(12, know_seq_len - 2)
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    all_domain_streams = [
        ("grounded_instruct", "1. grounded_instruct", lambda: stream_domain1_grounded_instruct(samples_per_source, max_clause_len)),
        ("code_api", "2. code_api", lambda: stream_domain2_code_api(samples_per_source, max_clause_len)),
        ("classical_quote", "3. classical_quote", lambda: stream_domain3_classical_quote(samples_per_source, max_clause_len)),
        ("encyclopedia_rag", "4. encyclopedia_rag", lambda: stream_domain4_encyclopedia_rag(samples_per_source, max_clause_len)),
        ("math_rule", "5. math_rule", lambda: stream_domain5_math_rule(samples_per_source, max_clause_len)),
        ("literature_lore", "6. literature_lore", lambda: stream_domain6_literature_lore(samples_per_source, max_clause_len)),
    ]

    if domains is None:
        selected_domains = {"grounded_instruct", "encyclopedia_rag"}
    elif isinstance(domains, str):
        parts = [d.strip() for d in domains.split(",") if d.strip()]
        if not parts or "all" in [p.lower() for p in parts]:
            selected_domains = {d[0] for d in all_domain_streams}
        else:
            selected_domains = set(parts)
    else:
        parts = [str(d).strip() for d in domains if str(d).strip()]
        if not parts or "all" in [p.lower() for p in parts]:
            selected_domains = {d[0] for d in all_domain_streams}
        else:
            selected_domains = set(parts)

    domain_streams = [
        (label, fn)
        for dom_key, label, fn in all_domain_streams
        if dom_key in selected_domains
    ]

    rolling_domain_pool: Dict[str, List[str]] = {}
    write_buffer: List[Dict[str, Any]] = []
    total_written = 0
    empty_count = 0
    active_count = 0

    with open(output_path, "w", encoding="utf-8") as out_f:
        for label, stream_fn in domain_streams:
            print(f"\n🚀 Executing HF Stream: {label} ...")
            for raw_item in stream_fn():
                domain = raw_item["domain"]
                prompt = raw_item["prompt"]
                target = raw_item["target"]
                k_plus = [
                    clean_text(k)[:max_clause_len]
                    for k in raw_item.get("k_plus", [])
                    if clean_text(k) and is_non_cheating_knowledge(clean_text(k), target)
                ]

                dom_pool = rolling_domain_pool.setdefault(domain, [])

                if rng.random() < empty_know_prob or not k_plus:
                    k_streams: List[str] = []
                    empty_count += 1
                else:
                    max_pos = max(1, min(2, max_know_streams - 1 if max_know_streams > 1 else 1))
                    k_pos = k_plus[:max_pos]
                    rem = max_know_streams - len(k_pos)
                    num_neg = min(rng.randint(1, 2), rem) if rem > 0 else 0
                    k_minus: List[str] = []
                    if num_neg > 0 and dom_pool:
                        attempts = 0
                        while len(k_minus) < num_neg and attempts < 20:
                            attempts += 1
                            cand = rng.choice(dom_pool)
                            if (
                                cand not in k_pos
                                and cand not in k_minus
                                and cand not in target
                                and target not in cand
                            ):
                                k_minus.append(cand)
                    if len(k_minus) < num_neg:
                        for fb in _FALLBACK_DISTRACTORS:
                            if len(k_minus) >= num_neg:
                                break
                            fb_s = fb[:max_clause_len]
                            if fb_s not in k_pos and fb_s not in target and target not in fb_s:
                                k_minus.append(fb_s)
                    k_streams = (k_pos + k_minus)[:max_know_streams]
                    rng.shuffle(k_streams)
                    active_count += 1

                # Update rolling domain pool with K+ clauses
                for kp in k_plus:
                    if len(kp) >= 8:
                        dom_pool.append(kp)
                if len(dom_pool) > 10000:
                    del dom_pool[:2000]

                write_buffer.append({
                    "domain": domain,
                    "prompt": prompt,
                    "knowledge": k_streams,
                    "target": target,
                })

                if len(write_buffer) >= shuffle_buffer_size:
                    rng.shuffle(write_buffer)
                    for b_item in write_buffer:
                        out_f.write(json.dumps(b_item, ensure_ascii=False) + "\n")
                    total_written += len(write_buffer)
                    write_buffer.clear()
                    out_f.flush()

        if write_buffer:
            rng.shuffle(write_buffer)
            for b_item in write_buffer:
                out_f.write(json.dumps(b_item, ensure_ascii=False) + "\n")
            total_written += len(write_buffer)
            write_buffer.clear()
            out_f.flush()

    return {
        "total_samples": total_written,
        "active_knowledge_count": active_count,
        "empty_knowledge_count": empty_count,
        "domains": list(rolling_domain_pool.keys()),
        "output_path": output_path,
    }


# =========================================================================
# Local File Converter
# =========================================================================

def _extract_raw_record(item: Dict[str, Any], max_clause_len: int = 126) -> Optional[Dict[str, Any]]:
    if not isinstance(item, dict):
        return None

    domain = str(item.get("domain", "general") or "general").strip()

    prompt_key_used = None
    prompt_str = None
    for k in PROMPT_CANDIDATE_KEYS + ["input", "context", "source"]:
        if k in item and item[k]:
            prompt_str = clean_text(str(item[k]))
            prompt_key_used = k
            break

    target_str = None
    for k in TARGET_CANDIDATE_KEYS:
        if k in item and item[k]:
            target_str = clean_text(str(item[k]))
            break

    if not (prompt_str and target_str):
        for conv_key in ["messages", "conversations"]:
            if conv_key in item and isinstance(item[conv_key], list):
                user_text, assistant_text = "", ""
                for turn in item[conv_key]:
                    if not isinstance(turn, dict):
                        continue
                    role = turn.get("role", turn.get("from", "")).lower()
                    content = turn.get("content", turn.get("value", ""))
                    if role in ["user", "human"]:
                        user_text = content
                    elif role in ["assistant", "bot", "gpt", "model"]:
                        assistant_text = content
                if user_text and assistant_text:
                    prompt_str = clean_text(str(user_text))
                    target_str = clean_text(str(assistant_text))
                    break

    if not prompt_str or not target_str:
        return None

    # Check explicit knowledge or separate multi-field knowledge (Case A)
    parsed_know: Optional[List[str]] = None
    existing_know = item.get("knowledge", None)
    if isinstance(existing_know, list) and len(existing_know) > 0:
        cleaned = [
            clean_text(str(x))[:max_clause_len]
            for x in existing_know
            if x is not None and clean_text(str(x))
        ]
        cleaned = [c for c in cleaned if is_non_cheating_knowledge(c, target_str)]
        if cleaned:
            parsed_know = cleaned
    elif isinstance(existing_know, str) and clean_text(existing_know):
        s = clean_text(existing_know)[:max_clause_len]
        if is_non_cheating_knowledge(s, target_str):
            parsed_know = [s]

    if parsed_know is None:
        for cand_k in SEPARATE_KNOWLEDGE_CANDIDATE_KEYS:
            if cand_k == prompt_key_used:
                continue
            val = item.get(cand_k)
            if not isinstance(val, str):
                continue
            val_s = val.strip()
            if cand_k == "source" and re.match(r"^[a-zA-Z0-9_./-]+$", val_s):
                continue
            if len(clean_text(val_s)) >= 6:
                s = clean_text(val_s)
                if is_non_cheating_knowledge(s, target_str):
                    clauses = extract_factual_clauses(s, min_len=6, max_len=max_clause_len)
                    if not clauses:
                        clauses = [s[:max_clause_len]]
                    clauses = [c for c in clauses if is_non_cheating_knowledge(c, target_str)]
                    if clauses:
                        parsed_know = clauses
                        break

    return {
        "domain": domain,
        "prompt": prompt_str,
        "target": target_str,
        "existing_knowledge": parsed_know,
    }


def build_phase5_knowledge_corpus(
    input_path: str,
    output_path: str = "/content/omni_knowledge_corpus.jsonl",
    max_samples: Optional[int] = None,
    max_know_streams: int = 4,
    know_seq_len: int = 128,
    empty_know_prob: float = 0.25,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Reads local `input_path` JSONL, groups factual clauses by domain, and writes out
    `output_path` JSONL with multi-stream `knowledge` arrays using `is_non_cheating_knowledge`.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input JSONL file not found: {input_path}")

    rng = random.Random(seed)
    max_clause_len = max(12, know_seq_len - 2)

    raw_records: List[Dict[str, Any]] = []
    global_clause_pool: List[Tuple[int, str]] = []
    domain_clause_pools: Dict[str, List[Tuple[int, str]]] = {}

    if verbose:
        print(f"📖 Reading input corpus from {input_path} ...")

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if max_samples is not None and len(raw_records) >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                rec = _extract_raw_record(obj, max_clause_len=max_clause_len)
                if rec is not None:
                    rec_idx = len(raw_records)
                    raw_records.append(rec)
                    if rec["existing_knowledge"]:
                        for kc in rec["existing_knowledge"]:
                            if len(kc) >= 6:
                                entry = (rec_idx, kc)
                                global_clause_pool.append(entry)
                                domain_clause_pools.setdefault(rec["domain"], []).append(entry)
                    clauses = extract_factual_clauses(
                        rec["target"], min_len=12, max_len=max_clause_len
                    )
                    for c in clauses:
                        entry = (rec_idx, c)
                        global_clause_pool.append(entry)
                        domain_clause_pools.setdefault(rec["domain"], []).append(entry)
            except Exception:
                continue

    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    count_structured = 0
    count_synthesized = 0
    count_empty = 0

    if verbose:
        print(
            f"🔧 Building Phase 5 multi-stream knowledge corpus for {len(raw_records)} samples "
            f"({len(global_clause_pool)} clauses across {len(domain_clause_pools)} domains)..."
        )

    with open(output_path, "w", encoding="utf-8") as out_f:
        for idx, rec in enumerate(raw_records):
            domain = rec["domain"]
            prompt = rec["prompt"]
            target = rec["target"]
            existing_know = rec["existing_knowledge"]

            max_pos_allowed = max(1, min(2, max_know_streams - 1 if max_know_streams > 1 else 1))
            structured_split = try_split_structured_prompt(
                prompt, max_know_streams=max_pos_allowed, max_clause_len=max_clause_len
            )

            # 1. Case A: Multi-field or Structured Context/Schema in Prompt
            if existing_know:
                k_plus = [
                    k for k in existing_know[:max_pos_allowed]
                    if k and is_non_cheating_knowledge(k, target)
                ]
                rem = max_know_streams - len(k_plus)
                k_minus: List[str] = []
                if rem > 0:
                    forbidden = set(k_plus) | {target.strip()}
                    dom_pool = domain_clause_pools.get(domain, [])
                    for pool in (dom_pool, global_clause_pool):
                        if not pool or len(k_minus) >= min(2, rem):
                            continue
                        attempts = 0
                        while len(k_minus) < min(2, rem) and attempts < 20:
                            attempts += 1
                            s_i, cand = rng.choice(pool)
                            if (
                                s_i != idx
                                and cand not in forbidden
                                and cand not in target
                                and target not in cand
                                and cand not in k_minus
                            ):
                                k_minus.append(cand)
                k_streams = (k_plus + k_minus)[:max_know_streams]
                rng.shuffle(k_streams)
                count_structured += 1

            elif structured_split is not None:
                new_prompt, ref_streams = structured_split
                prompt = new_prompt
                k_plus = [
                    k for k in ref_streams[:max_pos_allowed]
                    if k and is_non_cheating_knowledge(k, target)
                ]
                rem = max_know_streams - len(k_plus)
                k_minus = []
                if rem > 0:
                    forbidden = set(k_plus) | {target.strip()}
                    dom_pool = domain_clause_pools.get(domain, [])
                    for pool in (dom_pool, global_clause_pool):
                        if not pool or len(k_minus) >= min(1, rem):
                            continue
                        attempts = 0
                        while len(k_minus) < min(1, rem) and attempts < 20:
                            attempts += 1
                            s_i, cand = rng.choice(pool)
                            if (
                                s_i != idx
                                and cand not in forbidden
                                and cand not in target
                                and target not in cand
                                and cand not in k_minus
                            ):
                                k_minus.append(cand)
                k_streams = (k_plus + k_minus)[:max_know_streams]
                rng.shuffle(k_streams)
                count_structured += 1

            # 2. Empty Knowledge Stream (~25% of records)
            elif rng.random() < empty_know_prob:
                k_streams = []
                count_empty += 1

            # 3. Case B: Disjoint Clause Split (K+ removed from target + K- same-domain distractors)
            else:
                num_pos = min(rng.randint(1, 2), max_pos_allowed)
                k_plus, new_target = disjoint_split_target_clauses(
                    target,
                    max_pos=num_pos,
                    max_clause_len=max_clause_len,
                    min_remaining_len=12,
                )
                orig_target = target
                target = new_target

                remaining_slots = max_know_streams - len(k_plus)
                num_neg = min(rng.randint(1, 2), remaining_slots) if remaining_slots > 0 else 0

                k_minus = []
                if num_neg > 0:
                    forbidden = set(k_plus) | {target.strip(), orig_target.strip()}
                    dom_pool = domain_clause_pools.get(domain, [])
                    for pool in (dom_pool, global_clause_pool):
                        if not pool or len(k_minus) >= num_neg:
                            continue
                        attempts = 0
                        while len(k_minus) < num_neg and attempts < 24:
                            attempts += 1
                            s_i, cand = rng.choice(pool)
                            if (
                                s_i != idx
                                and cand not in forbidden
                                and cand not in target
                                and target not in cand
                                and cand not in orig_target
                                and cand not in k_minus
                            ):
                                k_minus.append(cand)
                    if len(k_minus) < num_neg:
                        for fb in _FALLBACK_DISTRACTORS:
                            if len(k_minus) >= num_neg:
                                break
                            fb_s = fb[:max_clause_len]
                            if (
                                fb_s not in forbidden
                                and fb_s not in target
                                and target not in fb_s
                                and fb_s not in k_minus
                            ):
                                k_minus.append(fb_s)

                k_streams = [
                    k for k in (k_plus + k_minus)[:max_know_streams]
                    if k and is_non_cheating_knowledge(k, target)
                ]
                rng.shuffle(k_streams)
                count_synthesized += 1

            out_record = {
                "domain": domain,
                "prompt": prompt,
                "knowledge": k_streams,
                "target": target,
            }
            out_f.write(json.dumps(out_record, ensure_ascii=False) + "\n")

    stats = {
        "total_samples": len(raw_records),
        "structured_count": count_structured,
        "synthesized_k_plus_minus_count": count_synthesized,
        "empty_knowledge_count": count_empty,
        "domains": list(domain_clause_pools.keys()),
        "output_path": output_path,
    }

    if verbose:
        total = max(1, len(raw_records))
        print("==================================================================")
        print(f"🎉 Phase 5 Knowledge Corpus Build Complete -> {output_path}")
        print(f"📦 Total Samples Written: {len(raw_records)}")
        print(f"  - Structured Context/Schema Split: {count_structured} ({count_structured / total * 100:.1f}%)")
        print(f"  - Disjoint K+/K- Mix:              {count_synthesized} ({count_synthesized / total * 100:.1f}%)")
        print(f"  - Empty Knowledge Stream ([]):     {count_empty} ({count_empty / total * 100:.1f}%)")
        print("==================================================================")

    return stats


def main():
    default_input = (
        "/content/omni_corpus.jsonl"
        if os.path.exists("/content/omni_corpus.jsonl")
        else "data/Basic_ZH/moss_dialogues.jsonl"
    )
    parser = argparse.ArgumentParser(
        description="Build Phase 5 Multi-Stream Knowledge Corpus (omni_knowledge_corpus.jsonl)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="auto",
        choices=["auto", "hf_stream", "local_file"],
        help="Execution mode: 'hf_stream' (HuggingFace streaming), 'local_file' (convert local JSONL), or 'auto'",
    )
    parser.add_argument(
        "--stream_hf",
        action="store_true",
        help="Shortcut flag to run in HuggingFace streaming mode (--mode hf_stream)",
    )
    parser.add_argument(
        "--domains",
        type=str,
        default="grounded_instruct,encyclopedia_rag",
        help="Comma-separated domains to stream in hf_stream mode (default: 'grounded_instruct,encyclopedia_rag', or 'all' for all 6 domains)",
    )
    parser.add_argument(
        "--input_path",
        "--input",
        dest="input_path",
        type=str,
        default=None,
        help="Input JSONL path for local_file mode (default: /content/omni_corpus.jsonl or data/Basic_ZH/moss_dialogues.jsonl)",
    )
    parser.add_argument(
        "--output_path",
        "--output",
        dest="output_path",
        type=str,
        default="/content/omni_knowledge_corpus.jsonl",
        help="Output multi-stream knowledge JSONL path",
    )
    parser.add_argument(
        "--samples_per_source",
        type=int,
        default=60000,
        help="Target samples per domain in hf_stream mode (default: 60000)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process in local_file mode",
    )
    parser.add_argument(
        "--max_know_streams",
        type=int,
        default=4,
        help="Maximum number of knowledge streams per sample (default: 4)",
    )
    parser.add_argument(
        "--know_seq_len",
        type=int,
        default=128,
        help="Maximum character length per knowledge stream (default: 128)",
    )
    parser.add_argument(
        "--empty_know_prob",
        type=float,
        default=0.15,
        help="Probability of empty knowledge stream [] for pure endogenous reasoning (default: 0.15)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic K+/K- sampling and shuffling (default: 42)",
    )
    args = parser.parse_args()

    use_hf_stream = args.stream_hf or args.mode == "hf_stream"
    resolved_input = args.input_path or default_input

    if not use_hf_stream and args.mode == "auto" and not os.path.isfile(resolved_input):
        use_hf_stream = True

    if use_hf_stream:
        stream_hf_6domain_knowledge_corpus(
            output_path=args.output_path,
            samples_per_source=args.samples_per_source,
            max_know_streams=args.max_know_streams,
            know_seq_len=args.know_seq_len,
            empty_know_prob=args.empty_know_prob,
            seed=args.seed,
            domains=args.domains,
        )
    else:
        build_phase5_knowledge_corpus(
            input_path=resolved_input,
            output_path=args.output_path,
            max_samples=args.max_samples,
            max_know_streams=args.max_know_streams,
            know_seq_len=args.know_seq_len,
            empty_know_prob=args.empty_know_prob,
            seed=args.seed,
            verbose=True,
        )


if __name__ == "__main__":
    main()
