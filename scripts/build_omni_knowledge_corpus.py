#!/usr/bin/env python3
"""
scripts/build_omni_knowledge_corpus.py
======================================
Grounded, Distractor-Free Multi-Domain Omni-Knowledge Corpus Builder for Phase 4 / Phase 5 Training.

Produces: `omni_knowledge_corpus.jsonl`

Downloads/streams high-quality data across 8 major domains, constructing 100% grounded
`(prompt, target)` pairs with zero artificial bracket tags (`【...】`) and zero negative
distractors ($K^-$).

In the redesigned Phase 4/5 architecture:
  - Right Brain (`FlowMatching1DUnet`) receives a 2D Sentence Vector `z_sent (1024,)`
    (`target_seq_len=128`) and pulls exact entity words, numbers, and terms from
    `k_attractors` (`prompt`, `prompt_seq_len=256`).
  - Every sample's `prompt` therefore contains the grounding material ($K^+$) required
    by `target` at non-identical token offsets (`prompt != target`, and
    `len(target) < 0.85 * len(prompt)` whenever `target in prompt`).

The 8 Core Grounded Domains:
  1. Grounded Chat & Instruct (真实受控对话与带材料指令: Belle, Alpaca-GPT4-ZH)
  2. Encyclopedia MRC & Grounded RAG (机器阅读理解与有据百科问答: CMRC2018, SQuAD, Wiki-CN, Zhihu)
  3. Literature & Narrative Summarization (文学语段归纳与叙事提炼: WebNovels)
  4. Code & Structured SQL (代码工程与结构化数据查询: CodeAlpaca, SQL Create Context)
  5. Math & Step-by-Step Logic (数学算式推导与严谨逻辑: GSM8K-ZH, Ape210K)
  6. Classical Chinese & Poetry (文言文互译与古典诗词鉴赏: Guji Classical Corpus, Tang/Song Poetry)
  7. News Summarization (新闻资讯核心事实概括: THUCNews)
  8. Bilingual Translation (中英双语平行语料互译: OPUS-100, WMT19)

Usage:
  python scripts/build_omni_knowledge_corpus.py \\
      --samples_per_source 100000 \\
      --output /content/omni_knowledge_corpus.jsonl \\
      --shuffle_buffer 50000 \\
      --seed 42
"""

import os
import re
import sys
import json
import random
import argparse
from typing import Iterator, List, Tuple, Dict, Any, Optional, Set, Union

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =========================================================================
# Text Cleaning, Grounding & Non-Cheating Utilities
# =========================================================================

_STRIP_PUNCT_RE = re.compile(
    r"^[\s,，.。!！?？;；:：、\"'“”‘’()（）\[\]【】]+|"
    r"[\s,，.。!！?？;；:：、\"'“”‘’()（）\[\]【】]+$"
)
_PUNCT_SPLIT_RE = re.compile(r"[\s,，.。!！?？;；:：、\"'“”‘’()（）\[\]【】<>《》—\-/\\|]+")
_LEADING_TAG_RE = re.compile(r"^\s*【[^】]{1,20}】[：:\s]*")

_REORDERED_BOILERPLATE_RE = re.compile(
    r"([^。！？；\n]+?)，\s*(?:划重点|先说结论|直接说结论)[：:]\s*([^。！？；\n]+?)([。！？；]|$)"
)
_FORUM_BOILERPLATE_RE = re.compile(
    r"(^|[。！？；\n]\s*)(?:划重点[：:]|先说结论[：:]|直接说结论[：:]|谢邀[，。！,.\s]*|泻药[，。！,.\s]*|答[：:])\s*"
)

_STOPWORD_CHARS = set("的了在是有和与或等中为对从到以及而但并把被让将给按于及其")
_STOPWORD_BIGRAMS = {
    "包括", "的是", "是在", "了一", "可以", "进行", "通过", "使用", "一个", "这个", "那个",
    "这些", "那些", "什么", "如何", "怎么", "哪些", "以下", "下列", "如下", "常见", "已知",
    "根据", "关于", "如果", "因为", "所以", "但是", "不过", "或者", "以及", "并且", "而且",
    "需要", "应该", "能够", "可能", "通常", "一般", "主要", "其中", "表示", "称为", "属于",
    "具有", "没有", "不是", "不能", "不会", "不同", "相同", "相关", "给定", "按类", "类别",
    "分组", "词语", "答案", "解析", "结果", "问题", "内容", "信息", "例如", "比如", "请根",
    "根据", "以上", "资料", "概括", "上文", "核心", "说明", "回答",
}


def clean_text(text: Optional[str]) -> str:
    """
    Cleans whitespace, strips all artificial `【...】` bracket tags, and removes
    forum/QA boilerplate prefixes (`划重点：`, `先说结论：`, `谢邀，`, etc.) while
    preserving the factual content that follows.
    """
    if not text or not isinstance(text, str):
        return ""
    text = text.strip()
    # Strip leading artificial bracket tags such as `【新闻报道】：` or `【经典原句】`
    while True:
        new_text = _LEADING_TAG_RE.sub("", text).strip()
        if new_text == text:
            break
        text = new_text
    # Unwrap any remaining inline `【...】` brackets into plain text
    text = re.sub(r"【([^】]*)】", r"\1", text)
    text = text.replace("【", "").replace("】", "")
    # Restore any pre-reordered `，划重点： <lead>` spans and strip forum boilerplate prefixes
    text = _REORDERED_BOILERPLATE_RE.sub(
        lambda m: f"{m.group(2).strip()}，{m.group(1).strip()}{m.group(3)}",
        text,
    )
    while True:
        next_text = _FORUM_BOILERPLATE_RE.sub(r"\1", text)
        if next_text == text:
            break
        text = next_text
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _ensure_sentence_end_punct(text: str) -> str:
    text = clean_text(text)
    if not text:
        return text
    text = re.sub(r"([。！？.!?;；]){2,}$", r"\1", text)
    if re.search(r"[。！？.!?;；\"'”’）\)]$", text):
        return text
    has_cjk = bool(re.search(r"[\u4e00-\u9fff]", text))
    text = re.sub(r"[,，、:：]+$", "", text).strip()
    return text + ("。" if has_cjk else ".")


def _trim_context_at_boundary(text: str, max_chars: int) -> str:
    """
    Trims `text` to at most `max_chars` characters at a natural sentence or clause
    boundary and ensures terminal punctuation (`。`/`.`), avoiding mid-word cuts.
    """
    text = text.strip()
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text

    window = text[:max_chars].strip()
    for punct in ["。", "！", "？", "；", "\n"]:
        pos = window.rfind(punct)
        if pos >= max(16, int(max_chars * 0.45)):
            return window[: pos + 1].strip()

    reserve_one = text[: max(1, max_chars - 1)].strip()
    for punct in ["，", ",", "、", "："]:
        pos = reserve_one.rfind(punct)
        if pos >= max(12, int(max_chars * 0.35)):
            prefix = reserve_one[:pos].strip()
            has_cjk = bool(re.search(r"[\u4e00-\u9fff]", prefix))
            return prefix + ("。" if has_cjk else ".")

    has_cjk = bool(re.search(r"[\u4e00-\u9fff]", reserve_one))
    if has_cjk and max_chars >= 4:
        return re.sub(r"[，,、：:；;\s]+$", "", reserve_one) + "。"
    return window


def _format_question_clean(question: str, max_q_chars: int = 100) -> str:
    """
    Formats a question string up to `max_q_chars` without mid-sentence truncation,
    cutting at the last clause boundary if `len(question) > max_q_chars` and ensuring
    terminal question/sentence punctuation.
    """
    q = clean_text(question)
    if not q:
        return ""
    has_cjk = bool(re.search(r"[\u4e00-\u9fff]", q))
    q_end = "？" if has_cjk else "?"

    if len(q) > max_q_chars:
        window = q[: max_q_chars - 1].strip()
        cut_pos = -1
        for punct in ["？", "?", "。", "！", "；", "，", ","]:
            pos = window.rfind(punct)
            if pos >= max(10, int(max_q_chars * 0.4)):
                cut_pos = pos
                break
        if cut_pos > 0:
            q = window[:cut_pos].strip()
        else:
            q = window

    if not re.search(r"[。！？?!.：:”’）\)]$", q):
        q = re.sub(r"[，,、；;]+$", "", q).strip() + q_end
    return q


def _extract_content_bigrams(text: str) -> Set[str]:
    """
    Extracts 2-character content substrings (and alphanumeric/numeric tokens) from
    non-punctuation segments of `text`, excluding generic functional stopwords.
    Matches `training/data/dialogue_dataset.py`.
    """
    if not text or not isinstance(text, str):
        return set()
    cleaned = clean_text(text)
    segments = [seg for seg in _PUNCT_SPLIT_RE.split(cleaned) if seg]
    bigrams: Set[str] = set()
    for seg in segments:
        if len(seg) >= 2:
            for i in range(len(seg) - 1):
                bg = seg[i : i + 2]
                if bg in _STOPWORD_BIGRAMS:
                    continue
                if bg[0] in _STOPWORD_CHARS or bg[1] in _STOPWORD_CHARS:
                    continue
                bigrams.add(bg)
    if not bigrams:
        joined = "".join(c for c in "".join(segments) if c not in _STOPWORD_CHARS)
        for i in range(len(joined) - 1):
            bg = joined[i : i + 2]
            if bg not in _STOPWORD_BIGRAMS:
                bigrams.add(bg)
    return bigrams


def has_substantive_overlap(prompt: str, target: str, min_bigrams: int = 2) -> bool:
    """
    Checks whether `prompt` and `target` share substantive entity/content overlap
    (`>= min_bigrams` content bigrams).
    """
    p_bg = _extract_content_bigrams(prompt)
    t_bg = _extract_content_bigrams(target)
    if not p_bg or not t_bg:
        return False
    overlap = len(p_bg & t_bg)
    if overlap >= min_bigrams:
        return True
    if len(t_bg) < min_bigrams and overlap >= 1:
        return True
    return False


def is_non_cheating_grounded_pair(prompt: str, target: str) -> bool:
    """
    Verifies that `(prompt, target)` is a valid grounded pair:
      1. Both `prompt` and `target` are non-empty and free of `【...】` bracket tags.
      2. `prompt != target` (after punctuation stripping).
      3. When `target in prompt`, requires `len(target) < 0.85 * len(prompt)`.
    """
    if not prompt or not target or not isinstance(prompt, str) or not isinstance(target, str):
        return False
    if "【" in prompt or "】" in prompt or "【" in target or "】" in target:
        return False

    p_clean = _STRIP_PUNCT_RE.sub("", prompt.strip())
    t_clean = _STRIP_PUNCT_RE.sub("", target.strip())
    if not p_clean or not t_clean:
        return False
    if p_clean == t_clean:
        return False

    if t_clean in p_clean:
        if len(t_clean) >= 0.85 * len(p_clean):
            return False

    if p_clean in t_clean:
        if len(p_clean) >= 24 and len(p_clean) >= 0.5 * len(t_clean):
            return False

    return True


def extract_factual_clauses(
    text: str,
    min_len: int = 10,
    max_len: int = 120,
) -> List[str]:
    """
    Splits text into natural factual/descriptive clauses or sentences by `。！？；\\n`
    with character length in `[min_len, max_len]`. Never adds artificial tags.
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


def _reorder_clauses_for_grounding(text: str) -> str:
    """
    Cleans trailing sentence punctuation while preserving the natural chronological
    and syntactic order of comma-separated sub-clauses.
    """
    return clean_text(re.sub(r"[。！？；\n]+$", "", text))


def try_split_structured_prompt(
    prompt: str,
    max_know_streams: int = 4,
    max_clause_len: int = 126,
) -> Optional[Tuple[str, List[str]]]:
    """
    Checks if `prompt` contains structured context/schema/input + query/question, such as:
      - `Schema: <ctx>\\nQuery: <question>`
      - `Context: <ctx>\\nQuestion: <question>`
      - Multi-line `instruction\\ninput` (common in BelleGroup/train_1M_CN)
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
        r"^\s*(?:Context|Reference|Passage|Background|Input|参考资料|背景知识|背景材料|已知条件|材料|给定文本|原文)[:：]\s*(.+?)\s*\n+\s*(?:Question|Query|Instruction|问题|提问|指令|请回答)[:：]\s*(.+)\s*$",
        prompt,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if ref_match:
        ref_str = clean_text(ref_match.group(1))
        q_str = clean_text(ref_match.group(2))
        if ref_str and q_str:
            ref_clauses = extract_factual_clauses(ref_str, min_len=6, max_len=max_clause_len)
            if not ref_clauses:
                ref_clauses = [ref_str[:max_clause_len]]
            return q_str, ref_clauses[:max_know_streams]

    # Pattern 3: Multi-line Instruction + Input block
    lines = [clean_text(ln) for ln in prompt.splitlines() if clean_text(ln)]
    if len(lines) >= 2 and len(lines[0]) >= 4:
        first_line = lines[0]
        rest = "\n".join(lines[1:]).strip()
        if len(rest) >= 8:
            rest_clauses = extract_factual_clauses(rest, min_len=6, max_len=max_clause_len)
            if not rest_clauses:
                rest_clauses = [rest[:max_clause_len]]
            return first_line, rest_clauses[:max_know_streams]

    return None


def try_split_classical_text(
    item_or_text: Union[Dict[str, Any], str],
    max_src_len: int = 200,
    max_tgt_len: int = 120,
) -> Optional[Tuple[str, str]]:
    """
    Extracts a genuine grounded classical-to-modern or classical context-to-clause pair
    `(src_prompt_material, target_text)` from `gujilab/chinese-classical-corpus` or similar
    classical Chinese records (supporting both dict items and raw text strings).
    """
    if isinstance(item_or_text, dict):
        # 1. Check explicit parallel fields first
        pair_candidates = [
            ("source", "target"),
            ("classical", "modern"),
            ("wenyan", "baihua"),
            ("origin", "translation"),
            ("src", "tgt"),
            ("input", "output"),
            ("question", "answer"),
        ]
        for k_src, k_tgt in pair_candidates:
            s_val = clean_text(str(item_or_text.get(k_src, "") or ""))
            t_val = clean_text(str(item_or_text.get(k_tgt, "") or ""))
            if len(s_val) >= 6 and len(t_val) >= 8 and s_val != t_val:
                instr = clean_text(str(item_or_text.get("instruction", "") or ""))
                if not instr or instr == s_val:
                    instr = "请将上述文言文译为现代汉语并阐释其意："
                prompt = f"古文原文：{_trim_context_at_boundary(s_val, max_src_len)}\n{instr}"
                target = _ensure_sentence_end_punct(t_val[:max_tgt_len])
                if is_non_cheating_grounded_pair(prompt, target):
                    return prompt, target

        # Check instruction + output when input is empty
        instr_val = clean_text(str(item_or_text.get("instruction", "") or ""))
        out_val = clean_text(str(item_or_text.get("output", item_or_text.get("response", "")) or ""))
        if len(instr_val) >= 8 and len(out_val) >= 8 and instr_val != out_val:
            split_res = try_split_structured_prompt(instr_val)
            if split_res is not None:
                q_str, k_list = split_res
                ctx_str = _trim_context_at_boundary("。".join(k_list), max_src_len)
                prompt = f"{ctx_str}\n{q_str}"
            else:
                prompt = f"古文原文：{_trim_context_at_boundary(instr_val, max_src_len)}\n请译为现代汉语："
            target = _ensure_sentence_end_punct(out_val[:max_tgt_len])
            if is_non_cheating_grounded_pair(prompt, target):
                return prompt, target

        # Otherwise fall through to raw text field extraction
        raw_text = ""
        for tk in ["text", "content", "paragraphs", "paragraph", "article", "sentence"]:
            val = item_or_text.get(tk)
            if isinstance(val, list):
                val = "\n".join(str(x) for x in val if x)
            if isinstance(val, str) and val.strip():
                raw_text = val.strip()
                break
        if not raw_text:
            return None
        title = clean_text(str(item_or_text.get("title", item_or_text.get("book", item_or_text.get("author", ""))) or ""))
    elif isinstance(item_or_text, str):
        raw_text = item_or_text.strip()
        title = ""
    else:
        return None

    if not raw_text:
        return None

    # 2. Check explicit 原/译 markers in raw_text
    marker_match = re.search(
        r"(?:【?原文】?|【?古文】?|【?文言文?】?)[:：\s]*(.+?)\s*(?:【?译文】?|【?现代文】?|【?白话文?】?|【?翻译】?|【?释义】?)[:：\s]*(.+)",
        raw_text,
        flags=re.DOTALL,
    )
    if marker_match:
        src_part = clean_text(marker_match.group(1))
        tgt_part = clean_text(marker_match.group(2))
        if len(src_part) >= 6 and len(tgt_part) >= 8 and src_part != tgt_part:
            prompt = f"古文原文：{_trim_context_at_boundary(src_part, max_src_len)}\n请将上述古文译为现代汉语："
            target = _ensure_sentence_end_punct(tgt_part[:max_tgt_len])
            if is_non_cheating_grounded_pair(prompt, target):
                return prompt, target

    # 3. Check line-separated parallel or multi-clause classical passages
    cleaned_raw = clean_text(raw_text)
    if len(cleaned_raw) < 20:
        return None

    clauses = extract_factual_clauses(cleaned_raw, min_len=8, max_len=90)
    if len(clauses) >= 2:
        target_clause = _ensure_sentence_end_punct(clauses[0][:max_tgt_len])
        natural_clauses = [_reorder_clauses_for_grounding(c) for c in clauses[:4]]
        supporting_ctx = _trim_context_at_boundary("。".join(natural_clauses) + "。", max_src_len)
        title_hint = f"（出自《{title}》）" if title else ""
        prompt = f"古籍语段{title_hint}：{supporting_ctx}\n请根据上述古文材料提炼核心首句："
        if is_non_cheating_grounded_pair(prompt, target_clause) and has_substantive_overlap(prompt, target_clause, min_bigrams=2):
            return prompt, target_clause

    return None


# =========================================================================
# Pair Builders for Domains 1 - 7 (Offline-Testable & Deterministic)
# =========================================================================

def build_grounded_instruct_pair(
    instruction: str,
    input_text: str,
    output_text: str,
    max_prompt_chars: int = 240,
    max_target_chars: int = 220,
) -> Optional[Tuple[str, str]]:
    """
    Domain 1 helper:
      - Recognizes structured multi-line prompts via `try_split_structured_prompt(instr)`
        when `input_text` is empty.
      - Places `inp` FIRST (`f"{inp}\\n{instr}"`) and keeps `instr` intact at the bottom
        while trimming `inp` at a clause boundary within the remaining budget.
      - Requires either `len(inp) >= 8` with content overlap or `>= 2` shared content bigrams
        between `prompt` and `output_text`.
    """
    instr = clean_text(instruction)
    inp = clean_text(input_text)
    out = clean_text(output_text)

    if len(instr) < 4 or len(out) < 10:
        return None

    if not inp:
        split_res = try_split_structured_prompt(instr, max_know_streams=3, max_clause_len=140)
        if split_res is not None:
            q_str, k_list = split_res
            inp = "。".join(k_list).strip()
            instr = q_str

    if inp:
        instr_clean = _format_question_clean(instr, max_q_chars=100) if len(instr) > 100 else instr
        inp_budget = max(60, max_prompt_chars - len(instr_clean) - 1)
        inp_trimmed = _trim_context_at_boundary(inp, inp_budget)
        prompt = f"{inp_trimmed}\n{instr_clean}".strip()
    else:
        prompt = _trim_context_at_boundary(instr, max_prompt_chars)

    target = _ensure_sentence_end_punct(out[:max_target_chars])

    is_grounded = (len(inp) >= 8 and has_substantive_overlap(prompt, target, min_bigrams=1)) or has_substantive_overlap(
        prompt, target, min_bigrams=2
    )
    if not is_grounded:
        return None

    if not is_non_cheating_grounded_pair(prompt, target):
        return None

    return prompt, target


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


def build_mrc_grounded_pair(
    context: str,
    question: str,
    raw_ans: str,
    max_window_chars: int = 200,
    max_target_chars: int = 120,
    max_prompt_chars: int = 240,
) -> Optional[Tuple[str, str]]:
    """
    Domain 2 MRC helper:
      - Preserves the complete question `q` (up to 100 chars, never cut mid-phrase at 45 chars).
      - Dynamically computes `ctx_budget = max(60, min(max_window_chars, max_prompt_chars - len(q_suffix)))`
        around the answer span so `prompt = f"{context_window}\\n问题：{q_clean}"` fits inside
        `prompt_seq_len=256` with the question 100% intact.
      - Uses the full sentence from `context` containing the answer (or `answer` if
        `len(answer) >= 10`) so `target` is always a complete clause (`10 <= len(target) <= 120`).
    """
    ctx = clean_text(context)
    q = clean_text(question)
    ans = clean_text(raw_ans)

    if len(ctx) < 25 or len(q) < 4 or len(ans) < 2:
        return None

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

    ans_sent = clean_text(sent_matches[best_sent_idx].group(0))

    if 10 <= len(ans) <= max_target_chars:
        target = _ensure_sentence_end_punct(ans)
    elif 10 <= len(ans_sent) <= max_target_chars:
        target = _ensure_sentence_end_punct(ans_sent)
    else:
        source_for_sub = ans_sent if len(ans_sent) >= 10 else ans
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
            while len(acc) < 16 and (left_i >= 0 or right_i < len(comma_spans)):
                if right_i < len(comma_spans) and len(acc) + len(comma_spans[right_i]) <= max_target_chars:
                    acc = acc + comma_spans[right_i]
                    right_i += 1
                elif left_i >= 0 and len(comma_spans[left_i]) + len(acc) <= max_target_chars:
                    acc = comma_spans[left_i] + acc
                    left_i -= 1
                else:
                    break
            target = _ensure_sentence_end_punct(acc[:max_target_chars])
        else:
            target = _ensure_sentence_end_punct(source_for_sub[:max_target_chars])

    if not (10 <= len(target) <= max_target_chars):
        return None

    q_clean = _format_question_clean(q, max_q_chars=100)
    q_suffix = f"\n问题：{q_clean}"
    eff_win_cap = max(60, min(max_window_chars, max_prompt_chars - len(q_suffix)))

    tgt_clean = _STRIP_PUNCT_RE.sub("", target)
    t_pos = ctx.find(tgt_clean)
    if t_pos < 0:
        t_pos = max(0, ans_pos)
    t_end = min(len(ctx), t_pos + len(tgt_clean))

    win_cap = min(eff_win_cap, len(ctx))
    span_len = max(1, t_end - t_pos)
    extra_room = max(0, win_cap - span_len)
    lead_room = min(t_pos, max(12, extra_room // 2))
    win_start = max(0, t_pos - lead_room)
    win_end = min(len(ctx), win_start + win_cap)
    win_start = max(0, win_end - win_cap)

    context_window = ctx[win_start:win_end].strip()
    if len(context_window) > eff_win_cap:
        context_window = context_window[:eff_win_cap].strip()

    prompt = f"{context_window}{q_suffix}"
    if not is_non_cheating_grounded_pair(prompt, target):
        return None

    return prompt, target


def build_grounded_wiki_pair(
    title: str,
    completion: str,
    max_prompt_chars: int = 240,
    max_target_chars: int = 120,
) -> Optional[Tuple[str, str]]:
    """
    Domain 2 Wikipedia helper:
      - Splits the Wikipedia entry (`completion`) into sentences.
      - Takes the first 1-2 core sentences as `target` (`15 <= len(target) <= 120`).
      - Constructs `prompt` from supporting context sentences in natural chronological order
        (`参考资料：...`) + question (`f"{supporting_context}\\n请根据以上资料概括{title}："`).
    """
    clean_title = clean_text(title)
    clean_body = clean_text(completion)
    if len(clean_body) < 45:
        return None

    sentences = [
        clean_text(s)
        for s in re.findall(r"[^。！？；\n]+[。！？；]*", clean_body[:900])
        if len(clean_text(s)) >= 10
    ]
    if len(sentences) < 2:
        clauses = extract_factual_clauses(clean_body, min_len=10, max_len=80)
        if len(clauses) < 2:
            return None
        sentences = [_ensure_sentence_end_punct(c) for c in clauses]

    s0 = sentences[0]
    if len(s0) > max_target_chars:
        sub_clauses = extract_factual_clauses(s0, min_len=12, max_len=max_target_chars - 2)
        if not sub_clauses:
            target = _ensure_sentence_end_punct(s0[: max_target_chars - 2])
        else:
            target = _ensure_sentence_end_punct(sub_clauses[0])
    elif len(s0) < 15 and len(sentences) >= 2 and len(s0) + len(sentences[1]) <= max_target_chars:
        target = _ensure_sentence_end_punct(f"{s0}{sentences[1]}")
    else:
        target = _ensure_sentence_end_punct(s0)

    if not (15 <= len(target) <= max_target_chars):
        return None

    natural_sents = [_reorder_clauses_for_grounding(s) for s in sentences[:4] if _reorder_clauses_for_grounding(s)]
    supporting_raw = "参考资料：" + "。".join(natural_sents) + "。"

    topic_label = clean_title if clean_title else "该词条"
    question_suffix = f"\n请根据以上资料概括{topic_label[:20]}："
    ctx_budget = max(60, max_prompt_chars - len(question_suffix))
    supporting_context = _trim_context_at_boundary(supporting_raw, ctx_budget)

    prompt = f"{supporting_context}{question_suffix}"

    if not is_non_cheating_grounded_pair(prompt, target):
        return None
    if not has_substantive_overlap(prompt, target, min_bigrams=2):
        return None

    return prompt, target


def build_grounded_zhihu_pair(
    question: str,
    answer: str,
    max_prompt_chars: int = 240,
    max_target_chars: int = 120,
) -> Optional[Tuple[str, str]]:
    """
    Domain 2 Zhihu helper:
      - Selects answers with multiple clauses (`len >= 60`).
      - Keeps supporting clauses in their natural chronological order (`背景资料：...`)
        so antecedents always precede pronouns.
      - Preserves the complete question `q` (up to 100 chars, never truncated at 45 chars)
        and trims background context to `ctx_budget = max_prompt_chars - len(q_suffix)`.
    """
    q = clean_text(question)
    a = clean_text(answer)
    if len(q) < 4 or len(a) < 60:
        return None

    clauses = extract_factual_clauses(a[:800], min_len=12, max_len=110)
    if len(clauses) < 2:
        return None

    target = _ensure_sentence_end_punct(clauses[0][:max_target_chars])
    if not (15 <= len(target) <= max_target_chars):
        return None

    natural_parts = [_reorder_clauses_for_grounding(c) for c in clauses[:4] if _reorder_clauses_for_grounding(c)]
    supporting_str = "背景资料：" + "。".join(natural_parts) + "。"

    q_clean = _format_question_clean(q, max_q_chars=100)
    q_suffix = f"\n问题：{q_clean}"
    ctx_budget = max(60, max_prompt_chars - len(q_suffix))
    supporting_trimmed = _trim_context_at_boundary(supporting_str, ctx_budget)
    prompt = f"{supporting_trimmed}{q_suffix}"

    if not is_non_cheating_grounded_pair(prompt, target):
        return None
    if not has_substantive_overlap(prompt, target, min_bigrams=2):
        return None

    return prompt, target


def build_grounded_literature_pair(
    text: str,
    max_prompt_chars: int = 240,
    max_target_chars: int = 120,
) -> Optional[Tuple[str, str]]:
    """
    Domain 3 Literature & Narrative Summarization helper:
      - Extracts a passage of 3-5 sentences (`100..240` chars) in natural chronological order.
      - Selects a key sentence from the passage that shares character/place entities
        with the rest of the passage as `target` (`15 <= len(target) <= 120`).
      - Places the passage context (`叙事片段：...`) + narrative prompt in `prompt`
        so sentence chronology is preserved and `prompt` and `target` do not start at the same token.
    """
    cleaned = clean_text(text)
    if len(cleaned) < 80:
        return None

    clauses = extract_factual_clauses(cleaned[:900], min_len=14, max_len=105)
    if len(clauses) < 3:
        return None

    passage_clauses = clauses[:4]
    best_idx = 0
    best_overlap = -1
    for i in range(len(passage_clauses)):
        cand = passage_clauses[i]
        if not (15 <= len(cand) <= max_target_chars - 2):
            continue
        others_text = "。".join(passage_clauses[:i] + passage_clauses[i + 1 :])
        ov = len(_extract_content_bigrams(cand) & _extract_content_bigrams(others_text))
        score = ov * 2 + (1 if i <= 1 else 0)
        if score > best_overlap:
            best_overlap = score
            best_idx = i

    target_raw = passage_clauses[best_idx]
    target = _ensure_sentence_end_punct(target_raw[:max_target_chars])
    if not (15 <= len(target) <= max_target_chars):
        return None

    natural_clauses = [_reorder_clauses_for_grounding(c) for c in passage_clauses if _reorder_clauses_for_grounding(c)]
    context_body = "叙事片段：" + "。".join(natural_clauses) + "。"

    suffix = "\n请根据上述叙事片段提炼关键情节："
    ctx_budget = max(80, max_prompt_chars - len(suffix))
    trimmed_body = _trim_context_at_boundary(context_body, ctx_budget)
    prompt = f"{trimmed_body}{suffix}"

    if not is_non_cheating_grounded_pair(prompt, target):
        return None
    if not has_substantive_overlap(prompt, target, min_bigrams=2):
        return None

    return prompt, target


def build_grounded_ape210k_pair(
    question: str,
    equation: str,
    cot: str = "",
    result: str = "",
) -> Optional[Tuple[str, str]]:
    """
    Domain 5 Math Ape210K helper:
      - Includes the equation `eq` in `prompt` (`f"{q}\\n已知算式：{eq}"`) when `cot` or `ans`
        is the target, so all numbers and mathematical operations in `target` are grounded
        in `prompt`.
    """
    q = clean_text(question)
    eq = clean_text(equation)
    cot_clean = clean_text(cot)
    res_clean = clean_text(str(result))

    if len(q) < 5 or not (eq or cot_clean):
        return None

    if eq:
        prompt = f"{q[:170]}\n已知算式：{eq[:60]}"
    else:
        prompt = q[:220]

    if cot_clean and len(cot_clean) >= 15:
        intact_cot = cot_clean.strip(" ，,。；;")
        if eq and res_clean:
            ans = f"{intact_cot}，依算式 {eq} 计算得出最终结果为 {res_clean}。"
        elif res_clean:
            ans = f"{intact_cot}，最终计算结果为 {res_clean}。"
        else:
            ans = _ensure_sentence_end_punct(intact_cot)
    elif eq and res_clean:
        ans = f"根据题设数量关系建立算式 {eq}，逐步求解得出最终答案 {res_clean}。"
    else:
        return None

    target = ans[:220].strip()
    if len(target) < 15 or not is_non_cheating_grounded_pair(prompt, target):
        return None

    return prompt, target


def build_grounded_poetry_pair(
    title: str,
    author: str,
    content: Union[str, List[str]],
    max_prompt_chars: int = 230,
    max_target_chars: int = 110,
) -> Optional[Tuple[str, str]]:
    """
    Domain 6 Classical Poetry helper:
      - Zero `【...】` bracket tags.
      - Keeps couplets in their natural order in `prompt` and preserves the trailing question intact.
    """
    clean_title = clean_text(title)
    clean_author = clean_text(author)
    if isinstance(content, list):
        raw_content = "。".join(
            re.sub(r"[。！？；\n]+$", "", clean_text(str(x)))
            for x in content
            if clean_text(str(x))
        ) + "。"
    else:
        raw_content = clean_text(str(content))

    if not raw_content or len(raw_content) < 16:
        return None

    couplets = [
        clean_text(s)
        for s in re.findall(r"[^。！？；\n]+[。！？；]*", raw_content)
        if len(clean_text(s)) >= 8
    ]
    if len(couplets) < 2:
        return None

    target_idx = 0
    target = _ensure_sentence_end_punct(couplets[target_idx][:max_target_chars])
    if not (10 <= len(target) <= max_target_chars):
        return None

    natural_couplets = [_reorder_clauses_for_grounding(c) for c in couplets[:4] if _reorder_clauses_for_grounding(c)]
    poem_ctx = "。".join(natural_couplets) + "。"

    meta_parts = []
    if clean_title:
        meta_parts.append(f"诗题：{clean_title}")
    if clean_author:
        meta_parts.append(f"作者：{clean_author}")
    meta_header = "，".join(meta_parts) if meta_parts else "古典诗词鉴赏"

    suffix = "\n请写出该诗对应的核心诗句："
    ctx_budget = max(60, max_prompt_chars - len(suffix))
    header_and_ctx = _trim_context_at_boundary(f"{meta_header}。关联诗句与意境：{poem_ctx}", ctx_budget)
    prompt = f"{header_and_ctx}{suffix}"

    if not is_non_cheating_grounded_pair(prompt, target):
        return None

    return prompt, target


def build_grounded_news_pair(
    text: str,
    label: str = "",
    max_prompt_chars: int = 240,
    max_target_chars: int = 120,
) -> Optional[Tuple[str, str]]:
    """
    Domain 7 News Summarization helper:
      - Zero `【新闻报道】` bracket tags.
      - Keeps news sentences in their natural chronological order (`新闻背景：...`)
        and places the lead summary sentence (`15 <= len(target) <= 120`) in `target`.
    """
    cleaned = clean_text(text)
    clean_label = clean_text(label)
    if len(cleaned) < 50:
        return None

    clauses = extract_factual_clauses(cleaned[:800], min_len=14, max_len=110)
    if len(clauses) < 2:
        return None

    target = _ensure_sentence_end_punct(clauses[0][:max_target_chars])
    if not (15 <= len(target) <= max_target_chars):
        return None

    natural_clauses = [_reorder_clauses_for_grounding(c) for c in clauses[:4] if _reorder_clauses_for_grounding(c)]
    news_body = "新闻背景：" + "。".join(natural_clauses) + "。"

    label_hint = f"{clean_label}领域" if clean_label and not clean_label.isdigit() else ""
    suffix = f"\n请根据上述新闻报道概括{label_hint}核心事实："
    ctx_budget = max(70, max_prompt_chars - len(suffix))
    trimmed_body = _trim_context_at_boundary(news_body, ctx_budget)
    prompt = f"{trimmed_body}{suffix}"

    if not is_non_cheating_grounded_pair(prompt, target):
        return None
    if not has_substantive_overlap(prompt, target, min_bigrams=2):
        return None

    return prompt, target


# =========================================================================
# Domain 1: Grounded Chat & Instruct (真实受控对话与带材料指令)
# =========================================================================

def extract_belle_instruct(max_samples: int) -> Iterator[Dict[str, str]]:
    """Domain 1: Grounded Chat & Instruct | Source: BelleGroup/train_1M_CN"""
    from datasets import load_dataset
    print(">>> [1/8 Chat] Streaming BelleGroup/train_1M_CN (Grounded)...")
    count = 0
    try:
        ds = load_dataset("BelleGroup/train_1M_CN", split="train", streaming=True)
        for item in ds:
            pair = build_grounded_instruct_pair(
                instruction=item.get("instruction", ""),
                input_text=item.get("input", ""),
                output_text=item.get("output", ""),
            )
            if pair is not None:
                yield {
                    "prompt": pair[0],
                    "target": pair[1],
                    "domain": "chat_instruct",
                    "source": "belle_1m",
                }
                count += 1
                if count >= max_samples:
                    break
    except Exception as e:
        print(f"⚠️ Warning during BelleGroup/train_1M_CN stream: {e}")
    print(f"  -> Extracted {count} grounded samples from BelleGroup/train_1M_CN")


def extract_gpt4_alpaca_zh(max_samples: int) -> Iterator[Dict[str, str]]:
    """Domain 1: Grounded Chat & Instruct | Source: silk-road/alpaca-data-gpt4-chinese"""
    from datasets import load_dataset
    print(">>> [1/8 Chat] Streaming silk-road/alpaca-data-gpt4-chinese (Grounded)...")
    count = 0
    try:
        ds = load_dataset("silk-road/alpaca-data-gpt4-chinese", split="train", streaming=True)
        for item in ds:
            pair = build_grounded_instruct_pair(
                instruction=item.get("instruction", ""),
                input_text=item.get("input", ""),
                output_text=item.get("output", ""),
            )
            if pair is not None:
                yield {
                    "prompt": pair[0],
                    "target": pair[1],
                    "domain": "chat_instruct",
                    "source": "alpaca_gpt4_zh",
                }
                count += 1
                if count >= max_samples:
                    break
    except Exception as e:
        print(f"⚠️ Warning during alpaca-data-gpt4-chinese stream: {e}")
    print(f"  -> Extracted {count} grounded samples from alpaca-data-gpt4-chinese")


# =========================================================================
# Domain 2: Encyclopedia MRC & Grounded RAG (百科阅读理解与有据问答)
# =========================================================================

def extract_chinese_mrc(max_samples: int) -> Iterator[Dict[str, str]]:
    """
    Domain 2: Encyclopedia MRC | Sources: `hfl/cmrc2018` (with Chinese MRC fallbacks)
    and `rajpurkar/squad`.
    """
    from datasets import load_dataset
    print(">>> [2/8 Encyclopedia MRC] Streaming hfl/cmrc2018 & rajpurkar/squad...")
    zh_target = max(1, int(0.75 * max_samples))
    en_target = max(1, max_samples - zh_target)

    zh_count = 0
    zh_sources: List[Tuple[str, Optional[str]]] = [
        ("hfl/cmrc2018", None),
        ("cmrc2018", None),
        ("clue", "cmrc2018"),
        ("PaddlePaddle/dureader_robust", None),
    ]
    for ds_name, ds_cfg in zh_sources:
        if zh_count >= zh_target:
            break
        try:
            ds = (
                load_dataset(ds_name, ds_cfg, split="train", streaming=True)
                if ds_cfg
                else load_dataset(ds_name, split="train", streaming=True)
            )
            for item in ds:
                ctx = str(item.get("context", item.get("passage", "")))
                q = str(item.get("question", item.get("query", "")))
                raw_ans = _extract_raw_answer_from_mrc_item(item)
                pair = build_mrc_grounded_pair(ctx, q, raw_ans)
                if pair is not None:
                    yield {
                        "prompt": pair[0],
                        "target": pair[1],
                        "domain": "encyclopedia",
                        "source": ds_name,
                    }
                    zh_count += 1
                    if zh_count >= zh_target:
                        break
        except Exception as e:
            print(f"⚠️ Warning during MRC stream {ds_name}: {e}")

    en_count = 0
    try:
        ds_en = load_dataset("rajpurkar/squad", split="train", streaming=True)
        for item in ds_en:
            ctx = str(item.get("context", ""))
            q = str(item.get("question", ""))
            raw_ans = _extract_raw_answer_from_mrc_item(item)
            pair = build_mrc_grounded_pair(ctx, q, raw_ans)
            if pair is not None:
                yield {
                    "prompt": pair[0],
                    "target": pair[1],
                    "domain": "encyclopedia",
                    "source": "squad",
                }
                en_count += 1
                if en_count >= en_target:
                    break
    except Exception as e:
        print(f"⚠️ Warning during rajpurkar/squad stream: {e}")

    print(f"  -> Extracted {zh_count + en_count} MRC samples (ZH={zh_count}, EN={en_count})")


def extract_wiki_chinese(max_samples: int) -> Iterator[Dict[str, str]]:
    """Domain 2: Grounded Encyclopedia | Source: pleisto/wikipedia-cn-20230720-filtered"""
    from datasets import load_dataset
    print(">>> [2/8 Encyclopedia] Streaming pleisto/wikipedia-cn-20230720-filtered (Grounded)...")
    count = 0
    try:
        ds = load_dataset("pleisto/wikipedia-cn-20230720-filtered", split="train", streaming=True)
        for item in ds:
            title = item.get("title", "")
            completion = item.get("completion", item.get("text", ""))
            pair = build_grounded_wiki_pair(title, completion)
            if pair is not None:
                yield {
                    "prompt": pair[0],
                    "target": pair[1],
                    "domain": "encyclopedia",
                    "source": "wiki_cn",
                }
                count += 1
                if count >= max_samples:
                    break
    except Exception as e:
        print(f"⚠️ Warning during wikipedia-cn stream: {e}")
    print(f"  -> Extracted {count} grounded samples from wikipedia-cn")


def extract_zhihu_qa(max_samples: int) -> Iterator[Dict[str, str]]:
    """Domain 2: Encyclopedia / Knowledge | Source: wangrui6/Zhihu-KOL or zirui3/zhihu_qa"""
    from datasets import load_dataset
    print(">>> [2/8 Encyclopedia] Streaming Zhihu Q&A & Essays (Grounded)...")
    count = 0
    for dname in ["wangrui6/Zhihu-KOL", "zirui3/zhihu_qa"]:
        if count >= max_samples:
            break
        try:
            ds = load_dataset(dname, split="train", streaming=True)
            for item in ds:
                q = item.get("INSTRUCTION", item.get("instruction", item.get("question", item.get("title", ""))))
                a = item.get("RESPONSE", item.get("output", item.get("answer", item.get("content", ""))))
                pair = build_grounded_zhihu_pair(q, a)
                if pair is not None:
                    yield {
                        "prompt": pair[0],
                        "target": pair[1],
                        "domain": "encyclopedia",
                        "source": dname,
                    }
                    count += 1
                    if count >= max_samples:
                        break
        except Exception as e:
            print(f"⚠️ Warning during {dname} stream: {e}")
    print(f"  -> Extracted {count} grounded samples from Zhihu")


# =========================================================================
# Domain 3: Literature & Narrative Summarization (文学语段归纳与叙事提炼)
# =========================================================================

def extract_webnovel_literature(max_samples: int) -> Iterator[Dict[str, str]]:
    """Domain 3: Literature & Narrative Summarization | Source: wdndev/webnovel-chinese or jetaudio/chinese_web_novels"""
    from datasets import load_dataset
    print(">>> [3/8 Literature] Streaming Chinese novels & narrative summarization...")
    count = 0
    candidates = ["wdndev/webnovel-chinese", "jetaudio/chinese_web_novels"]
    for cname in candidates:
        if count >= max_samples:
            break
        try:
            ds = load_dataset(cname, split="train", streaming=True)
            for item in ds:
                text = item.get("text", item.get("content", ""))
                pair = build_grounded_literature_pair(text)
                if pair is not None:
                    yield {
                        "prompt": pair[0],
                        "target": pair[1],
                        "domain": "literature",
                        "source": cname,
                    }
                    count += 1
                    if count >= max_samples:
                        break
        except Exception as e:
            print(f"⚠️ Warning during novel stream {cname}: {e}")
    print(f"  -> Extracted {count} grounded samples from Literature & Novels")


# =========================================================================
# Domain 4: Code & Structured Data (代码工程与结构化数据)
# =========================================================================

def extract_code_alpaca(max_samples: int) -> Iterator[Dict[str, str]]:
    """Domain 4: Code | Source: sahil2801/CodeAlpaca-20k or theblackcat102/evol-codealpaca-v1"""
    from datasets import load_dataset
    print(">>> [4/8 Code] Streaming CodeAlpaca...")
    count = 0
    for sname in ["sahil2801/CodeAlpaca-20k", "theblackcat102/evol-codealpaca-v1"]:
        if count >= max_samples:
            break
        try:
            ds = load_dataset(sname, split="train", streaming=True)
            for item in ds:
                instr = clean_text(item.get("instruction", ""))
                inp = clean_text(item.get("input", ""))
                out = clean_text(item.get("output", ""))
                if len(instr) < 5 or len(out) < 10:
                    continue
                prompt = f"{inp[:150]}\n{instr}".strip() if inp else instr
                target = out[:250].strip()
                if is_non_cheating_grounded_pair(prompt, target):
                    yield {
                        "prompt": prompt,
                        "target": target,
                        "domain": "code",
                        "source": sname,
                    }
                    count += 1
                    if count >= max_samples:
                        break
        except Exception as e:
            print(f"⚠️ Warning during {sname} stream: {e}")
    print(f"  -> Extracted {count} samples from CodeAlpaca")


def extract_sql_structured(max_samples: int) -> Iterator[Dict[str, str]]:
    """Domain 4: Code & Structured | Source: b-mc2/sql-create-context"""
    from datasets import load_dataset
    print(">>> [4/8 Code/SQL] Streaming b-mc2/sql-create-context...")
    count = 0
    try:
        ds = load_dataset("b-mc2/sql-create-context", split="train", streaming=True)
        for item in ds:
            prompt_q = clean_text(item.get("question", ""))
            ctx = clean_text(item.get("context", ""))
            answer = clean_text(item.get("answer", ""))
            prompt = f"Schema: {ctx}\nQuery: {prompt_q}"
            if prompt_q and ctx and answer and is_non_cheating_grounded_pair(prompt, answer):
                yield {
                    "prompt": prompt,
                    "target": answer[:250],
                    "domain": "code",
                    "source": "sql_create_context",
                }
                count += 1
                if count >= max_samples:
                    break
    except Exception as e:
        print(f"⚠️ Warning during sql-create-context stream: {e}")
    print(f"  -> Extracted {count} samples from SQL Context")


# =========================================================================
# Domain 5: Math & Step-by-Step Logic (数学推导与严谨逻辑)
# =========================================================================

def extract_math_gsm8k_zh(max_samples: int) -> Iterator[Dict[str, str]]:
    """Domain 5: Math & Logic | Source: meta-math/GSM8K_zh"""
    from datasets import load_dataset
    print(">>> [5/8 Math] Streaming meta-math/GSM8K_zh...")
    count = 0
    try:
        ds = load_dataset("meta-math/GSM8K_zh", split="train", streaming=True)
        for item in ds:
            q = clean_text(item.get("question_zh", item.get("question", item.get("instruction", ""))))
            ans = clean_text(item.get("answer_zh", item.get("answer", item.get("output", ""))))
            if len(q) >= 8 and len(ans) >= 12:
                target = ans[:250].strip()
                if is_non_cheating_grounded_pair(q, target):
                    yield {
                        "prompt": q[:240],
                        "target": target,
                        "domain": "math_logic",
                        "source": "gsm8k_zh",
                    }
                    count += 1
                    if count >= max_samples:
                        break
    except Exception as e:
        print(f"⚠️ Warning during GSM8K_zh stream: {e}")
    print(f"  -> Extracted {count} samples from GSM8K_zh")


def extract_math_ape210k(max_samples: int) -> Iterator[Dict[str, str]]:
    """Domain 5: Math & Logic | Source: MU-NLPC/calc-ape210k"""
    from datasets import load_dataset
    print(">>> [5/8 Math] Streaming MU-NLPC/calc-ape210k (Grounded with Equation)...")
    count = 0
    try:
        ds = load_dataset("MU-NLPC/calc-ape210k", split="train", streaming=True)
        for item in ds:
            q = item.get("question_chinese", item.get("question", ""))
            eq = item.get("equation", "")
            cot = item.get("chain_of_thought", "")
            res = item.get("result", item.get("answer", ""))
            pair = build_grounded_ape210k_pair(q, eq, cot, res)
            if pair is not None:
                yield {
                    "prompt": pair[0],
                    "target": pair[1],
                    "domain": "math_logic",
                    "source": "ape210k",
                }
                count += 1
                if count >= max_samples:
                    break
    except Exception as e:
        print(f"⚠️ Warning during calc-ape210k stream: {e}")
    print(f"  -> Extracted {count} grounded samples from Ape210K")


# =========================================================================
# Domain 6: Classical Chinese & Poetry (文言文与古典诗词)
# =========================================================================

def extract_classical_chinese(max_samples: int) -> Iterator[Dict[str, str]]:
    """
    Domain 6: Classical Chinese | Source: `gujilab/chinese-classical-corpus`
    """
    from datasets import load_dataset
    print(">>> [6/8 Classical Chinese] Streaming gujilab/chinese-classical-corpus...")
    count = 0
    try:
        ds = load_dataset("gujilab/chinese-classical-corpus", split="train", streaming=True)
        for item in ds:
            pair = try_split_classical_text(item)
            if pair is not None:
                yield {
                    "prompt": pair[0],
                    "target": pair[1],
                    "domain": "classical_chinese",
                    "source": "classical_corpus",
                }
                count += 1
                if count >= max_samples:
                    break
    except Exception as e:
        print(f"⚠️ Warning during classical-corpus stream: {e}")
    print(f"  -> Extracted {count} samples from gujilab/chinese-classical-corpus")


def extract_classical_poetry(max_samples: int) -> Iterator[Dict[str, str]]:
    """Domain 6: Classical Poetry | Source: Lifan-Z/Chinese-poetries-txt or erhwenkuo/poetry-chinese-zhtw"""
    from datasets import load_dataset
    print(">>> [6/8 Poetry] Streaming classical poetry (Grounded, bracket-free)...")
    count = 0
    for cname in ["Lifan-Z/Chinese-poetries-txt", "erhwenkuo/poetry-chinese-zhtw"]:
        if count >= max_samples:
            break
        try:
            ds = load_dataset(cname, split="train", streaming=True)
            for item in ds:
                title = item.get("title", "")
                author = item.get("author", "")
                content = item.get("content", item.get("text", item.get("paragraphs", "")))
                pair = build_grounded_poetry_pair(title, author, content)
                if pair is not None:
                    yield {
                        "prompt": pair[0],
                        "target": pair[1],
                        "domain": "classical_chinese",
                        "source": cname,
                    }
                    count += 1
                    if count >= max_samples:
                        break
        except Exception as e:
            print(f"⚠️ Warning during poetry stream {cname}: {e}")
    print(f"  -> Extracted {count} grounded samples from Classical Poetry")


# =========================================================================
# Domain 7: News Summarization (新闻资讯与现代纪实概括)
# =========================================================================

def extract_news_reporting(max_samples: int) -> Iterator[Dict[str, str]]:
    """Domain 7: News Summarization | Source: SirlyDreamer/THUCNews or seamew/THUCNews"""
    from datasets import load_dataset
    print(">>> [7/8 News & Reporting] Streaming THUCNews Chinese news summarization...")
    count = 0
    for dname in ["SirlyDreamer/THUCNews", "seamew/THUCNews"]:
        if count >= max_samples:
            break
        try:
            ds = load_dataset(dname, split="train", streaming=True)
            for item in ds:
                text = item.get("text", item.get("content", ""))
                label = str(item.get("label", ""))
                pair = build_grounded_news_pair(text, label=label)
                if pair is not None:
                    yield {
                        "prompt": pair[0],
                        "target": pair[1],
                        "domain": "news_reporting",
                        "source": dname,
                    }
                    count += 1
                    if count >= max_samples:
                        break
        except Exception as e:
            print(f"⚠️ Warning during news stream {dname}: {e}")
    print(f"  -> Extracted {count} grounded samples from News & Reporting")


# =========================================================================
# Domain 8: Bilingual Translation (中英双语平行语料与翻译)
# =========================================================================

def extract_opus_bilingual(max_samples: int) -> Iterator[Dict[str, str]]:
    """Domain 8: Bilingual Translation | Source: Helsinki-NLP/opus-100 (en-zh) with wmt/wmt19 fallback"""
    from datasets import load_dataset
    print(">>> [8/8 Bilingual/Translation] Streaming Helsinki-NLP/opus-100 (en-zh)...")
    count = 0
    sources: List[Tuple[str, str, str]] = [
        ("Helsinki-NLP/opus-100", "en-zh", "opus_100_en_zh"),
        ("wmt/wmt19", "zh-en", "wmt19_zh_en"),
    ]
    for ds_name, cfg, src_tag in sources:
        if count >= max_samples:
            break
        try:
            ds = load_dataset(ds_name, cfg, split="train", streaming=True)
            for item in ds:
                trans = item.get("translation", item)
                if not isinstance(trans, dict):
                    continue
                en = clean_text(trans.get("en", ""))
                zh = clean_text(trans.get("zh", ""))
                if not (10 <= len(en) <= 150 and 10 <= len(zh) <= 150):
                    continue
                if random.random() < 0.5:
                    prompt = f"Translate English to Chinese: {en}"
                    target = _ensure_sentence_end_punct(zh)
                else:
                    prompt = f"请将以下中文翻译为英文：{zh}"
                    target = _ensure_sentence_end_punct(en)
                if 10 <= len(target) <= 150 and is_non_cheating_grounded_pair(prompt, target):
                    yield {
                        "prompt": prompt,
                        "target": target,
                        "domain": "bilingual",
                        "source": src_tag,
                    }
                    count += 1
                    if count >= max_samples:
                        break
        except Exception as e:
            print(f"⚠️ Warning during bilingual stream {ds_name} ({cfg}): {e}")
    print(f"  -> Extracted {count} bilingual translation samples")


# =========================================================================
# Main Assembler
# =========================================================================

def build_omni_knowledge_corpus(
    output_path: str = "/content/omni_knowledge_corpus.jsonl",
    samples_per_source: int = 100000,
    shuffle_buffer_size: int = 50000,
    seed: int = 42,
):
    """
    Assembles the 8-domain grounded omni-knowledge corpus into `output_path`
    with streaming extractors and a randomized shuffle buffer.
    """
    random.seed(seed)
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # 8 Core Grounded Domains - Distractor-Free Phase 4/5 Spectrum
    extractors = [
        # Domain 1: Grounded Chat & Instruct
        ("1. Chat (Belle Grounded)", lambda: extract_belle_instruct(samples_per_source)),
        ("1. Chat (Alpaca-GPT4 Grounded)", lambda: extract_gpt4_alpaca_zh(samples_per_source)),

        # Domain 2: Encyclopedia MRC & Grounded RAG
        ("2. Encyclopedia (Chinese & SQuAD MRC)", lambda: extract_chinese_mrc(samples_per_source)),
        ("2. Encyclopedia (Wiki-CN Grounded)", lambda: extract_wiki_chinese(samples_per_source)),
        ("2. Encyclopedia (Zhihu Grounded)", lambda: extract_zhihu_qa(samples_per_source)),

        # Domain 3: Literature & Narrative Summarization
        ("3. Literature (Narrative Summarization)", lambda: extract_webnovel_literature(samples_per_source)),

        # Domain 4: Code & Structured SQL
        ("4. Code (CodeAlpaca)", lambda: extract_code_alpaca(samples_per_source)),
        ("4. Code (SQL Context)", lambda: extract_sql_structured(samples_per_source)),

        # Domain 5: Math & Logic
        ("5. Math (GSM8K-ZH)", lambda: extract_math_gsm8k_zh(samples_per_source)),
        ("5. Math (Ape210K Grounded)", lambda: extract_math_ape210k(samples_per_source)),

        # Domain 6: Classical Chinese & Poetry
        ("6. Classical (Classical Corpus)", lambda: extract_classical_chinese(samples_per_source)),
        ("6. Poetry (Tang/Song Grounded)", lambda: extract_classical_poetry(samples_per_source)),

        # Domain 7: News Summarization
        ("7. News (THUCNews Summarization)", lambda: extract_news_reporting(samples_per_source)),

        # Domain 8: Bilingual Translation
        ("8. Bilingual (OPUS-100 / WMT EN-ZH)", lambda: extract_opus_bilingual(samples_per_source)),
    ]

    print("==================================================================")
    print(f"🚀 Starting 8-Domain Grounded Omni-Knowledge Corpus Assembly -> {output_path}")
    print(f"🎯 Target Samples per Source: {samples_per_source}")
    print(f"📚 Total Data Streams: {len(extractors)}")
    print("==================================================================")

    total_written = 0
    buffer: List[Dict[str, str]] = []

    with open(output_path, "w", encoding="utf-8") as out_f:
        for idx, (label, ext_fn) in enumerate(extractors, 1):
            print(f"\n[{idx}/{len(extractors)}] Executing: {label}...")
            source_count = 0
            for item in ext_fn():
                buffer.append(item)
                source_count += 1

                if len(buffer) >= shuffle_buffer_size:
                    random.shuffle(buffer)
                    for b_item in buffer:
                        out_f.write(json.dumps(b_item, ensure_ascii=False) + "\n")
                    total_written += len(buffer)
                    print(f"  💾 Flushed {len(buffer)} samples (Total written: {total_written})...")
                    buffer.clear()
                    out_f.flush()
            print(f"  -> Stream [{idx}] collected {source_count} samples.")

        if buffer:
            random.shuffle(buffer)
            for b_item in buffer:
                out_f.write(json.dumps(b_item, ensure_ascii=False) + "\n")
            total_written += len(buffer)
            buffer.clear()
            out_f.flush()

    print("\n==================================================================")
    print("🎉 8-Domain Grounded Omni-Knowledge Corpus Assembly Complete!")
    print(f"📦 Total Samples Saved: {total_written}")
    print(f"📁 Output: {output_path}")
    if os.path.exists(output_path):
        print(f"📊 Size: {os.path.getsize(output_path) / (1024 * 1024):.2f} MB")
    print("==================================================================")


# Backwards-compatible alias
build_omni_corpus = build_omni_knowledge_corpus


def main():
    parser = argparse.ArgumentParser(
        description="Build Grounded 8-Domain Omni-Knowledge Corpus for Phase 4 / Phase 5 Continuous Flow Matching"
    )
    parser.add_argument(
        "--samples_per_source",
        type=int,
        default=100000,
        help="Max samples extracted per data source (default: 100000)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/content/omni_knowledge_corpus.jsonl",
        help="Output JSONL file path (default: /content/omni_knowledge_corpus.jsonl)",
    )
    parser.add_argument(
        "--shuffle_buffer",
        type=int,
        default=50000,
        help="Buffer size for randomized writing (default: 50000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling and bidirectional sampling (default: 42)",
    )
    args = parser.parse_args()

    build_omni_knowledge_corpus(
        output_path=args.output,
        samples_per_source=args.samples_per_source,
        shuffle_buffer_size=args.shuffle_buffer,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
