import os
import re
import math
import glob
import json
from typing import List, Union, Optional, Tuple, Dict, Any, Set
import torch
from torch.utils.data import Dataset


_PUNCT_SPLIT_RE = re.compile(r"[\s,，.。!！?？;；:：、\"'“”‘’()（）\[\]【】<>《》—\-/\\|]+")
_LEADING_TAG_RE = re.compile(r"^\s*【[^】]{1,12}】[：:\s]*")

# Forum / QA boilerplate patterns
_REORDERED_BOILERPLATE_RE = re.compile(
    r"([^。！？；\n]+?)，\s*(?:划重点|先说结论|直接说结论)[：:]\s*([^。！？；\n]+?)([。！？；]|$)"
)
_LEADING_BOILERPLATE_RE = re.compile(
    r"(^|[。！？；\n]\s*)(?:划重点[：:]|先说结论[：:]|直接说结论[：:]|谢邀[，。！,.\s]*|泻药[，。！,.\s]*|答[：:])\s*"
)
_DANGLING_QUESTION_TAIL_RULES: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"([，,]\s*如何)$"), r"\1看待？"),
    (re.compile(r"([，,]\s*怎么)$"), r"\1看？"),
    (re.compile(r"([，,]\s*怎样)$"), r"\1看待？"),
    (re.compile(r"([，,]\s*(?:为何|为什么))$"), r"，原因是什么？"),
    (re.compile(r"([，,]\s*看待)$"), r"，如何看待？"),
    (re.compile(r"([，,]\s*评价)$"), r"，如何评价？"),
    (re.compile(r"(如何看待|如何评价|怎么看|怎么评价)$"), r"\1？"),
]

_STOPWORD_CHARS = set("的了在是有和与或等中为对从到以及而但并把被让将给按于及其")
_STOPWORD_BIGRAMS = {
    "包括", "的是", "是在", "了一", "可以", "进行", "通过", "使用", "一个", "这个", "那个",
    "这些", "那些", "什么", "如何", "怎么", "哪些", "以下", "下列", "如下", "常见", "已知",
    "根据", "关于", "如果", "因为", "所以", "但是", "不过", "或者", "以及", "并且", "而且",
    "需要", "应该", "能够", "可能", "通常", "一般", "主要", "其中", "表示", "称为", "属于",
    "具有", "没有", "不是", "不能", "不会", "不同", "相同", "相关", "给定", "按类", "类别",
    "分组", "词语", "答案", "解析", "结果", "问题", "内容", "信息", "例如", "比如",
}


def _clean_forum_boilerplate(text: str) -> str:
    """
    Cleans forum/QA boilerplate markers and repairs pre-existing corpus artifacts:
      1. Restores sentences where `_reorder_clauses_for_grounding` moved
         `，划重点： <lead_span>` to the end of a sentence back to `<lead_span>，<rest>`.
      2. Strips leading forum boilerplate prefixes (`划重点：`, `先说结论：`, `直接说结论：`,
         `谢邀，`, `泻药，`, `答：`) at the start of text or after sentence boundaries (`。！？；\\n`),
         preserving the factual content that follows.
      3. Repairs dangling truncated question tails (`，如何$`, `，怎么$`, `，为何$`, etc.)
         on `问题：...` lines and ensures terminal punctuation.
    """
    if not text or not isinstance(text, str):
        return ""

    cleaned = text.strip()
    if not cleaned:
        return ""

    # 1. Restore `<rest>，划重点： <lead>` -> `<lead>，<rest>`
    cleaned = _REORDERED_BOILERPLATE_RE.sub(
        lambda m: f"{m.group(2).strip()}，{m.group(1).strip()}{m.group(3)}",
        cleaned,
    )

    # 2. Strip leading forum boilerplate prefixes after start-of-string or sentence boundary
    while True:
        next_cleaned = _LEADING_BOILERPLATE_RE.sub(r"\1", cleaned)
        if next_cleaned == cleaned:
            break
        cleaned = next_cleaned

    # 3. Repair dangling truncated question lines (`问题：...`)
    lines = cleaned.split("\n")
    for idx, line in enumerate(lines):
        stripped_line = line.strip()
        if stripped_line.startswith("问题：") or stripped_line.startswith("问题:"):
            for pat, repl in _DANGLING_QUESTION_TAIL_RULES:
                if pat.search(stripped_line):
                    stripped_line = pat.sub(repl, stripped_line)
                    break
            if not re.search(r"[。！？?!.：:”’）\)]$", stripped_line):
                stripped_line = re.sub(r"[，,、；;]+$", "", stripped_line).strip() + "？"
            lines[idx] = stripped_line

    cleaned = "\n".join(lines).strip()
    return cleaned


def _ensure_sent_punct(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    text = re.sub(r"([。！？.!?;；]){2,}$", r"\1", text)
    if re.search(r"[。！？.!?;；\"'”’）\)]$", text):
        return text
    has_cjk = bool(re.search(r"[\u4e00-\u9fff]", text))
    text = re.sub(r"[,，、:：]+$", "", text).strip()
    return text + ("。" if has_cjk else ".")


def _restore_displaced_lead_sentence(base_context: str, target_text: str) -> str:
    """
    Repairs legacy corpus samples where `build_omni_knowledge_corpus.py` placed a
    comma-reordered (`parts[1:] + parts[:1]`) version of the lead target sentence at the
    END of the background paragraph (`clauses[1:4] + [reordered_lead]`):
      1. Restores the sentence's natural comma order using `t_first`.
      2. Moves the displaced lead sentence back to the FRONT of the background paragraph
         so chronological/narrative order (`[t_first] + bg_sents[:-1]`) is preserved.
    """
    if not base_context or not target_text:
        return base_context or ""

    if "\n" in base_context:
        bg_text, q_line = base_context.rsplit("\n", 1)
        bg_text = bg_text.strip()
        q_line = q_line.strip()
    else:
        bg_text = base_context.strip()
        q_line = ""

    if not bg_text:
        return base_context

    t_sents = [s.strip() for s in re.findall(r"[^。！？；\n]+[。！？；]*", target_text.strip()) if s.strip()]
    if not t_sents:
        return base_context
    t_first = _ensure_sent_punct(t_sents[0])
    b_t = _extract_content_bigrams(t_first)
    if not b_t:
        return base_context

    bg_sents = [s.strip() for s in re.findall(r"[^。！？；\n]+[。！？；]*", bg_text) if s.strip()]
    if not bg_sents:
        return base_context

    # Case A: Single background sentence that had its comma sub-clauses rotated
    if len(bg_sents) == 1:
        b_0 = _extract_content_bigrams(bg_sents[0])
        if b_0 and (len(b_0 & b_t) / float(max(1, min(len(b_0), len(b_t))))) >= 0.80:
            t_parts = [p.strip() for p in re.split(r"[，,；;。！？]+", t_first) if p.strip()]
            s_parts = [p.strip() for p in re.split(r"[，,；;。！？]+", bg_sents[0]) if p.strip()]
            if len(t_parts) >= 2 and s_parts == t_parts[1:] + t_parts[:1]:
                return f"{t_first}\n{q_line}" if q_line else t_first
        return base_context

    # Case B: Multi-sentence background where `bg_sents[0]` is NOT `t_first`,
    #         and `bg_sents[-1]` IS the displaced/comma-reordered version of `t_first`.
    b_first = _extract_content_bigrams(bg_sents[0])
    first_overlap = len(b_first & b_t) / float(max(1, min(len(b_first), len(b_t)))) if b_first else 0.0
    if first_overlap >= 0.75:
        return base_context

    b_last = _extract_content_bigrams(bg_sents[-1])
    if not b_last:
        return base_context
    last_overlap = len(b_last & b_t) / float(max(1, min(len(b_last), len(b_t))))
    if last_overlap >= 0.70:
        restored_sents = [t_first] + [_ensure_sent_punct(s) for s in bg_sents[:-1]]
        new_bg = "".join(restored_sents)
        return f"{new_bg}\n{q_line}" if q_line else new_bg

    return base_context


def _trim_at_boundary(text: str, budget: int) -> str:
    """
    Trims `text` to at most `budget` characters at a natural sentence or clause boundary,
    ensuring CJK text ends with terminal punctuation (`。`) rather than mid-word.
    """
    text = text.strip()
    if budget <= 0:
        return ""
    if len(text) <= budget:
        return text

    window = text[:budget].strip()
    # Prefer cutting at the last full sentence boundary if it retains >= 45% of budget
    for punct in ["。", "！", "？", "；", "\n"]:
        pos = window.rfind(punct)
        if pos >= max(12, int(budget * 0.45)):
            return window[: pos + 1].strip()

    # Otherwise cut at the last comma/clause boundary if it retains >= 35% of budget
    reserve_one = text[: max(1, budget - 1)].strip()
    for punct in ["，", ",", "、", "："]:
        pos = reserve_one.rfind(punct)
        if pos >= max(10, int(budget * 0.35)):
            prefix = reserve_one[:pos].strip()
            has_cjk = bool(re.search(r"[\u4e00-\u9fff]", prefix))
            return prefix + ("。" if has_cjk else ".")

    has_cjk = bool(re.search(r"[\u4e00-\u9fff]", reserve_one))
    if has_cjk and budget >= 4:
        return re.sub(r"[，,、：:；;\s]+$", "", reserve_one) + "。"
    return window


def _trim_context_preserving_question(base_context: str, max_context_chars: int) -> str:
    """
    Trims `base_context` to fit within `max_context_chars` while keeping the trailing
    question/instruction line (`\\n问题：...` or `\\n请根据...`) 100% intact.
    """
    base_clean = _clean_forum_boilerplate(base_context or "")
    if not base_clean or max_context_chars <= 0:
        return ""
    if len(base_clean) <= max_context_chars:
        return base_clean

    if "\n" in base_clean:
        bg_text, q_line = base_clean.rsplit("\n", 1)
        q_line = q_line.strip()
        bg_text = bg_text.strip()
        if q_line and len(q_line) < max_context_chars - 16:
            bg_budget = max_context_chars - len(q_line) - 1
            bg_trimmed = _trim_at_boundary(bg_text, bg_budget)
            if bg_trimmed:
                return f"{bg_trimmed}\n{q_line}"
            return q_line[:max_context_chars]

    return _trim_at_boundary(base_clean, max_context_chars)


def _extract_content_bigrams(text: str) -> Set[str]:
    """
    Extracts 2-character content substrings from non-punctuation/non-whitespace segments of `text`,
    excluding generic functional stopwords and particle bigrams.
    """
    if not text or not isinstance(text, str):
        return set()
    cleaned = _LEADING_TAG_RE.sub("", text.strip())
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


def _filter_genuine_knowledge(
    prompt: str,
    knowledge_list: List[str],
    target: str,
    max_clauses: int = 2,
) -> List[str]:
    """
    Filters a record's `"knowledge"` list (such as from `omni_knowledge_corpus.jsonl`) to strip out
    unrelated negative distractors (K-, e.g. `"旅行目的地: 日本"`, `"常见编程语言包括 Python 和 Rust。"`)
    and retain only genuine positive background clauses (K+) that share substantive 2-character content
    substrings with `target` or `prompt`. Keeps up to `max_clauses` (default 2) matching items.
    """
    if not knowledge_list or not isinstance(knowledge_list, list):
        return []

    ref_bigrams = _extract_content_bigrams(f"{prompt or ''} {target or ''}")
    if not ref_bigrams:
        return []

    scored_matches: List[Tuple[int, int, str]] = []
    for idx, raw_k in enumerate(knowledge_list):
        if not raw_k or not isinstance(raw_k, str):
            continue
        k_clean = _LEADING_TAG_RE.sub("", raw_k.strip()).strip()
        if not k_clean:
            continue
        k_bigrams = _extract_content_bigrams(k_clean)
        if not k_bigrams:
            continue
        overlap = len(k_bigrams & ref_bigrams)
        if overlap <= 0:
            continue
        # For longer knowledge clauses (>= 5 content bigrams), require >= 2 matching bigrams or >= 25% overlap
        if len(k_bigrams) >= 5 and overlap < 2 and (overlap / float(len(k_bigrams))) < 0.25:
            continue
        scored_matches.append((overlap, -idx, k_clean))

    if not scored_matches:
        return []

    best_overlap = max(item[0] for item in scored_matches)
    min_keep_overlap = max(1, int(math.ceil(0.35 * best_overlap)))
    filtered_matches = [item for item in scored_matches if item[0] >= min_keep_overlap]

    # Preserve original order among top matching clauses (up to max_clauses)
    filtered_matches.sort(key=lambda item: (item[0], item[1]), reverse=True)
    top_matches = sorted(filtered_matches[:max_clauses], key=lambda item: -item[1])
    return [item[2] for item in top_matches]


def _split_overlong_span(span: str, max_clause_chars: int) -> List[str]:
    """
    Splits a single sentence span exceeding `max_clause_chars` by commas `，,`,
    and hard-slices any still-overlong segment so no returned piece exceeds `max_clause_chars`.
    """
    span = span.strip()
    if not span:
        return []
    if len(span) <= max_clause_chars:
        return [span]

    comma_parts = [p.strip() for p in re.findall(r"[^，,]+[，,]*", span) if p.strip()]
    pieces: List[str] = []
    acc = ""
    for part in comma_parts:
        if len(part) > max_clause_chars:
            if acc:
                pieces.append(acc)
                acc = ""
            for i in range(0, len(part), max_clause_chars):
                chunk = part[i : i + max_clause_chars].strip()
                if chunk:
                    pieces.append(chunk)
            continue

        candidate = f"{acc}{part}" if acc else part
        if len(candidate) <= max_clause_chars:
            acc = candidate
        else:
            if acc:
                pieces.append(acc)
            acc = part

    if acc:
        pieces.append(acc)
    return pieces


def _split_target_by_punctuation(
    base_context: str,
    target_text: str,
    max_clause_chars: int = 120,
    min_clause_chars: int = 8,
    max_context_chars: int = 248,
) -> List[Tuple[str, str]]:
    """
    Splits `target_text` by sentence/line boundaries (`r"[^。！？；\\n]+[。！？；\\n]*"`),
    further splits over-long sentences (`> max_clause_chars`) by comma `，,`,
    merges adjacent short fragments (`< min_clause_chars`) into `[min_clause_chars, max_clause_chars]`,
    and builds a progressive sentence chain `(curr_context, clause_i)`:
      - Clause 0: `(base_context, clause_0)`
      - Clause i >= 1: appends `clause_0 .. clause_{i-1}` to `base_context` within `max_context_chars`.
    """
    if not target_text or not isinstance(target_text, str):
        return []
    text = _clean_forum_boilerplate(target_text)
    if not text:
        return []

    raw_sentences = [
        s.strip()
        for s in re.findall(r"[^。！？；\n]+[。！？；\n]*", text)
        if s.strip()
    ]
    if not raw_sentences:
        raw_sentences = [text]

    bounded_spans: List[str] = []
    for sent in raw_sentences:
        bounded_spans.extend(_split_overlong_span(sent, max_clause_chars))

    # Merge adjacent short fragments (< min_clause_chars)
    clauses: List[str] = []
    buf = ""
    for span in bounded_spans:
        if not buf:
            buf = span
        elif len(buf) < min_clause_chars:
            sep = "" if re.search(r"[。！？；，,：:\n]$", buf) else " "
            candidate = f"{buf}{sep}{span}"
            if len(candidate) <= max_clause_chars:
                buf = candidate
            else:
                clauses.append(buf)
                buf = span
        else:
            clauses.append(buf)
            buf = span

        if len(buf) >= min_clause_chars:
            clauses.append(buf)
            buf = ""

    if buf:
        if clauses and len(buf) < min_clause_chars:
            sep = "" if re.search(r"[。！？；，,：:\n]$", clauses[-1]) else " "
            merged_tail = f"{clauses[-1]}{sep}{buf}"
            if len(merged_tail) <= max_clause_chars:
                clauses[-1] = merged_tail
            else:
                clauses.append(buf)
        else:
            clauses.append(buf)

    # Filter out any residual fragment shorter than min_clause_chars
    clauses = [c.strip() for c in clauses if len(c.strip()) >= min_clause_chars]
    if not clauses:
        return []

    raw_base_clean = _restore_displaced_lead_sentence(
        _clean_forum_boilerplate(base_context or ""),
        text,
    )
    base_clean = _trim_context_preserving_question(raw_base_clean, max_context_chars)

    pairs: List[Tuple[str, str]] = []
    for i, clause in enumerate(clauses):
        if i == 0:
            curr_context = base_clean
        else:
            history_str = "".join(clauses[:i]).strip()
            candidate_ctx = f"{base_clean}\n{history_str}" if base_clean else history_str
            if len(candidate_ctx) <= max_context_chars:
                curr_context = candidate_ctx
            else:
                # Preserve trailing question line in base_context + most recent history tail within max_context_chars
                hist_budget = min(len(history_str), max(24, min(80, max_context_chars // 3)))
                kept_hist = history_str[-hist_budget:] if hist_budget > 0 else ""
                base_budget = max(40, max_context_chars - len(kept_hist) - 1)
                kept_base = _trim_context_preserving_question(raw_base_clean, base_budget)
                curr_context = f"{kept_base}\n{kept_hist}".strip() if kept_base else kept_hist[:max_context_chars]
        pairs.append((curr_context, clause))

    return pairs


class GenericPairDataset(Dataset):
    """
    A unified dataset loader for pair/dialogue/knowledge tasks (Prompt + Genuine Knowledge -> Single-Clause Target).
    Supports:
      - Multiple formats: JSONL, JSON, Parquet
      - Automatic negative distractor (K-) removal on `"knowledge"` lists via `_filter_genuine_knowledge`
      - Automatic forum boilerplate cleaning & question-preserving context trimming
      - Punctuation-split progressive single-clause target chains via `_split_target_by_punctuation`
      - Default `seq_len = 128` (`target_seq_len = 128`, `prompt_seq_len = max(seq_len, 256)`)
    """

    PROMPT_CANDIDATE_KEYS = [
        "prompt", "instruction", "input", "query", "user", "human",
        "question", "source", "context", "text_a"
    ]
    TARGET_CANDIDATE_KEYS = [
        "target", "output", "response", "assistant", "bot",
        "answer", "completion", "text_b"
    ]

    def __init__(
        self,
        data_path: Union[str, List[str]],
        tokenizer,
        seq_len: int = 128,
        prompt_seq_len: Optional[int] = None,
        target_seq_len: Optional[int] = None,
        prompt_key: Optional[str] = None,
        target_key: Optional[str] = None,
        max_samples: Optional[int] = None,
        min_target_len: int = 0,
        split_by_punctuation: bool = True,
        max_clause_chars: int = 120,
        min_clause_chars: int = 8,
        verbose: bool = True,
    ):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.prompt_seq_len = prompt_seq_len if prompt_seq_len is not None else max(seq_len, 256)
        self.target_seq_len = target_seq_len if target_seq_len is not None else seq_len
        self.prompt_key = prompt_key
        self.target_key = target_key
        self.max_samples = max_samples
        self.min_target_len = min_target_len
        self.split_by_punctuation = split_by_punctuation
        self.max_clause_chars = max_clause_chars
        self.min_clause_chars = min_clause_chars
        self.verbose = verbose

        self.samples: List[Tuple[str, str]] = []
        self._load_data(data_path)

        if self.verbose:
            print(
                f"✅ Loaded {len(self.samples)} single-clause pairs successfully "
                f"(prompt_seq_len={self.prompt_seq_len}, target_seq_len={self.target_seq_len}, "
                f"split_by_punctuation={self.split_by_punctuation})."
            )

    def _collect_file_paths(self, data_path: Union[str, List[str]]) -> List[str]:
        raw_paths = [data_path] if isinstance(data_path, str) else data_path
        files = []
        for path in raw_paths:
            if os.path.isdir(path):
                for ext in ["*.jsonl", "*.json", "*.parquet"]:
                    files.extend(glob.glob(os.path.join(path, ext)))
            elif "*" in path or "?" in path:
                files.extend(glob.glob(path))
            elif os.path.isfile(path):
                files.append(path)
            else:
                if self.verbose:
                    print(f"⚠️ Warning: Path not found: {path}")
        return sorted(list(set(files)))

    def _extract_raw_prompt_target(self, item: Dict[str, Any]) -> Optional[Tuple[str, str]]:
        if not isinstance(item, dict):
            return None

        # 1. Check for OpenAI/ShareGPT message lists
        for conv_key in ["messages", "conversations"]:
            if conv_key in item and isinstance(item[conv_key], list):
                turns = item[conv_key]
                user_text, assistant_text = "", ""
                for turn in turns:
                    if not isinstance(turn, dict):
                        continue
                    role = turn.get("role", turn.get("from", "")).lower()
                    content = turn.get("content", turn.get("value", ""))
                    if role in ["user", "human"]:
                        user_text = content
                    elif role in ["assistant", "bot", "gpt", "model"]:
                        assistant_text = content
                if user_text and assistant_text:
                    return str(user_text).strip(), str(assistant_text).strip()

        # 2. Check explicitly provided keys
        if self.prompt_key and self.target_key:
            if self.prompt_key in item and self.target_key in item:
                return str(item[self.prompt_key]).strip(), str(item[self.target_key]).strip()

        # 3. Auto-detect keys from candidates
        found_prompt_key = None
        if self.prompt_key and self.prompt_key in item:
            found_prompt_key = self.prompt_key
        else:
            for k in self.PROMPT_CANDIDATE_KEYS:
                if k in item and item[k]:
                    found_prompt_key = k
                    break

        found_target_key = None
        if self.target_key and self.target_key in item:
            found_target_key = self.target_key
        else:
            for k in self.TARGET_CANDIDATE_KEYS:
                if k in item and item[k]:
                    found_target_key = k
                    break

        if found_prompt_key and found_target_key:
            return str(item[found_prompt_key]).strip(), str(item[found_target_key]).strip()

        return None

    def _build_base_context(self, item: Dict[str, Any], prompt_str: str, target_str: str) -> str:
        raw_know = item.get("knowledge")
        k_list: List[str] = []
        if isinstance(raw_know, list):
            k_list = [str(k).strip() for k in raw_know if k is not None and str(k).strip()]
        elif isinstance(raw_know, str) and raw_know.strip():
            k_list = [raw_know.strip()]

        if k_list:
            k_plus = _filter_genuine_knowledge(prompt_str, k_list, target_str)
            if k_plus:
                return f"{' '.join(k_plus)}\n{prompt_str}"
        return prompt_str

    def _extract_pairs_from_dict(self, item: Dict[str, Any]) -> List[Tuple[str, str]]:
        raw_pair = self._extract_raw_prompt_target(item)
        if not raw_pair:
            return []
        p_str, t_str = raw_pair
        if not p_str or not t_str:
            return []

        p_str = _clean_forum_boilerplate(p_str)
        t_str = _clean_forum_boilerplate(t_str)
        if not p_str or not t_str:
            return []

        base_context = self._build_base_context(item, p_str, t_str)
        eff_min_len = max(self.min_clause_chars, self.min_target_len) if self.split_by_punctuation else self.min_target_len
        max_ctx_chars = max(120, self.prompt_seq_len - 8)

        if self.split_by_punctuation:
            return _split_target_by_punctuation(
                base_context=base_context,
                target_text=t_str,
                max_clause_chars=self.max_clause_chars,
                min_clause_chars=eff_min_len,
                max_context_chars=max_ctx_chars,
            )

        if eff_min_len > 0 and len(t_str.strip()) < eff_min_len:
            return []
        restored_ctx = _restore_displaced_lead_sentence(base_context, t_str)
        return [(_trim_context_preserving_question(restored_ctx, max_ctx_chars), t_str)]

    def _extract_pair_from_dict(self, item: Dict[str, Any]) -> Optional[Tuple[str, str]]:
        pairs = self._extract_pairs_from_dict(item)
        return pairs[0] if pairs else None

    def _load_data(self, data_path: Union[str, List[str]]):
        files = self._collect_file_paths(data_path)
        if not files:
            raise FileNotFoundError(f"No valid data files found from path(s): {data_path}")

        for file_path in files:
            if self.max_samples and len(self.samples) >= self.max_samples:
                break

            if self.verbose:
                print(f"📖 Reading dataset file: {file_path} ...")

            if file_path.endswith(".jsonl"):
                self._load_jsonl(file_path)
            elif file_path.endswith(".parquet"):
                self._load_parquet(file_path)
            elif file_path.endswith(".json"):
                self._load_json(file_path)

    def _load_jsonl(self, file_path: str):
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if self.max_samples and len(self.samples) >= self.max_samples:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    pairs = self._extract_pairs_from_dict(obj)
                    for pair in pairs:
                        if self.max_samples and len(self.samples) >= self.max_samples:
                            break
                        self.samples.append(pair)
                except Exception:
                    continue

    def _load_json(self, file_path: str):
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                if isinstance(data, list):
                    for obj in data:
                        if self.max_samples and len(self.samples) >= self.max_samples:
                            break
                        pairs = self._extract_pairs_from_dict(obj)
                        for pair in pairs:
                            if self.max_samples and len(self.samples) >= self.max_samples:
                                break
                            self.samples.append(pair)
            except Exception as e:
                if self.verbose:
                    print(f"⚠️ Error reading JSON file {file_path}: {e}")

    def _load_parquet(self, file_path: str):
        try:
            import pyarrow.parquet as pq
            table = pq.read_table(file_path)
            col_names = table.column_names
            prompt_col = self.prompt_key or next((c for c in self.PROMPT_CANDIDATE_KEYS if c in col_names), None)
            target_col = self.target_key or next((c for c in self.TARGET_CANDIDATE_KEYS if c in col_names), None)

            if not prompt_col or not target_col:
                if self.verbose:
                    print(f"⚠️ Parquet file {file_path} missing prompt/target columns (available: {col_names})")
                return

            prompts = table[prompt_col].to_pylist()
            targets = table[target_col].to_pylist()
            for p, t in zip(prompts, targets):
                if self.max_samples and len(self.samples) >= self.max_samples:
                    break
                if p and t:
                    pairs = self._extract_pairs_from_dict({prompt_col: str(p), target_col: str(t)})
                    for pair in pairs:
                        if self.max_samples and len(self.samples) >= self.max_samples:
                            break
                        self.samples.append(pair)
        except Exception as e:
            if self.verbose:
                print(f"⚠️ Error reading Parquet file {file_path}: {e}")

    def __len__(self) -> int:
        return len(self.samples)

    def _pad_trunc(self, ids: List[int], target_len: int) -> List[int]:
        if len(ids) > target_len:
            return ids[:target_len]
        return ids + [self.tokenizer.pad_token_id] * (target_len - len(ids))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        prompt_text, target_text = self.samples[idx]

        w_prompt = self.tokenizer.encode(prompt_text, add_special_tokens=True)
        w_target = self.tokenizer.encode(target_text, add_special_tokens=True)

        w_prompt = self._pad_trunc(w_prompt, self.prompt_seq_len)
        w_target = self._pad_trunc(w_target, self.target_seq_len)

        return (
            torch.tensor(w_prompt, dtype=torch.long),
            torch.tensor(w_target, dtype=torch.long),
        )


# Convenient aliases
DialogueDataset = GenericPairDataset
