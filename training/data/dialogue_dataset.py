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

_STOPWORD_CHARS = set("的了在是有和与或等中为对从到以及而但并把被让将给按于及其")
_STOPWORD_BIGRAMS = {
    "包括", "的是", "是在", "了一", "可以", "进行", "通过", "使用", "一个", "这个", "那个",
    "这些", "那些", "什么", "如何", "怎么", "哪些", "以下", "下列", "如下", "常见", "已知",
    "根据", "关于", "如果", "因为", "所以", "但是", "不过", "或者", "以及", "并且", "而且",
    "需要", "应该", "能够", "可能", "通常", "一般", "主要", "其中", "表示", "称为", "属于",
    "具有", "没有", "不是", "不能", "不会", "不同", "相同", "相关", "给定", "按类", "类别",
    "分组", "词语", "答案", "解析", "结果", "问题", "内容", "信息", "例如", "比如",
}


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
    max_context_chars: int = 220,
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
    text = target_text.strip()
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

    base_clean = (base_context or "").strip()
    if len(base_clean) > max_context_chars:
        base_clean = base_clean[:max_context_chars]

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
                # Keep base_context prefix + most recent history tail within max_context_chars
                base_budget = min(len(base_clean), max(40, max_context_chars - 80))
                kept_base = base_clean[:base_budget]
                rem_budget = max(0, max_context_chars - len(kept_base) - 1)
                kept_hist = history_str[-rem_budget:] if rem_budget > 0 else ""
                curr_context = f"{kept_base}\n{kept_hist}".strip() if kept_base else kept_hist
        pairs.append((curr_context, clause))

    return pairs


class GenericPairDataset(Dataset):
    """
    A unified dataset loader for pair/dialogue/knowledge tasks (Prompt + Genuine Knowledge -> Single-Clause Target).
    Supports:
      - Multiple formats: JSONL, JSON, Parquet
      - Automatic negative distractor (K-) removal on `"knowledge"` lists via `_filter_genuine_knowledge`
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

        base_context = self._build_base_context(item, p_str, t_str)
        eff_min_len = max(self.min_clause_chars, self.min_target_len) if self.split_by_punctuation else self.min_target_len

        if self.split_by_punctuation:
            max_ctx_chars = max(220, self.prompt_seq_len - 24)
            return _split_target_by_punctuation(
                base_context=base_context,
                target_text=t_str,
                max_clause_chars=self.max_clause_chars,
                min_clause_chars=eff_min_len,
                max_context_chars=max_ctx_chars,
            )

        if eff_min_len > 0 and len(t_str.strip()) < eff_min_len:
            return []
        return [(base_context, t_str)]

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
