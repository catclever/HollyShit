import os
import re
import glob
import json
import math
import random
from typing import List, Union, Optional, Tuple, Dict, Any
import torch
from torch.utils.data import Dataset, DataLoader

from training.core.char_tokenizer import CharTokenizer
from training.data.dialogue_dataset import (
    _clean_forum_boilerplate,
    _split_into_raw_clauses,
    _split_overlong_span,
)


CANONICAL_DOMAINS = [
    "grounded_instruct",
    "code_api",
    "classical_quote",
    "encyclopedia_rag",
    "math_rule",
    "literature_lore",
    "general",
    "dialogue",
]
DOMAIN_TO_ID = {d: i for i, d in enumerate(CANONICAL_DOMAINS)}
ID_TO_DOMAIN = {i: d for i, d in enumerate(CANONICAL_DOMAINS)}


_STRIP_PUNCT_RE = re.compile(r"^[\s,，.。!！?？;；:：、\"'“”‘’()（）\[\]【】]+|[\s,，.。!！?？;；:：、\"'“”‘’()（）\[\]【】]+$")


def _split_into_short_clauses(
    text: str,
    min_chars: int = 6,
    max_chars: int = 28,
) -> List[str]:
    """
    Splits `text` into natural punctuation-driven short clauses (`min_chars .. max_chars` chars)
    preserving short parentheticals (`（...）`, `(...)`), decimals (`3.14`), times (`12:30`),
    and code identifiers (`user_id`). Merges tiny fragments (`< min_chars`) into adjacent clauses
    within a soft ceiling of `max(max_chars, 30)` characters (`<= 30` chars + `<BOS>/<EOS>` <= 32 tokens).
    """
    if not text or not isinstance(text, str):
        return []
    cleaned = _clean_forum_boilerplate(text)
    if not cleaned:
        return []

    raw_clauses = _split_into_raw_clauses(cleaned, short_paren_limit=16)
    if not raw_clauses:
        raw_clauses = [cleaned]

    bounded_spans: List[str] = []
    for raw_c in raw_clauses:
        bounded_spans.extend(
            _split_overlong_span(raw_c, max_clause_chars=max_chars, min_clause_chars=min_chars)
        )

    soft_merge_ceiling = max(max_chars, min(max_chars + 2, 30))
    clauses: List[str] = []
    buf = ""
    for span in bounded_spans:
        if not buf:
            buf = span
        elif len(buf) < min_chars:
            sep = "" if re.search(r"[。！？；，,：:、\n）\)]$", buf) else " "
            candidate = f"{buf}{sep}{span}"
            if len(candidate) <= soft_merge_ceiling:
                buf = candidate
            elif clauses and len(clauses[-1]) + len(buf) <= soft_merge_ceiling:
                prev_sep = "" if re.search(r"[。！？；，,：:、\n）\)]$", clauses[-1]) else " "
                clauses[-1] = f"{clauses[-1]}{prev_sep}{buf}"
                buf = span
            else:
                re_split = _split_overlong_span(
                    candidate, max_clause_chars=max_chars, min_clause_chars=min_chars
                )
                clauses.extend(re_split[:-1])
                buf = re_split[-1] if re_split else ""
        else:
            clauses.append(buf)
            buf = span

        if len(buf) >= min_chars:
            clauses.append(buf)
            buf = ""

    if buf:
        if clauses and len(buf) < min_chars:
            sep = "" if re.search(r"[。！？；，,：:、\n）\)]$", clauses[-1]) else " "
            merged_tail = f"{clauses[-1]}{sep}{buf}"
            if len(merged_tail) <= soft_merge_ceiling:
                clauses[-1] = merged_tail
            else:
                re_split = _split_overlong_span(
                    merged_tail, max_clause_chars=max_chars, min_clause_chars=min_chars
                )
                clauses[-1] = re_split[0]
                clauses.extend(re_split[1:])
        else:
            clauses.append(buf)

    filtered = [c.strip() for c in clauses if len(c.strip()) >= min_chars]
    if not filtered and cleaned and min_chars <= 4:
        return [cleaned[:max_chars].strip()]
    return filtered


def _extract_clauses_from_text(
    text: Union[str, List[str]],
    min_chars: int = 6,
    max_chars: int = 28,
) -> List[str]:
    """
    Extracts punctuation short clauses from either a single string or a list of strings.
    """
    if not text:
        return []
    if isinstance(text, list):
        out: List[str] = []
        for item in text:
            if item and isinstance(item, str):
                out.extend(_split_into_short_clauses(item, min_chars=min_chars, max_chars=max_chars))
        return out
    if isinstance(text, str):
        return _split_into_short_clauses(text, min_chars=min_chars, max_chars=max_chars)
    return []


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


def extract_factual_clauses(
    text: str,
    min_len: int = 12,
    max_len: int = 128,
) -> List[str]:
    """
    Splits text into natural factual/descriptive clauses or sentences by `。！？；\\n`
    with character length in [min_len, max_len]. Never adds artificial tags.
    """
    if not text or not isinstance(text, str):
        return []

    raw_parts = re.split(r"[。！？；\n]+", text)
    clauses: List[str] = []

    for part in raw_parts:
        part = part.strip()
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


def split_structured_prompt(
    prompt: str,
    max_know_streams: int = 4,
    max_clause_len: int = 126,
) -> Optional[Tuple[str, List[str]]]:
    """
    Detects if `prompt` contains structured context/schema/input + question/query,
    and splits the reference into natural `knowledge` streams (K+) and the question into `prompt`.
    Never prepends artificial type tags.
    """
    if not prompt or "\n" not in prompt:
        return None

    # 1. SQL Schema + Query
    sql_match = re.match(
        r"^\s*Schema:\s*(.+?)\s*\n+\s*Query:\s*(.+)\s*$",
        prompt,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if sql_match:
        schema_str = sql_match.group(1).strip()
        query_str = sql_match.group(2).strip()
        if schema_str and query_str:
            schema_parts = [
                p.strip()[:max_clause_len]
                for p in re.split(r"[;\n]+", schema_str)
                if p.strip()
            ]
            if not schema_parts:
                schema_parts = [schema_str[:max_clause_len]]
            return query_str, schema_parts[:max_know_streams]

    # 2. Explicit Context/Reference/Background + Question/Instruction markers
    ref_match = re.match(
        r"^\s*(?:Context|Reference|Passage|Background|Input|参考资料|背景知识|背景材料|已知条件|材料)[:：]\s*(.+?)\s*\n+\s*(?:Question|Query|Instruction|问题|提问|指令|请回答)[:：]\s*(.+)\s*$",
        prompt,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if ref_match:
        ref_str = ref_match.group(1).strip()
        q_str = ref_match.group(2).strip()
        if ref_str and q_str:
            ref_clauses = extract_factual_clauses(ref_str, min_len=8, max_len=max_clause_len)
            if not ref_clauses:
                ref_clauses = [ref_str[:max_clause_len]]
            return q_str, ref_clauses[:max_know_streams]

    # 3. Multi-line Instruction + Input block (e.g. Belle / Alpaca `instruction\ninput`)
    lines = [ln.strip() for ln in prompt.splitlines() if ln.strip()]
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
      - `new_target`: the remaining continuation text with `k_plus` removed.
    """
    text = target_text.strip()
    if not text:
        return [], text

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
                re.sub(r"[。！？；\n]+$", "", s).strip()[:max_clause_len]
                for s in sentence_spans[:num_take]
            ]
            k_plus = [
                k for k in raw_k
                if k and is_non_cheating_knowledge(k, rem_text)
            ]
            if k_plus:
                return k_plus, rem_text

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
                re.sub(r"[,，;；。！？\n]+$", "", s).strip()[:max_clause_len]
                for s in comma_spans[:num_take]
            ]
            k_plus = [
                k for k in raw_k
                if len(k) >= 6 and is_non_cheating_knowledge(k, rem_text)
            ]
            if k_plus:
                return k_plus, rem_text

    return [], text


def reground_legacy_disjoint_sample(
    target_text: str,
    explicit_know: Optional[List[str]],
    domain: str,
    max_clause_len: int = 126,
    max_know_streams: int = 4,
) -> Tuple[str, Optional[List[str]]]:
    """
    On-the-fly Legacy Disjoint-Split Re-Grounding for `encyclopedia_rag` / `literature_lore`.
    """
    if domain not in ("encyclopedia_rag", "literature_lore"):
        return target_text, explicit_know

    if not explicit_know or not isinstance(explicit_know, list) or len(explicit_know) == 0:
        return target_text, explicit_know

    k0_raw = str(explicit_know[0]).strip()
    t_raw = target_text.strip()
    if not k0_raw or not t_raw:
        return target_text, explicit_know

    tgt_head = _STRIP_PUNCT_RE.sub("", t_raw[:60])
    k0_clean = _STRIP_PUNCT_RE.sub("", k0_raw)
    if len(tgt_head) >= 3:
        trigrams = [tgt_head[i : i + 3] for i in range(len(tgt_head) - 2)]
        has_overlap = any(tg in k0_clean for tg in trigrams)
    else:
        has_overlap = tgt_head in k0_clean

    if has_overlap:
        return target_text, explicit_know

    k0_no_end = re.sub(r"[。！？；，,\s]+$", "", k0_raw)
    full_passage = f"{k0_no_end}。{t_raw}"
    if len(full_passage) < 40 and len(explicit_know) > 1:
        k1_no_end = re.sub(r"[。！？；，,\s]+$", "", str(explicit_know[1]).strip())
        if k1_no_end:
            full_passage = f"{full_passage}{k1_no_end}。"

    eff_win_cap = min(max_clause_len, len(full_passage))
    max_tgt_len = min(85, max(16, int(0.72 * eff_win_cap)))

    sent_spans = [
        s.strip()
        for s in re.findall(r"[^。！？；\n]+[。！？；\n]*", t_raw)
        if s.strip()
    ]
    new_target = ""
    if sent_spans and 15 <= len(sent_spans[0]) <= max_tgt_len:
        new_target = sent_spans[0]
    else:
        comma_spans = [
            s.strip()
            for s in re.findall(r"[^，,;；。！？\n]+[，,;；。！？]*", t_raw)
            if s.strip()
        ]
        acc = ""
        for sp in comma_spans:
            cand = f"{acc}{sp}" if acc else sp
            if len(cand) <= max_tgt_len:
                acc = cand
                if len(acc) >= 24 and re.search(r"[。！？；]$", acc):
                    break
            else:
                break
        if len(acc) >= 15:
            new_target = acc
        else:
            new_target = t_raw[:max_tgt_len].strip()

    new_target_core = _STRIP_PUNCT_RE.sub("", new_target)
    if not re.search(r"[。！？.!?;；]$", new_target):
        if len(new_target) < max_tgt_len:
            new_target = re.sub(r"[,，、:：]+$", "", new_target) + "。"

    start_pos = full_passage.find(new_target_core)
    if start_pos < 0:
        start_pos = len(k0_no_end) + 1

    min_k_len = max(int(len(new_target_core) / 0.75) + 6, len(new_target_core) + 16)
    desired_k_len = min(max_clause_len, max(min(80, len(full_passage)), min_k_len, eff_win_cap))
    desired_k_len = min(desired_k_len, len(full_passage))

    target_end = start_pos + len(new_target_core)
    if target_end <= desired_k_len:
        win_start = 0
        win_end = desired_k_len
    else:
        win_end = min(len(full_passage), target_end + 16)
        win_start = max(0, win_end - desired_k_len)

    new_k_plus = full_passage[win_start:win_end].strip()

    if not is_non_cheating_knowledge(new_k_plus, new_target):
        trim_len = max(15, int(0.70 * len(_STRIP_PUNCT_RE.sub("", new_k_plus))))
        new_target = _STRIP_PUNCT_RE.sub("", new_target)[:trim_len] + "。"

    other_know = [
        k for k in explicit_know[1:]
        if k and is_non_cheating_knowledge(str(k), new_target)
    ]
    new_know_list = ([new_k_plus] + other_know)[:max_know_streams]
    return new_target, new_know_list


class MultiStreamKnowledgeDataset(Dataset):
    """
    Phase 5 Multi-Stream Knowledge Dataset for Dual-Brain Mamba Planner + Flow Matching.
    """

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

    def __init__(
        self,
        data_path: Union[str, List[str]],
        tokenizer=None,
        seq_len: int = 512,
        context_len: Optional[int] = None,
        max_know_streams: int = 4,
        know_seq_len: int = 128,
        prompt_key: Optional[str] = None,
        target_key: Optional[str] = None,
        knowledge_key: str = "knowledge",
        max_samples: Optional[int] = None,
        min_target_len: int = 15,
        auto_extract_knowledge: bool = True,
        return_domain_id: bool = False,
        active_domains: Optional[Union[List[str], str]] = None,
        seed: int = 42,
        verbose: bool = True,
    ):
        self.tokenizer = tokenizer if tokenizer is not None else CharTokenizer()
        self.seq_len = seq_len
        self.context_len = context_len if context_len is not None else seq_len
        self.max_know_streams = max_know_streams
        self.know_seq_len = know_seq_len
        self.prompt_key = prompt_key
        self.target_key = target_key
        self.knowledge_key = knowledge_key
        self.max_samples = max_samples
        self.min_target_len = min_target_len
        self.auto_extract_knowledge = auto_extract_knowledge
        self.return_domain_id = return_domain_id
        self.seed = seed
        self.verbose = verbose

        if active_domains is None:
            self.active_domains: Optional[set] = None
        elif isinstance(active_domains, str):
            parts = [d.strip() for d in active_domains.split(",") if d.strip()]
            if not parts or "all" in [p.lower() for p in parts]:
                self.active_domains = None
            else:
                self.active_domains = set(parts)
        else:
            parts = [str(d).strip() for d in active_domains if str(d).strip()]
            if not parts or "all" in [p.lower() for p in parts]:
                self.active_domains = None
            else:
                self.active_domains = set(parts)

        self.pad_token_id = getattr(self.tokenizer, "pad_token_id", 0)
        self.eos_token_id = getattr(self.tokenizer, "eos_token_id", None)

        self.samples: List[Tuple[str, str, Optional[List[str]], str]] = []
        self.clause_pool: List[Tuple[int, str]] = []
        self.domain_clause_pool: Dict[str, List[Tuple[int, str]]] = {}
        self.regrounded_count: int = 0

        self._load_data(data_path)

        if self.auto_extract_knowledge:
            self._build_clause_pool()

        if self.verbose:
            reg_msg = f", re-grounded legacy samples: {self.regrounded_count}" if self.regrounded_count > 0 else ""
            dom_msg = f", active_domains={sorted(self.active_domains)}" if self.active_domains else ""
            print(
                f"✅ Loaded {len(self.samples)} multi-stream samples "
                f"(clause pool: {len(self.clause_pool)} clauses across {len(self.domain_clause_pool)} domains{reg_msg}{dom_msg})."
            )

    def _collect_file_paths(self, data_path: Union[str, List[str]]) -> List[str]:
        if isinstance(data_path, str):
            raw_paths = [p.strip() for p in data_path.split(",") if p.strip()]
        else:
            raw_paths = list(data_path)
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

    @staticmethod
    def _strip_artificial_tags(text: str) -> str:
        return re.sub(r"^\s*【[^】]{1,12}】[：:\s]*", "", text).strip()

    def _parse_knowledge_field(
        self, item: Dict[str, Any], prompt_key_used: Optional[str], target_text: str
    ) -> Optional[List[str]]:
        max_clause_len = max(12, self.know_seq_len - 2)

        if self.knowledge_key and self.knowledge_key in item and item[self.knowledge_key] is not None:
            raw_know = item[self.knowledge_key]
            if isinstance(raw_know, list):
                cleaned = [
                    self._strip_artificial_tags(str(k))
                    for k in raw_know
                    if k is not None and str(k).strip()
                ]
                cleaned = [k for k in cleaned if k and is_non_cheating_knowledge(k, target_text)]
                return cleaned[: self.max_know_streams]
            if isinstance(raw_know, str):
                s = self._strip_artificial_tags(raw_know)
                if not s:
                    return None
                if is_non_cheating_knowledge(s, target_text):
                    return [s[:max_clause_len]][: self.max_know_streams]

        for cand_k in self.SEPARATE_KNOWLEDGE_CANDIDATE_KEYS:
            if cand_k == prompt_key_used:
                continue
            val = item.get(cand_k)
            if not isinstance(val, str):
                continue
            val_s = val.strip()
            if cand_k == "source" and re.match(r"^[a-zA-Z0-9_./-]+$", val_s):
                continue
            if len(val_s) >= 6:
                s = self._strip_artificial_tags(val_s)
                if s and is_non_cheating_knowledge(s, target_text):
                    clauses = extract_factual_clauses(s, min_len=6, max_len=max_clause_len)
                    if not clauses:
                        clauses = [s[:max_clause_len]]
                    clauses = [c for c in clauses if is_non_cheating_knowledge(c, target_text)]
                    if clauses:
                        return clauses[: self.max_know_streams]

        return None

    def _finalize_record(
        self,
        p_str: str,
        t_str: str,
        explicit_know: Optional[List[str]],
        domain: str,
    ) -> Optional[Tuple[str, str, Optional[List[str]], str]]:
        max_clause_len = max(12, self.know_seq_len - 2)
        if domain in ("encyclopedia_rag", "literature_lore") and explicit_know:
            new_t, new_k = reground_legacy_disjoint_sample(
                target_text=t_str,
                explicit_know=explicit_know,
                domain=domain,
                max_clause_len=max_clause_len,
                max_know_streams=self.max_know_streams,
            )
            if new_t != t_str:
                self.regrounded_count += 1
            t_str, explicit_know = new_t, new_k

        effective_min_len = min(self.min_target_len, 15) if self.regrounded_count > 0 or domain == "encyclopedia_rag" else self.min_target_len
        if effective_min_len > 0 and len(t_str) < effective_min_len:
            return None
        return p_str, t_str, explicit_know, domain

    def _extract_record_from_dict(
        self, item: Dict[str, Any]
    ) -> Optional[Tuple[str, str, Optional[List[str]], str]]:
        if not isinstance(item, dict):
            return None

        domain = str(item.get("domain", "general") or "general").strip()
        if self.active_domains is not None and domain not in self.active_domains:
            return None

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
                    p_str, t_str = str(user_text).strip(), str(assistant_text).strip()
                    explicit_know = self._parse_knowledge_field(item, None, t_str)
                    return self._finalize_record(p_str, t_str, explicit_know, domain)

        if self.prompt_key and self.target_key:
            if self.prompt_key in item and self.target_key in item:
                p_str, t_str = str(item[self.prompt_key]).strip(), str(item[self.target_key]).strip()
                explicit_know = self._parse_knowledge_field(item, self.prompt_key, t_str)
                return self._finalize_record(p_str, t_str, explicit_know, domain)

        found_prompt_key = None
        if self.prompt_key and self.prompt_key in item:
            found_prompt_key = self.prompt_key
        else:
            for k in self.PROMPT_CANDIDATE_KEYS + ["input", "context", "source"]:
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
            p_str, t_str = str(item[found_prompt_key]).strip(), str(item[found_target_key]).strip()
            explicit_know = self._parse_knowledge_field(item, found_prompt_key, t_str)
            return self._finalize_record(p_str, t_str, explicit_know, domain)

        return None

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
                    rec = self._extract_record_from_dict(obj)
                    if rec:
                        self.samples.append(rec)
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
                        rec = self._extract_record_from_dict(obj)
                        if rec:
                            self.samples.append(rec)
            except Exception as e:
                if self.verbose:
                    print(f"⚠️ Error reading JSON file {file_path}: {e}")

    def _load_parquet(self, file_path: str):
        try:
            import pyarrow.parquet as pq
            table = pq.read_table(file_path)
            col_names = table.column_names
            prompt_col = self.prompt_key or next(
                (c for c in self.PROMPT_CANDIDATE_KEYS + ["input", "context", "source"] if c in col_names),
                None,
            )
            target_col = self.target_key or next(
                (c for c in self.TARGET_CANDIDATE_KEYS if c in col_names),
                None,
            )
            know_col = self.knowledge_key if self.knowledge_key in col_names else None
            domain_col = "domain" if "domain" in col_names else None

            if not prompt_col or not target_col:
                if self.verbose:
                    print(f"⚠️ Parquet file {file_path} missing prompt/target columns (available: {col_names})")
                return

            prompts = table[prompt_col].to_pylist()
            targets = table[target_col].to_pylist()
            knows = table[know_col].to_pylist() if know_col else [None] * len(prompts)
            domains = table[domain_col].to_pylist() if domain_col else ["general"] * len(prompts)

            for p, t, k, d in zip(prompts, targets, knows, domains):
                if self.max_samples and len(self.samples) >= self.max_samples:
                    break
                dom_str = str(d or "general").strip()
                if self.active_domains is not None and dom_str not in self.active_domains:
                    continue
                if p and t:
                    t_str = str(t).strip()
                    item_dict: Dict[str, Any] = {
                        "prompt": str(p).strip(),
                        "target": t_str,
                        "domain": dom_str,
                    }
                    if k is not None:
                        item_dict[self.knowledge_key] = k
                    explicit_know = self._parse_knowledge_field(item_dict, prompt_col, t_str)
                    rec = self._finalize_record(str(p).strip(), t_str, explicit_know, dom_str)
                    if rec:
                        self.samples.append(rec)
        except Exception as e:
            if self.verbose:
                print(f"⚠️ Error reading Parquet file {file_path}: {e}")

    def _build_clause_pool(self):
        max_clause_len = max(12, self.know_seq_len - 2)
        for sample_idx, (_, target_text, explicit_know, domain) in enumerate(self.samples):
            if explicit_know:
                for k_str in explicit_know:
                    c_clean = self._strip_artificial_tags(k_str)[:max_clause_len]
                    if len(c_clean) >= 6:
                        entry = (sample_idx, c_clean)
                        self.clause_pool.append(entry)
                        self.domain_clause_pool.setdefault(domain, []).append(entry)
            clauses = extract_factual_clauses(target_text, min_len=12, max_len=max_clause_len)
            for c in clauses:
                c_clean = self._strip_artificial_tags(c)
                if len(c_clean) >= 12:
                    entry = (sample_idx, c_clean)
                    self.clause_pool.append(entry)
                    self.domain_clause_pool.setdefault(domain, []).append(entry)

    def _sample_distractors(
        self,
        rng: random.Random,
        idx: int,
        domain: str,
        num_neg: int,
        k_plus: List[str],
        effective_target: str,
        original_target: str,
        max_clause_len: int,
    ) -> List[str]:
        k_minus: List[str] = []
        if num_neg <= 0:
            return k_minus

        forbidden = set(k_plus) | {effective_target.strip(), original_target.strip()}
        domain_candidates = self.domain_clause_pool.get(domain, [])
        for pool in (domain_candidates, self.clause_pool):
            if not pool or len(k_minus) >= num_neg:
                continue
            attempts = 0
            while len(k_minus) < num_neg and attempts < 24:
                attempts += 1
                s_idx, cand_str = rng.choice(pool)
                if (
                    s_idx != idx
                    and cand_str not in forbidden
                    and cand_str not in effective_target
                    and effective_target not in cand_str
                    and cand_str not in original_target
                    and cand_str not in k_minus
                ):
                    k_minus.append(cand_str)

        if len(k_minus) < num_neg:
            for fb in self._FALLBACK_DISTRACTORS:
                if len(k_minus) >= num_neg:
                    break
                fb_s = fb[:max_clause_len]
                if (
                    fb_s not in forbidden
                    and fb_s not in effective_target
                    and effective_target not in fb_s
                    and fb_s not in k_minus
                ):
                    k_minus.append(fb_s)
        return k_minus

    def _synthesize_disjoint_knowledge(
        self, idx: int, prompt_text: str, target_text: str, domain: str
    ) -> Tuple[str, List[str], str]:
        if self.max_know_streams <= 0:
            return prompt_text, [], target_text

        rng = random.Random(self.seed * 1000003 + idx)

        if rng.random() < 0.25:
            return prompt_text, [], target_text

        max_clause_len = max(12, self.know_seq_len - 2)
        max_pos_allowed = max(1, min(2, self.max_know_streams - 1 if self.max_know_streams > 1 else 1))

        effective_prompt = prompt_text
        effective_target = target_text
        k_plus: List[str] = []

        structured = split_structured_prompt(
            prompt_text,
            max_know_streams=max_pos_allowed,
            max_clause_len=max_clause_len,
        )
        if structured is not None:
            cand_prompt, cand_k_plus = structured
            cand_k_plus = [
                k for k in cand_k_plus
                if k and is_non_cheating_knowledge(k, effective_target)
            ]
            if cand_k_plus:
                effective_prompt = cand_prompt
                k_plus = cand_k_plus[:max_pos_allowed]

        if not k_plus:
            num_pos_target = min(rng.randint(1, 2), max_pos_allowed)
            k_plus, effective_target = disjoint_split_target_clauses(
                target_text,
                max_pos=num_pos_target,
                max_clause_len=max_clause_len,
                min_remaining_len=12,
            )

        remaining_slots = self.max_know_streams - len(k_plus)
        num_neg = min(rng.randint(1, 2), remaining_slots) if remaining_slots > 0 else 0
        k_minus = self._sample_distractors(
            rng=rng,
            idx=idx,
            domain=domain,
            num_neg=num_neg,
            k_plus=k_plus,
            effective_target=effective_target,
            original_target=target_text,
            max_clause_len=max_clause_len,
        )

        combined = [
            k for k in (k_plus + k_minus)[: self.max_know_streams]
            if k and is_non_cheating_knowledge(k, effective_target)
        ]
        rng.shuffle(combined)
        return effective_prompt, combined, effective_target

    def __len__(self) -> int:
        return len(self.samples)

    def _pad_trunc(self, ids: List[int], target_len: int) -> List[int]:
        if len(ids) > target_len:
            return ids[:target_len]
        return ids + [self.pad_token_id] * (target_len - len(ids))

    def _encode_pair_stream(self, text: str) -> List[int]:
        try:
            ids = list(self.tokenizer.encode(text, add_special_tokens=True))
        except TypeError:
            ids = list(self.tokenizer.encode(text))
        if self.eos_token_id is not None and (not ids or ids[-1] != self.eos_token_id):
            ids.append(self.eos_token_id)
        return ids

    def _encode_knowledge_stream(self, k_str: str) -> List[int]:
        ids = list(self.tokenizer.encode(k_str))
        if self.eos_token_id is not None and (not ids or ids[-1] != self.eos_token_id):
            ids.append(self.eos_token_id)
        return self._pad_trunc(ids, self.know_seq_len)

    def __getitem__(
        self, idx: int
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        prompt_text, target_text, explicit_know, domain = self.samples[idx]

        if explicit_know is not None:
            effective_prompt = prompt_text
            effective_target = target_text
            active_know_strs = [
                k for k in explicit_know[: self.max_know_streams]
                if k and is_non_cheating_knowledge(k, effective_target)
            ]
        elif self.auto_extract_knowledge:
            effective_prompt, active_know_strs, effective_target = self._synthesize_disjoint_knowledge(
                idx, prompt_text, target_text, domain
            )
        else:
            effective_prompt = prompt_text
            effective_target = target_text
            active_know_strs = []

        w_context_ids = self._pad_trunc(
            self._encode_pair_stream(effective_prompt),
            self.context_len,
        )

        w_know_matrix: List[List[int]] = []
        for m in range(self.max_know_streams):
            if m < len(active_know_strs) and active_know_strs[m]:
                w_know_matrix.append(self._encode_knowledge_stream(active_know_strs[m]))
            else:
                w_know_matrix.append([self.pad_token_id] * self.know_seq_len)

        w_target_ids = self._pad_trunc(
            self._encode_pair_stream(effective_target),
            self.seq_len,
        )

        w_ctx_t = torch.tensor(w_context_ids, dtype=torch.long)
        w_know_t = torch.tensor(w_know_matrix, dtype=torch.long)
        w_tgt_t = torch.tensor(w_target_ids, dtype=torch.long)

        if self.return_domain_id:
            domain_id = DOMAIN_TO_ID.get(domain, DOMAIN_TO_ID["general"])
            return w_ctx_t, w_know_t, w_tgt_t, torch.tensor(domain_id, dtype=torch.long)

        return w_ctx_t, w_know_t, w_tgt_t


BicameralKnowledgeDataset = MultiStreamKnowledgeDataset


def create_knowledge_dataloader(
    data_path: Union[str, List[str]],
    tokenizer: Optional[CharTokenizer] = None,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    **dataset_kwargs,
) -> Tuple[MultiStreamKnowledgeDataset, DataLoader]:
    dataset = MultiStreamKnowledgeDataset(data_path=data_path, tokenizer=tokenizer, **dataset_kwargs)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataset, dataloader


class BicameralClauseChainDataset(Dataset):
    """
    Stage 2 & Stage 3 Multi-Clause Progressive Discourse & Candidate Knowledge Pool Dataset (#027):
      - Builds multi-clause progressive discourse trajectories (`w_discourse`, `discourse_valid_mask`),
        goal question clause sequences (`w_goal`, `goal_valid_mask`), and candidate knowledge pools
        (`w_cand`, `cand_pos_mask`, `cand_valid_mask` interleaving `K+` and `K-`) for training and
        evaluating `MultiStreamRightBrain` + `KnowledgeSelector`.
      - Supports loading from:
          * Dialogue JSONL files (`data/Basic_ZH/moss_dialogues.jsonl`, `data/Basic_ZH/qa_dialogues.jsonl`, etc.)
          * Knowledge QA JSONL files (`data/Knowledge_ZH/bicameral_knowledge_qa.jsonl`, etc.)
          * Comma-separated list or multiple paths (automatically skipping non-existent optional paths
            if at least one valid path exists).
    """

    PROMPT_CANDIDATE_KEYS = [
        "question", "prompt", "instruction", "query", "user", "human",
        "question_chinese", "question_zh", "input", "text_a",
    ]
    TARGET_CANDIDATE_KEYS = [
        "answer", "target", "output", "response", "assistant", "bot",
        "answer_zh", "completion", "text_b",
    ]

    _FALLBACK_DISTRACTOR_CLAUSES = [
        "地球大气层主要由氮气和氧气组成，",
        "其中氮气约占百分之七十八。",
        "计算机操作系统负责管理硬件资源，",
        "并为各类应用程序提供运行环境。",
        "光合作用是绿色植物吸收光能的过程，",
        "能够将二氧化碳和水转化为有机物。",
        "牛顿第一定律又称惯性定律，",
        "指出物体不受外力时保持静止或匀速运动。",
        "水在标准大气压下的沸点是一百摄氏度，",
        "凝固点则为零摄氏度。",
        "太阳系由太阳和八大行星共同组成，",
        "木星是太阳系中体积最大的行星。",
    ]

    _BUILTIN_FALLBACK_RECORDS = [
        {
            "domain": "dialogue",
            "question": "你能简要介绍一下人工智能的发展阶段吗？为什么近年来发展这么快？",
            "answer": "人工智能主要经历了符号主义阶段，随后的统计机器学习阶段，以及当前的深度学习与大模型阶段。近年来由于算力硬件的飞跃，加上海量互联网数据的积累，使得模型展现出强大的泛化能力。",
            "positive_knowledge": [
                "人工智能经历了符号主义、统计学习与深度学习三个阶段。",
                "算力硬件飞跃与海量数据积累推动了近年来大模型的突破。",
            ],
            "negative_distractors": [
                "光合作用是植物利用叶绿素将二氧化碳转化为有机物的过程。",
                "唐代诗人李白被称为诗仙，代表作有《将进酒》。",
            ],
        },
        {
            "domain": "encyclopedia_rag",
            "question": "请根据天文常识回答，八大行星中体积最大的是哪一颗？它有哪些显著特征？",
            "answer": "太阳系八大行星中体积最大的是木星，它属于典型的气态巨行星。木星表面拥有著名的大红斑风暴，并且周围环绕着数十颗天然卫星。",
            "positive_knowledge": [
                "木星是太阳系八大行星中体积和质量最大的气态巨行星。",
                "木星表面有持续数百年的大红斑反气旋风暴。",
            ],
            "negative_distractors": [
                "Python语言由吉多·范罗苏姆于一九八九年底发明。",
                "勾股定理指出直角三角形两直角边的平方和等于斜边平方。",
            ],
        },
        {
            "domain": "code_api",
            "question": "在Python的PyTorch框架中，如何使用AdamW优化器进行梯度清零与参数更新？",
            "answer": "首先调用optimizer.zero_grad()清空历史梯度，接着通过loss.backward()计算反向传播梯度，最后执行optimizer.step()完成模型参数更新。",
            "positive_knowledge": [
                "optimizer.zero_grad()用于清除上一轮残留的参数梯度。",
                "loss.backward()计算梯度后需调用optimizer.step()更新权重。",
            ],
            "negative_distractors": [
                "苏轼是北宋著名文学家，豪放派词人的代表人物。",
                "赤道是地球表面随地球自转产生的最长纬线圈。",
            ],
        },
        {
            "domain": "math_rule",
            "question": "已知一个长方形的长为十二厘米，宽为八厘米，请计算它的周长和面积分别是多少？",
            "answer": "长方形的周长等于长加宽之和乘以二，即二十乘以二等于四十厘米。长方形的面积等于长乘以宽，即十二乘以八等于九十六平方厘米。",
            "positive_knowledge": [
                "长方形周长公式为(长+宽)×2，面积公式为长×宽。",
                "12加8等于20，12乘8等于96。",
            ],
            "negative_distractors": [
                "超声波是指频率高于两万赫兹的机械波。",
                "长江是中国第一长河，发源于青藏高原唐古拉山脉。",
            ],
        },
        {
            "domain": "classical_quote",
            "question": "王勃在《滕王阁序》中描绘秋水与长天一色的千古名句是什么？表达了怎样的意境？",
            "answer": "千古名句是落霞与孤鹜齐飞，秋水共长天一色。这两句将绚丽的晚霞、飞翔的水鸟与辽阔的江天融为一体，展现出壮美悠远的秋日画卷。",
            "positive_knowledge": [
                "《滕王阁序》名句：落霞与孤鹜齐飞，秋水共长天一色。",
                "王勃借晚霞孤鹜与秋水长天描绘出壮阔澄澈的江天秋景。",
            ],
            "negative_distractors": [
                "关系型数据库通过SQL语句执行表连接与聚合查询。",
                "线粒体是真核细胞中进行有氧呼吸的主要场所。",
            ],
        },
        {
            "domain": "grounded_instruct",
            "question": "为什么高海拔地区烧开水时，水沸腾的温度会低于一百摄氏度？日常做饭需要注意什么？",
            "answer": "因为高海拔地区的空气稀薄，大气压强明显低于标准大气压，而液体的沸点会随气压降低而下降。因此在高原地区做饭建议使用高压锅，以提升锅内气压并加快食物熟透。",
            "positive_knowledge": [
                "高海拔地区大气压低于标准大气压，导致水的沸点低于100℃。",
                "使用高压锅可以增大锅内气压，从而提高水的沸腾温度。",
            ],
            "negative_distractors": [
                "二进制数系统仅使用零和一两个数字来表示数值。",
                "文艺复兴运动起源于十四世纪的意大利佛罗伦萨。",
            ],
        },
    ]

    def __init__(
        self,
        data_path: Union[str, List[str]],
        tokenizer: Optional[CharTokenizer] = None,
        seq_len: int = 32,
        max_goal_clauses: int = 4,
        max_cand_clauses: int = 8,
        max_discourse_clauses: int = 8,
        min_clause_chars: int = 6,
        max_clause_chars: int = 28,
        max_samples: Optional[int] = None,
        seed: int = 42,
        verbose: bool = True,
    ):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer if tokenizer is not None else CharTokenizer()
        self.seq_len = int(seq_len)
        self.max_goal_clauses = int(max_goal_clauses)
        self.max_cand_clauses = int(max_cand_clauses)
        self.max_discourse_clauses = max(2, int(max_discourse_clauses))
        self.min_clause_chars = int(min_clause_chars)
        self.max_clause_chars = int(max_clause_chars)
        self.max_samples = max_samples
        self.seed = int(seed)
        self.verbose = verbose

        self.pad_token_id = getattr(self.tokenizer, "pad_token_id", 0)
        self.eos_token_id = getattr(self.tokenizer, "eos_token_id", None)

        # Each record in `self.records`:
        # {
        #   "q_clauses": List[str],
        #   "a_clauses": List[str],
        #   "pos_clauses": List[str],
        #   "neg_clauses": List[str],
        #   "has_explicit_know": bool,
        #   "question_text": str,
        #   "answer_text": str,
        #   "domain": str,
        # }
        self.records: List[Dict[str, Any]] = []
        self.samples: List[Tuple[str, str, Optional[List[str]], str]] = []
        self.global_distractor_pool: List[Tuple[int, str]] = []

        self._load_data(data_path)
        self._build_global_distractor_pool()

        if self.verbose:
            print(
                f"✅ BicameralClauseChainDataset loaded {len(self.records)} multi-clause records "
                f"(global distractor pool: {len(self.global_distractor_pool)} clauses, "
                f"G={self.max_goal_clauses}, C={self.max_cand_clauses}, T={self.max_discourse_clauses}, L={self.seq_len})."
            )

    def _collect_file_paths(self, data_path: Union[str, List[str]]) -> List[str]:
        if isinstance(data_path, str):
            raw_paths = [p.strip() for p in data_path.split(",") if p.strip()]
        else:
            raw_paths = []
            for item in data_path:
                if isinstance(item, str):
                    raw_paths.extend([p.strip() for p in item.split(",") if p.strip()])

        files: List[str] = []
        for path in raw_paths:
            if os.path.isdir(path):
                for ext in ("*.jsonl", "*.json", "*.parquet"):
                    files.extend(glob.glob(os.path.join(path, ext)))
            elif "*" in path or "?" in path:
                files.extend(glob.glob(path))
            elif os.path.isfile(path):
                files.append(path)
            else:
                if self.verbose:
                    print(f"ℹ️ Skipping non-existent optional dataset path: {path}")
        return sorted(list(set(files)))

    def _parse_raw_item(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not isinstance(item, dict):
            return None

        domain = str(item.get("domain", item.get("category", "general")) or "general").strip()
        question_str = ""
        answer_str = ""

        # 1. Check OpenAI / ShareGPT conversations
        for conv_key in ("messages", "conversations"):
            if conv_key in item and isinstance(item[conv_key], list):
                for turn in item[conv_key]:
                    if not isinstance(turn, dict):
                        continue
                    role = str(turn.get("role", turn.get("from", ""))).lower()
                    content = str(turn.get("content", turn.get("value", ""))).strip()
                    if role in ("user", "human") and content:
                        question_str = content
                    elif role in ("assistant", "bot", "gpt", "model") and content:
                        answer_str = content
                if question_str and answer_str:
                    break

        # 2. Check standard prompt/question and target/answer keys
        if not question_str:
            for k in self.PROMPT_CANDIDATE_KEYS:
                val = item.get(k)
                if isinstance(val, str) and val.strip():
                    question_str = val.strip()
                    break

        if not answer_str:
            for k in self.TARGET_CANDIDATE_KEYS:
                val = item.get(k)
                if isinstance(val, str) and val.strip():
                    answer_str = val.strip()
                    break

        if not question_str or not answer_str:
            return None

        # Check explicit positive_knowledge / negative_distractors (e.g., from bicameral_knowledge_qa.jsonl)
        pos_raw = item.get("positive_knowledge")
        if pos_raw is None:
            pos_raw = item.get("knowledge")
        neg_raw = item.get("negative_distractors")
        if neg_raw is None:
            neg_raw = item.get("distractors")

        # For multi-line grounded prompts in omni_knowledge_corpus.jsonl ("Context/Ref...\nQuestion..."),
        # separate the reference passage lines[:-1] into K+ (pos_raw) and keep lines[-1] as the goal question G.
        if pos_raw is None and "\n" in question_str:
            lines = [ln.strip() for ln in question_str.splitlines() if ln.strip()]
            if len(lines) >= 2:
                q_tail_clauses = _split_into_short_clauses(
                    lines[-1], min_chars=4, max_chars=self.max_clause_chars
                )
                ref_head_clauses = _extract_clauses_from_text(
                    lines[:-1], min_chars=4, max_chars=self.max_clause_chars
                )
                if q_tail_clauses and ref_head_clauses:
                    pos_raw = lines[:-1]
                    question_str = lines[-1]

        # Split question into short clauses (min_chars=4) and answer into short clauses (min_chars=self.min_clause_chars)
        q_clauses = _split_into_short_clauses(
            question_str, min_chars=4, max_chars=self.max_clause_chars
        )
        a_clauses = _split_into_short_clauses(
            answer_str, min_chars=self.min_clause_chars, max_chars=self.max_clause_chars
        )
        if not q_clauses or not a_clauses:
            return None

        has_explicit_know = pos_raw is not None
        pos_clauses = _extract_clauses_from_text(
            pos_raw, min_chars=4, max_chars=self.max_clause_chars
        ) if pos_raw is not None else []
        neg_clauses = _extract_clauses_from_text(
            neg_raw, min_chars=4, max_chars=self.max_clause_chars
        ) if neg_raw is not None else []

        return {
            "q_clauses": q_clauses,
            "a_clauses": a_clauses,
            "pos_clauses": pos_clauses,
            "neg_clauses": neg_clauses,
            "has_explicit_know": has_explicit_know,
            "question_text": question_str,
            "answer_text": answer_str,
            "domain": domain,
        }

    def _append_record(self, rec: Dict[str, Any]):
        self.records.append(rec)
        self.samples.append((
            rec["question_text"],
            rec["answer_text"],
            rec["pos_clauses"] if rec["pos_clauses"] else None,
            rec["domain"],
        ))

    def _load_data(self, data_path: Union[str, List[str]]):
        files = self._collect_file_paths(data_path)
        for file_path in files:
            if self.max_samples and len(self.records) >= self.max_samples:
                break
            if self.verbose:
                print(f"📖 Reading clause-chain dataset file: {file_path} ...")

            if file_path.endswith(".jsonl"):
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if self.max_samples and len(self.records) >= self.max_samples:
                            break
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                            rec = self._parse_raw_item(obj)
                            if rec is not None:
                                self._append_record(rec)
                        except Exception:
                            continue
            elif file_path.endswith(".json"):
                with open(file_path, "r", encoding="utf-8") as f:
                    try:
                        data = json.load(f)
                        if isinstance(data, list):
                            for obj in data:
                                if self.max_samples and len(self.records) >= self.max_samples:
                                    break
                                rec = self._parse_raw_item(obj)
                                if rec is not None:
                                    self._append_record(rec)
                    except Exception:
                        pass

        # Fallback to built-in multi-domain records if external data symlink is unmounted in local/sandbox test
        if not self.records:
            if self.verbose:
                print(
                    f"⚠️ Warning: No dataset files found at '{data_path}'; "
                    f"populating built-in multi-domain clause-chain records for verification."
                )
            target_n = self.max_samples if (self.max_samples and self.max_samples > 0) else 24
            for i in range(target_n):
                raw_item = self._BUILTIN_FALLBACK_RECORDS[i % len(self._BUILTIN_FALLBACK_RECORDS)]
                rec = self._parse_raw_item(raw_item)
                if rec is not None:
                    self._append_record(rec)

    def _build_global_distractor_pool(self):
        for r_idx, rec in enumerate(self.records):
            for c in rec["pos_clauses"] + rec["a_clauses"] + rec["neg_clauses"]:
                if len(c) >= self.min_clause_chars:
                    self.global_distractor_pool.append((r_idx, c))
        if not self.global_distractor_pool:
            for c in self._FALLBACK_DISTRACTOR_CLAUSES:
                self.global_distractor_pool.append((-1, c))

    def _encode_clause_tokens(self, clause_str: str) -> List[int]:
        ids = list(self.tokenizer.encode(clause_str, add_special_tokens=True))
        if len(ids) > self.seq_len:
            ids = ids[: self.seq_len]
            if self.eos_token_id is not None:
                ids[-1] = self.eos_token_id
        elif len(ids) < self.seq_len:
            ids = ids + [self.pad_token_id] * (self.seq_len - len(ids))
        return ids

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        q_clauses: List[str] = rec["q_clauses"]
        a_clauses: List[str] = rec["a_clauses"]
        pad_row = [self.pad_token_id] * self.seq_len

        # 1. Goal Stream (`w_goal`, `goal_valid_mask`): up to `max_goal_clauses` from `q_clauses[-max_goal_clauses:]`
        goal_selected = q_clauses[-self.max_goal_clauses :]
        w_goal_list: List[List[int]] = []
        goal_valid_list: List[bool] = []
        for g_idx in range(self.max_goal_clauses):
            if g_idx < len(goal_selected):
                w_goal_list.append(self._encode_clause_tokens(goal_selected[g_idx]))
                goal_valid_list.append(True)
            else:
                w_goal_list.append(list(pad_row))
                goal_valid_list.append(False)

        # 2. Discourse Stream (`w_discourse`, `discourse_valid_mask`):
        #    Clause 0 is `q_clauses[-1]` (lead departure clause), Clauses 1..M are `a_clauses[:M]`
        max_ans_clauses = self.max_discourse_clauses - 1
        ans_selected = a_clauses[:max_ans_clauses]
        discourse_clauses = [q_clauses[-1]] + ans_selected

        w_discourse_list: List[List[int]] = []
        discourse_valid_list: List[bool] = []
        for t_idx in range(self.max_discourse_clauses):
            if t_idx < len(discourse_clauses):
                w_discourse_list.append(self._encode_clause_tokens(discourse_clauses[t_idx]))
                discourse_valid_list.append(True)
            else:
                w_discourse_list.append(list(pad_row))
                discourse_valid_list.append(False)

        # 3. Candidate Knowledge Pool (`w_cand`, `cand_pos_mask`, `cand_valid_mask`)
        max_pos_slots = max(1, self.max_cand_clauses // 2)
        pos_pool: List[str] = []
        seen_pos = set()

        if rec["has_explicit_know"] and rec["pos_clauses"]:
            raw_pos_source = rec["pos_clauses"] + q_clauses
        else:
            # Strictly avoid leaking target answer clauses (a_clauses) into K+ candidate pool
            raw_pos_source = list(q_clauses)

        for c in raw_pos_source:
            c_clean = c.strip()
            if c_clean and c_clean not in seen_pos:
                seen_pos.add(c_clean)
                pos_pool.append(c_clean)
                if len(pos_pool) >= max_pos_slots:
                    break

        if not pos_pool:
            pos_pool = [q_clauses[-1]]
            seen_pos.add(q_clauses[-1])

        rng = random.Random(self.seed + idx)
        neg_needed = max(0, self.max_cand_clauses - len(pos_pool))
        neg_pool: List[str] = []
        forbidden = seen_pos | set(q_clauses) | set(a_clauses)

        for nc in rec["neg_clauses"]:
            nc_clean = nc.strip()
            if nc_clean and nc_clean not in forbidden and nc_clean not in neg_pool:
                neg_pool.append(nc_clean)
                if len(neg_pool) >= neg_needed:
                    break

        attempts = 0
        while len(neg_pool) < neg_needed and attempts < 32 and self.global_distractor_pool:
            attempts += 1
            other_idx, cand_c = rng.choice(self.global_distractor_pool)
            if other_idx != idx and cand_c not in forbidden and cand_c not in neg_pool:
                neg_pool.append(cand_c)

        if len(neg_pool) < neg_needed:
            for fb_c in self._FALLBACK_DISTRACTOR_CLAUSES:
                if len(neg_pool) >= neg_needed:
                    break
                fb_short = fb_c[: self.max_clause_chars]
                if fb_short not in forbidden and fb_short not in neg_pool:
                    neg_pool.append(fb_short)

        cand_entries: List[Tuple[str, float, bool]] = (
            [(c, 1.0, True) for c in pos_pool]
            + [(c, 0.0, True) for c in neg_pool]
        )[: self.max_cand_clauses]

        # Deterministic per-index shuffle of active candidate entries so K+ and K- have no positional bias
        rng.shuffle(cand_entries)

        while len(cand_entries) < self.max_cand_clauses:
            cand_entries.append(("", 0.0, False))

        w_cand_list: List[List[int]] = []
        cand_pos_list: List[float] = []
        cand_valid_list: List[bool] = []
        for c_str, pos_label, is_valid in cand_entries:
            if is_valid and c_str:
                w_cand_list.append(self._encode_clause_tokens(c_str))
                cand_pos_list.append(float(pos_label))
                cand_valid_list.append(True)
            else:
                w_cand_list.append(list(pad_row))
                cand_pos_list.append(0.0)
                cand_valid_list.append(False)

        return {
            "w_goal": torch.tensor(w_goal_list, dtype=torch.long),
            "goal_valid_mask": torch.tensor(goal_valid_list, dtype=torch.bool),
            "w_cand": torch.tensor(w_cand_list, dtype=torch.long),
            "cand_pos_mask": torch.tensor(cand_pos_list, dtype=torch.float32),
            "cand_valid_mask": torch.tensor(cand_valid_list, dtype=torch.bool),
            "w_discourse": torch.tensor(w_discourse_list, dtype=torch.long),
            "discourse_valid_mask": torch.tensor(discourse_valid_list, dtype=torch.bool),
            "question_text": rec["question_text"],
            "answer_text": rec["answer_text"],
            "domain": rec["domain"],
        }


def create_clause_chain_dataloader(
    data_path: Union[str, List[str]],
    tokenizer: Optional[CharTokenizer] = None,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    seq_len: int = 32,
    max_goal_clauses: int = 4,
    max_cand_clauses: int = 8,
    max_discourse_clauses: int = 8,
    min_clause_chars: int = 6,
    max_clause_chars: int = 28,
    max_samples: Optional[int] = None,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[BicameralClauseChainDataset, DataLoader]:
    """
    Convenience factory returning `(dataset, dataloader)` for `BicameralClauseChainDataset`.
    """
    dataset = BicameralClauseChainDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        seq_len=seq_len,
        max_goal_clauses=max_goal_clauses,
        max_cand_clauses=max_cand_clauses,
        max_discourse_clauses=max_discourse_clauses,
        min_clause_chars=min_clause_chars,
        max_clause_chars=max_clause_chars,
        max_samples=max_samples,
        seed=seed,
        verbose=verbose,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    return dataset, dataloader
