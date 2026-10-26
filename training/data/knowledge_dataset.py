import os
import re
import glob
import json
import random
from typing import List, Union, Optional, Tuple, Dict, Any
import torch
from torch.utils.data import Dataset


CANONICAL_DOMAINS = [
    "grounded_instruct",
    "code_api",
    "classical_quote",
    "encyclopedia_rag",
    "math_rule",
    "literature_lore",
    "general",
]
DOMAIN_TO_ID = {d: i for i, d in enumerate(CANONICAL_DOMAINS)}
ID_TO_DOMAIN = {i: d for i, d in enumerate(CANONICAL_DOMAINS)}


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
    Uses `is_non_cheating_knowledge(k, rem_text)` so shared entity names or numbers across
    sentences are preserved while preventing full-sentence duplication.
    """
    text = target_text.strip()
    if not text:
        return [], text

    # 1. Try sentence-level split preserving sentence-ending punctuation
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

    # 2. Fallback: split single sentence by comma/semicolon into disjoint prefix (K+) and suffix (target)
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

    # 3. If indivisible, return empty K+ and intact target
    return [], text


def reground_legacy_disjoint_sample(
    target_text: str,
    explicit_know: Optional[List[str]],
    domain: str,
    max_clause_len: int = 126,
    max_know_streams: int = 4,
) -> Tuple[str, Optional[List[str]]]:
    """
    On-the-fly Legacy Disjoint-Split Re-Grounding:
    Detects legacy `encyclopedia_rag` or `literature_lore` records where `explicit_know` was produced
    by disjoint splitting (none of the 3-gram character chunks of `target[:60]` appear in `explicit_know[0]`),
    and automatically re-grounds the sample:
      1. Combines `explicit_know[0]` + `target` back into a continuous passage `full_passage`;
      2. Takes the first complete sentence/clause (`15..85` chars) of `target` as `new_target`;
      3. Takes a wider `80..126` char reference window from `full_passage` covering `new_target` +
         surrounding context as `new_k_plus` (guaranteeing `len(new_target) < 0.80 * len(new_k_plus)`
         and `is_non_cheating_knowledge(new_k_plus, new_target) == True`).
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

    # Combine k0 and target_text back into a continuous passage
    k0_no_end = re.sub(r"[。！？；，,\s]+$", "", k0_raw)
    full_passage = f"{k0_no_end}。{t_raw}"
    if len(full_passage) < 40 and len(explicit_know) > 1:
        k1_no_end = re.sub(r"[。！？；，,\s]+$", "", str(explicit_know[1]).strip())
        if k1_no_end:
            full_passage = f"{full_passage}{k1_no_end}。"

    eff_win_cap = min(max_clause_len, len(full_passage))
    max_tgt_len = min(85, max(16, int(0.72 * eff_win_cap)))

    # Extract first complete sentence/clause (15..max_tgt_len chars) of t_raw as new_target
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

    # Locate new_target_core in full_passage and extract a wider reference window up to max_clause_len
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

    # Final safety guard: ensure is_non_cheating_knowledge(new_k_plus, new_target) holds
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

    When `return_domain_id=False` (default), returns a 3-tuple `(w_context, w_knowledge, w_target)`:
      - w_context:   LongTensor of shape (context_len,)
      - w_knowledge: LongTensor of shape (max_know_streams, know_seq_len)
      - w_target:    LongTensor of shape (seq_len,)
    When `return_domain_id=True`, returns a 4-tuple `(w_context, w_knowledge, w_target, domain_id)`.
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
        tokenizer,
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
        self.tokenizer = tokenizer
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

        # Each record: (prompt_str, target_str, explicit_knowledge_or_None, domain_str)
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

    @staticmethod
    def _strip_artificial_tags(text: str) -> str:
        """Removes any leading artificial bracket tags like `【经典原句】：` if present."""
        return re.sub(r"^\s*【[^】]{1,12}】[：:\s]*", "", text).strip()

    def _parse_knowledge_field(
        self, item: Dict[str, Any], prompt_key_used: Optional[str], target_text: str
    ) -> Optional[List[str]]:
        max_clause_len = max(12, self.know_seq_len - 2)

        # 1. Explicit `self.knowledge_key` field
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

        # 2. Case A: Multi-field record with separate `input`/`context`/`schema`/`equation`/`source` field
        for cand_k in self.SEPARATE_KNOWLEDGE_CANDIDATE_KEYS:
            if cand_k == prompt_key_used:
                continue
            val = item.get(cand_k)
            if not isinstance(val, str):
                continue
            val_s = val.strip()
            # Skip dataset source slugs like "belle_1m" or "wiki_cn" in omni_corpus.jsonl
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
                    p_str, t_str = str(user_text).strip(), str(assistant_text).strip()
                    explicit_know = self._parse_knowledge_field(item, None, t_str)
                    return self._finalize_record(p_str, t_str, explicit_know, domain)

        # 2. Check explicitly provided keys
        if self.prompt_key and self.target_key:
            if self.prompt_key in item and self.target_key in item:
                p_str, t_str = str(item[self.prompt_key]).strip(), str(item[self.target_key]).strip()
                explicit_know = self._parse_knowledge_field(item, self.prompt_key, t_str)
                return self._finalize_record(p_str, t_str, explicit_know, domain)

        # 3. Auto-detect keys from candidates
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

        # 25% probability: Empty Knowledge Stream
        if rng.random() < 0.25:
            return prompt_text, [], target_text

        max_clause_len = max(12, self.know_seq_len - 2)
        max_pos_allowed = max(1, min(2, self.max_know_streams - 1 if self.max_know_streams > 1 else 1))

        effective_prompt = prompt_text
        effective_target = target_text
        k_plus: List[str] = []

        # Case A: Check if prompt contains structured context/reference
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

        # Case B: Disjoint clause split on target_text (removing K+ from effective_target)
        if not k_plus:
            num_pos_target = min(rng.randint(1, 2), max_pos_allowed)
            k_plus, effective_target = disjoint_split_target_clauses(
                target_text,
                max_pos=num_pos_target,
                max_clause_len=max_clause_len,
                min_remaining_len=12,
            )

        # Sample 1-2 negative distractor clauses K- from other samples
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

        # 1. Encode w_context: shape (context_len,)
        w_context_ids = self._pad_trunc(
            self._encode_pair_stream(effective_prompt),
            self.context_len,
        )

        # 2. Encode w_knowledge: shape (max_know_streams, know_seq_len)
        w_know_matrix: List[List[int]] = []
        for m in range(self.max_know_streams):
            if m < len(active_know_strs) and active_know_strs[m]:
                w_know_matrix.append(self._encode_knowledge_stream(active_know_strs[m]))
            else:
                w_know_matrix.append([self.pad_token_id] * self.know_seq_len)

        # 3. Encode w_target: shape (seq_len,)
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
