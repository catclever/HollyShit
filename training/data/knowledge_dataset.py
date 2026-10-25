import os
import re
import glob
import json
import random
from typing import List, Union, Optional, Tuple, Dict, Any
import torch
from torch.utils.data import Dataset


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
    Partitions `target_text` into two strictly DISJOINT parts:
      - `k_plus`: 1-2 leading premise/background clauses extracted as positive knowledge (K+)
      - `new_target`: the remaining continuation text with `k_plus` completely removed.
    Guarantees 0% verbatim overlap between `k_plus` and `new_target`.
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
        # Decide whether we can take 1 or 2 premise sentences while leaving enough continuation
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
                if k and k not in rem_text and rem_text not in k
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
                if len(k) >= 6 and k not in rem_text and rem_text not in k
            ]
            if k_plus:
                return k_plus, rem_text

    # 3. If indivisible, return empty K+ and intact target so zero overlap is still guaranteed
    return [], text


class MultiStreamKnowledgeDataset(Dataset):
    """
    Phase 5 Multi-Stream Knowledge Dataset for Dual-Brain Mamba Planner + Flow Matching.

    Returns a 3-tuple (w_context, w_knowledge, w_target):
      - w_context:   LongTensor of shape (context_len,)
      - w_knowledge: LongTensor of shape (max_know_streams, know_seq_len)
      - w_target:    LongTensor of shape (seq_len,)

    Enforces strict 0% verbatim overlap between `knowledge` streams and `target`:
      - Case A (Structured / Multi-Field Records): Extracts `input`/`context`/`schema`/`equation`/`source`
        into `knowledge` (K+), `instruction`/`question` into `prompt`, and `output`/`target` into `target`.
      - Case B (Disjoint Clause Split for Plain Pair Records): Partitions `target` (or multi-sentence `prompt`)
        into disjoint premise clauses (K+) and continuation clauses (`target`), removing K+ from `target`.
      - All knowledge strings are kept completely natural without artificial `【...】` tags.
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
        min_target_len: int = 30,
        auto_extract_knowledge: bool = True,
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
        self.seed = seed
        self.verbose = verbose

        self.pad_token_id = getattr(self.tokenizer, "pad_token_id", 0)
        self.eos_token_id = getattr(self.tokenizer, "eos_token_id", None)

        # Each record: (prompt_str, target_str, explicit_knowledge_or_None, domain_str)
        self.samples: List[Tuple[str, str, Optional[List[str]], str]] = []
        self.clause_pool: List[Tuple[int, str]] = []
        self.domain_clause_pool: Dict[str, List[Tuple[int, str]]] = {}

        self._load_data(data_path)

        if self.auto_extract_knowledge:
            self._build_clause_pool()

        if self.verbose:
            print(
                f"✅ Loaded {len(self.samples)} multi-stream samples "
                f"(clause pool: {len(self.clause_pool)} clauses across {len(self.domain_clause_pool)} domains)."
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
                cleaned = [k for k in cleaned if k and k not in target_text and target_text not in k]
                return cleaned[: self.max_know_streams]
            if isinstance(raw_know, str):
                s = self._strip_artificial_tags(raw_know)
                if not s:
                    return None
                if s not in target_text and target_text not in s:
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
                if s and s not in target_text and target_text not in s:
                    clauses = extract_factual_clauses(s, min_len=6, max_len=max_clause_len)
                    if not clauses:
                        clauses = [s[:max_clause_len]]
                    clauses = [c for c in clauses if c not in target_text and target_text not in c]
                    if clauses:
                        return clauses[: self.max_know_streams]

        return None

    def _extract_record_from_dict(
        self, item: Dict[str, Any]
    ) -> Optional[Tuple[str, str, Optional[List[str]], str]]:
        if not isinstance(item, dict):
            return None

        domain = str(item.get("domain", "general") or "general").strip()

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
                    if self.min_target_len > 0 and len(t_str) < self.min_target_len:
                        return None
                    explicit_know = self._parse_knowledge_field(item, None, t_str)
                    return p_str, t_str, explicit_know, domain

        # 2. Check explicitly provided keys
        if self.prompt_key and self.target_key:
            if self.prompt_key in item and self.target_key in item:
                p_str, t_str = str(item[self.prompt_key]).strip(), str(item[self.target_key]).strip()
                if self.min_target_len > 0 and len(t_str) < self.min_target_len:
                    return None
                explicit_know = self._parse_knowledge_field(item, self.prompt_key, t_str)
                return p_str, t_str, explicit_know, domain

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
            if self.min_target_len > 0 and len(t_str) < self.min_target_len:
                return None
            explicit_know = self._parse_knowledge_field(item, found_prompt_key, t_str)
            return p_str, t_str, explicit_know, domain

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
                if p and t:
                    t_str = str(t).strip()
                    if self.min_target_len > 0 and len(t_str) < self.min_target_len:
                        continue
                    item_dict: Dict[str, Any] = {
                        "prompt": str(p).strip(),
                        "target": t_str,
                        "domain": str(d or "general"),
                    }
                    if k is not None:
                        item_dict[self.knowledge_key] = k
                    explicit_know = self._parse_knowledge_field(item_dict, prompt_col, t_str)
                    self.samples.append((str(p).strip(), t_str, explicit_know, str(d or "general")))
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
        """
        Deterministically synthesizes (effective_prompt, knowledge_streams, effective_target)
        for plain records without an explicit `knowledge` field, guaranteeing 0% verbatim
        overlap between `knowledge_streams` and `effective_target`:
          - 25% probability: Empty knowledge stream (`[]`), original prompt and target intact.
          - 75% probability:
            * Case A: If `prompt_text` contains structured context/schema/input, extract it into K+
              and keep `target_text` intact.
            * Case B: Otherwise, partition `target_text` into disjoint premise clauses (K+) and
              continuation clauses (`effective_target`), removing K+ from `effective_target`.
            * Mix 1-2 same-domain distractor clauses K- and shuffle `[K+, K-]`.
        """
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
                if k and k not in effective_target and effective_target not in k
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
            if k and k not in effective_target and effective_target not in k
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

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prompt_text, target_text, explicit_know, domain = self.samples[idx]

        if explicit_know is not None:
            effective_prompt = prompt_text
            effective_target = target_text
            active_know_strs = [
                k for k in explicit_know[: self.max_know_streams]
                if k and k not in effective_target and effective_target not in k
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

        return (
            torch.tensor(w_context_ids, dtype=torch.long),
            torch.tensor(w_know_matrix, dtype=torch.long),
            torch.tensor(w_target_ids, dtype=torch.long),
        )
