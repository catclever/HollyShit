"""
=====================================================================================================
Phase 5 · Stage-1 Single-Stream Clause Dataset (+ Knowledge-Selector Pseudo Labels)   [#032 / #043]
=====================================================================================================
Purpose:
  - Builds ONE main clause stream per sample for the Stage-1 single-stream right brain:
        main stream = context/prefix clauses  +  target clauses      (T = max_clauses rows, L = seq_len)
    * QA source:            prefix = multi-turn history (if any) + the WHOLE prompt (material lines included,
                            never split out as "knowledge"); targets = answer clauses.
    * Continuation source:  a random window of contiguous clauses from ONE long document; the first k rows
                            are context, all remaining rows are targets.
  - Emits the stop-head labels (QA only) and the self-supervised Knowledge-Selector labels
    (`ctx_rel`, `ctx_rel_valid`, `sink_label`): candidate pool = every earlier row of the same stream.

Design reference (authoritative):
  /Users/kael/.gemini/Projects/HollyShit/design/tentative/2026-11-02_phase5_staged_training_plan.md (§2, §3, §5)

Caveats (READ before changing / consuming):
  1. `ctx_rel` is an IDF-weighted RARE-SPAN overlap pseudo label (rare char bigrams + digit runs + ASCII words).
     It under-recalls paraphrased reuse (same fact, different wording) and can over-fire on shared rare
     named entities. Treat it as a noisy positive signal, not ground truth.
  2. The rare-bigram threshold is an OCCURRENCE-weighted (df-weighted) IDF quantile, so high-frequency
     function bigrams ("的是", "我们", ...) fall below it. Bigrams never seen in the IDF sample get the
     maximum IDF (= treated as rare).
  3. Continuation windows have NO stop supervision (`stop_valid` all False): a window boundary is not a real end.
     Truncated QA windows keep `stop_valid=True` on all answer rows with label 0 (true end is outside).
  4. `source_id` (0 = qa, 1 = continuation) is for GROUPED LOGGING ONLY. It must NEVER be fed to the model
     (design §0.2: no style/source label inputs; style is an implicit condition of the context).
  5. Clause tokens follow `BicameralClauseChainDataset._encode_clause_tokens`: CharTokenizer
     `add_special_tokens=True` => [BOS] chars [EOS], truncated to L with EOS forced at L-1, PAD-padded.
  6. Index → source mapping is deterministic (Bresenham interleave by `cont_ratio`); per-index content is
     deterministic given (seed, index, epoch). `set_epoch(e)` only rotates which QA record / continuation
     slot an index maps to (default epoch 0); continuation window sampling uses (seed, slot) only.
=====================================================================================================
"""

import os
import re
import glob
import json
import math
import random
import bisect
from typing import List, Union, Optional, Tuple, Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader

from training.core.char_tokenizer import CharTokenizer
from training.data.knowledge_dataset import (
    _split_into_short_clauses,
    _extract_clauses_from_text,
    _normalize_for_bigram_overlap,
    BicameralClauseChainDataset,
)


# Keys that may hold a long document in continuation JSONL files (checked in order).
CONT_TEXT_KEYS = ["text", "content", "completion", "RESPONSE", "response", "body", "article"]

# Extra short-form QA keys (e.g. {"q": ..., "a": ...}) on top of BicameralClauseChainDataset's lists.
_EXTRA_PROMPT_KEYS = ["q"]
_EXTRA_TARGET_KEYS = ["a"]

_USER_ROLES = ("user", "human")
_ASSISTANT_ROLES = ("assistant", "bot", "gpt", "model")

_DIGIT_RUN_RE = re.compile(r"\d+")
_ASCII_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*")

# Interrogative / instruction markers stripped from the tail of QA prefixes when deriving continuation docs (#043).
_QA_TAIL_MARKERS = (
    "请问",
    "为什么",
    "如何",
    "什么",
    "哪",
    "根据上述",
    "回答以下",
    "问题：",
    "问题:",
    "提问：",
    "提问:",
)

# Entries of ctx_rel below this value are zeroed to keep labels sparse.
_REL_SPARSE_FLOOR = 0.15

_BUILTIN_FALLBACK_PARAGRAPHS = [
    "长江发源于青藏高原的唐古拉山脉，全长约六千三百公里，是中国第一长河。"
    "它自西向东流经青海、西藏、四川、云南、重庆、湖北、湖南、江西、安徽、江苏和上海，"
    "最终在崇明岛附近注入东海。长江流域气候温暖湿润，水量充沛，"
    "孕育了灿烂的巴蜀文化、荆楚文化与吴越文化。三峡工程位于湖北宜昌，"
    "是世界上规模最大的水利枢纽之一，兼具防洪、发电与航运等综合效益。"
    "每年汛期，三峡水库通过调节下泄流量，有效削减了中下游的洪峰。",
    "计算机的发展大致经历了电子管、晶体管、集成电路和超大规模集成电路四个阶段。"
    "1946年诞生的ENIAC使用了一万七千多个电子管，占地约一百七十平方米，"
    "运算速度为每秒五千次加法。晶体管的发明使计算机体积大幅缩小，"
    "可靠性显著提高。随后集成电路把成千上万个元件集成在一块硅片上，"
    "个人计算机由此走进千家万户。如今，基于GPU的并行计算推动了深度学习的突破，"
    "大模型在语言理解、图像生成和代码编写等任务上展现出惊人的能力。",
    "秋天的清晨，山间弥漫着薄薄的雾气，远处的村庄若隐若现。"
    "老人推开木门，沿着青石小路慢慢走向溪边，溪水清澈见底，几尾小鱼在石缝间游动。"
    "他蹲下身子，掬起一捧溪水洗了洗脸，凉意顿时驱散了困倦。"
    "太阳渐渐升起，雾气散去，金色的阳光洒在稻田上，稻穗沉甸甸地低着头。"
    "村里传来鸡鸣狗吠，炊烟袅袅升起，新的一天就这样开始了。",
]


def _split_paths(paths: Union[str, List[str], None]) -> List[str]:
    if not paths:
        return []
    raw: List[str] = []
    items = [paths] if isinstance(paths, str) else list(paths)
    for item in items:
        if isinstance(item, str):
            raw.extend([p.strip() for p in item.split(",") if p.strip()])
    return raw


def _collect_files(paths: Union[str, List[str], None], exts: Tuple[str, ...], verbose: bool) -> List[str]:
    files: List[str] = []
    for path in _split_paths(paths):
        if os.path.isdir(path):
            for ext in exts:
                files.extend(glob.glob(os.path.join(path, f"*{ext}")))
        elif "*" in path or "?" in path:
            files.extend([f for f in glob.glob(path) if f.endswith(exts)])
        elif os.path.isfile(path):
            files.append(path)
        elif verbose:
            print(f"ℹ️ Skipping non-existent optional dataset path: {path}")
    return sorted(set(files))


class ClauseStreamDataset(Dataset):
    """
    Stage-1 single-stream clause dataset mixing QA and long-document continuation (see module docstring).

    __getitem__ returns (T = max_clauses, L = seq_len):
        w_main (T,L) long | main_valid (T,) bool | target_mask (T,) bool | stop_labels (T,) float
        stop_valid (T,) bool | source_id () long [LOGGING ONLY] | ctx_rel (T,T) float | ctx_rel_valid (T,) bool
        sink_label (T,) float | domain str | context_text str | target_text str
    """

    def __init__(
        self,
        qa_paths: Union[str, List[str], None],
        cont_paths: Union[str, List[str], None],
        tokenizer: Optional[CharTokenizer] = None,
        seq_len: int = 32,
        max_clauses: int = 32,
        min_clause_chars: int = 6,
        max_clause_chars: int = 28,
        min_q_clause_chars: int = 4,
        cont_ratio: float = 0.3,
        cont_min_context: int = 2,
        cont_max_context: int = 8,
        max_qa_samples: Optional[int] = None,
        max_cont_docs: Optional[int] = None,
        min_cont_clauses: int = 8,
        rel_idf_sample: int = 20000,
        rel_min_idf_quantile: float = 0.6,
        rel_threshold: float = 0.5,
        seed: int = 42,
        verbose: bool = True,
        epoch_size: Optional[int] = None,
    ):
        super().__init__()
        self.tokenizer = tokenizer if tokenizer is not None else CharTokenizer()
        self.seq_len = int(seq_len)
        self.max_clauses = max(2, int(max_clauses))
        self.min_clause_chars = int(min_clause_chars)
        self.max_clause_chars = int(max_clause_chars)
        self.min_q_clause_chars = int(min_q_clause_chars)
        self.cont_ratio = float(min(max(cont_ratio, 0.0), 1.0))
        self.cont_min_context = max(1, int(cont_min_context))
        self.cont_max_context = max(self.cont_min_context, int(cont_max_context))
        self.max_qa_samples = max_qa_samples
        self.max_cont_docs = max_cont_docs
        self.min_cont_clauses = max(2, int(min_cont_clauses))
        self.rel_idf_sample = int(rel_idf_sample)
        self.rel_min_idf_quantile = float(rel_min_idf_quantile)
        self.rel_threshold = float(rel_threshold)
        self.seed = int(seed)
        self.verbose = verbose
        self.epoch = 0

        self.pad_token_id = getattr(self.tokenizer, "pad_token_id", 0)
        self.eos_token_id = getattr(self.tokenizer, "eos_token_id", None)

        # QA record: {"prefix": [str], "answer": [str], "P": int, "A": int, "truncated": bool, "domain": str}
        self.qa_records: List[Dict[str, Any]] = []
        # Continuation doc: {"clauses": [str], "domain": str}
        self.cont_docs: List[Dict[str, Any]] = []
        self.used_fallback = False
        self.cont_from_qa_fallback = False

        self._load_qa(qa_paths)
        self._load_cont(cont_paths)
        if not self.cont_docs and self.cont_ratio > 0.0 and self.qa_records:
            self._populate_cont_from_qa()
        if not self.qa_records and not self.cont_docs:
            self._load_builtin_fallback(qa_paths, cont_paths)

        # Continuation window slots: doc i owns max(1, n_clauses // T) slots (weighted sampling by length).
        self._cont_slot_cum: List[int] = []
        total = 0
        for d in self.cont_docs:
            total += max(1, len(d["clauses"]) // self.max_clauses)
            self._cont_slot_cum.append(total)
        self.n_cont_windows = total

        # Effective continuation ratio (single-source fallbacks).
        if not self.cont_docs:
            self.eff_cont_ratio = 0.0
        elif not self.qa_records:
            self.eff_cont_ratio = 1.0
        else:
            self.eff_cont_ratio = self.cont_ratio

        if epoch_size is not None and int(epoch_size) > 0:
            self._length = int(epoch_size)
        else:
            self._length = max(1, len(self.qa_records) * (1 if self.eff_cont_ratio < 1.0 else 0)
                               + self.n_cont_windows * (1 if self.eff_cont_ratio > 0.0 else 0))

        # IDF over sampled clauses (document = clause).
        self.idf: Dict[str, float] = {}
        self.idf_default = 1.0
        self.rare_bigram_idf_threshold = 0.0
        self._build_idf()

        if self.verbose:
            self._print_stats()

    # ------------------------------------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------------------------------------
    def _iter_json_objects(self, file_path: str):
        if file_path.endswith(".jsonl"):
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except Exception:
                        continue
        elif file_path.endswith(".json"):
            with open(file_path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except Exception:
                    return
            if isinstance(data, list):
                for obj in data:
                    yield obj
            elif isinstance(data, dict):
                yield data

    def _parse_qa_item(self, item: Any) -> Optional[Tuple[List[str], str, str]]:
        """
        Returns (prefix_turn_texts, answer_text, domain) or None.
        Field detection mirrors `BicameralClauseChainDataset._parse_raw_item` (PROMPT/TARGET candidate keys,
        messages/conversations) with two Stage-1 differences: (a) multi-turn history is preserved as prefix;
        (b) instruction + input are concatenated; multi-line material is NOT split out of the prompt.
        """
        domain = "dialogue"
        turns: Optional[List[Tuple[str, str]]] = None

        if isinstance(item, list):
            # Plain dialogue list ["u1", "a1", "u2", ...]: last utterance = answer, earlier ones = prefix.
            utts = [str(u).strip() for u in item if isinstance(u, str) and str(u).strip()]
            if len(utts) < 2:
                return None
            return utts[:-1], utts[-1], domain

        if not isinstance(item, dict):
            return None
        domain = str(item.get("domain", item.get("category", "general")) or "general").strip()

        for conv_key in ("messages", "conversations"):
            conv = item.get(conv_key)
            if isinstance(conv, list):
                turns = []
                for turn in conv:
                    if not isinstance(turn, dict):
                        continue
                    role = str(turn.get("role", turn.get("from", ""))).lower()
                    content = str(turn.get("content", turn.get("value", "")) or "").strip()
                    if content and role in _USER_ROLES + _ASSISTANT_ROLES:
                        turns.append((role, content))
                break
        if turns:
            last_a = max((i for i, (r, _) in enumerate(turns) if r in _ASSISTANT_ROLES), default=-1)
            if last_a <= 0:
                return None
            prefix = [c for _, c in turns[:last_a]]
            if not any(r in _USER_ROLES for r, _ in turns[:last_a]):
                return None
            return prefix, turns[last_a][1], domain

        prompt = ""
        instr = item.get("instruction")
        if isinstance(instr, str) and instr.strip():
            prompt = instr.strip()
            inp = item.get("input")
            if isinstance(inp, str) and inp.strip() and inp.strip() != prompt:
                prompt = f"{prompt}\n{inp.strip()}"
        if not prompt:
            for k in BicameralClauseChainDataset.PROMPT_CANDIDATE_KEYS + _EXTRA_PROMPT_KEYS:
                val = item.get(k)
                if isinstance(val, str) and val.strip():
                    prompt = val.strip()
                    break
        answer = ""
        for k in BicameralClauseChainDataset.TARGET_CANDIDATE_KEYS + _EXTRA_TARGET_KEYS:
            val = item.get(k)
            if isinstance(val, str) and val.strip():
                answer = val.strip()
                break
        if not prompt or not answer:
            return None
        return [prompt], answer, domain

    def _layout_qa(self, prefix_clauses: List[str], answer_clauses: List[str]) -> Tuple[int, int, bool]:
        """P prefix rows (most recent) + A answer rows (earliest), P >= 1, P + A <= T."""
        T = self.max_clauses
        answer_cap = T - min(len(prefix_clauses), 8)
        A = min(len(answer_clauses), answer_cap)
        P = min(len(prefix_clauses), T - A)
        return P, A, A < len(answer_clauses)

    def _make_qa_record(self, parsed: Tuple[List[str], str, str]) -> Optional[Dict[str, Any]]:
        prefix_texts, answer_text, domain = parsed
        prefix_clauses: List[str] = []
        for t in prefix_texts:
            prefix_clauses.extend(
                _split_into_short_clauses(t, min_chars=self.min_q_clause_chars, max_chars=self.max_clause_chars)
            )
        answer_clauses = _split_into_short_clauses(
            answer_text, min_chars=self.min_clause_chars, max_chars=self.max_clause_chars
        )
        if not prefix_clauses or not answer_clauses:
            return None
        P, A, truncated = self._layout_qa(prefix_clauses, answer_clauses)
        return {
            "prefix": prefix_clauses[-P:],
            "answer": answer_clauses[:A],
            "P": P,
            "A": A,
            "truncated": truncated,
            "n_answer_total": len(answer_clauses),
            "domain": domain,
        }

    def _load_qa(self, qa_paths):
        for fp in _collect_files(qa_paths, (".jsonl", ".json"), self.verbose):
            if self.max_qa_samples and len(self.qa_records) >= self.max_qa_samples:
                break
            if self.verbose:
                print(f"📖 [ClauseStream] reading QA file: {fp}")
            for obj in self._iter_json_objects(fp):
                if self.max_qa_samples and len(self.qa_records) >= self.max_qa_samples:
                    break
                parsed = self._parse_qa_item(obj)
                if parsed is None:
                    continue
                rec = self._make_qa_record(parsed)
                if rec is not None:
                    self.qa_records.append(rec)

    def _add_cont_doc(self, text: str, domain: str = "continuation", min_clauses: Optional[int] = None) -> bool:
        clauses = _extract_clauses_from_text(text, min_chars=self.min_clause_chars, max_chars=self.max_clause_chars)
        if len(clauses) < (self.min_cont_clauses if min_clauses is None else min_clauses):
            return False
        self.cont_docs.append({"clauses": clauses, "domain": domain})
        return True

    def _load_cont(self, cont_paths):
        for fp in _collect_files(cont_paths, (".jsonl", ".json", ".txt"), self.verbose):
            if self.max_cont_docs and len(self.cont_docs) >= self.max_cont_docs:
                break
            if self.verbose:
                print(f"📖 [ClauseStream] reading continuation file: {fp}")
            if fp.endswith(".txt"):
                with open(fp, "r", encoding="utf-8") as f:
                    blocks = re.split(r"\n\s*\n", f.read())
                for b in blocks:
                    if self.max_cont_docs and len(self.cont_docs) >= self.max_cont_docs:
                        break
                    if b.strip():
                        self._add_cont_doc(b.strip())
                continue
            for obj in self._iter_json_objects(fp):
                if self.max_cont_docs and len(self.cont_docs) >= self.max_cont_docs:
                    break
                if isinstance(obj, str):
                    self._add_cont_doc(obj)
                    continue
                if not isinstance(obj, dict):
                    continue
                for k in CONT_TEXT_KEYS:
                    val = obj.get(k)
                    if isinstance(val, str) and val.strip():
                        dom = str(obj.get("domain", "continuation") or "continuation")
                        self._add_cont_doc(val, domain=dom)
                        break

    # =================================================================================================
    # [Feature Addition & Bug Fix - Task #043: Rolling Continuation Stream Fallback from QA Passages]
    # 1. Problem fixed:
    #    When `--cont_paths` is empty (default in `train_phase5_right_brain_stage1.py`) and the dataset
    #    consists solely of QA records (e.g. `omni_knowledge_corpus.jsonl`), `self.cont_docs` was empty
    #    and `self.eff_cont_ratio` collapsed to `0.0` (`samples qa/cont = 320/0`), disabling both:
    #      (a) 30% rolling window continuation training (`source_id == 1`) on Mamba's main stream;
    #      (b) `bench_cont=3` benchmark items and the `📉 σ vs context length` diagnostic curve in eval.
    # 2. Method used:
    #    `_populate_cont_from_qa` derives multi-clause continuation documents from `self.qa_records`
    #    whenever `not self.cont_docs and self.cont_ratio > 0.0 and self.qa_records`:
    #      (a) Extracts contiguous narrative passages from `rec["prefix"]` after stripping leading
    #          short colon-header clauses (e.g. `"叙事片段："`, `"背景资料："`, `"阅读材料："`) and trailing
    #          question/instruction clauses (`"？"`, `"?"`, `"为什么"`, `"请问"`, `"如何"`, etc.).
    #          If the original prefix did not end with a question and `len(rec["answer"]) >= 3`,
    #          appends `rec["answer"]` to form a full continuation sequence.
    #      (b) Otherwise, if `rec["answer"]` itself is a long explanatory passage (`>= self.min_cont_clauses`),
    #          uses `rec["answer"]` as a continuation document.
    # 3. Caveats for subsequent development:
    #    - Only triggered when `cont_paths` yields 0 documents and `cont_ratio > 0.0`.
    #    - Continuation items always keep `stop_valid` all False (window boundaries are not true stops).
    # =================================================================================================
    @staticmethod
    def _is_trailing_question_or_prompt(clause: str) -> bool:
        s = clause.strip()
        if not s:
            return True
        if s.endswith(("？", "?")):
            return True
        if len(s) <= 8 and s.endswith(("：", ":")):
            return True
        return any(m in s for m in _QA_TAIL_MARKERS)

    def _populate_cont_from_qa(self) -> int:
        min_ctx_body = max(6, self.min_cont_clauses - 2)
        for rec in self.qa_records:
            if self.max_cont_docs and len(self.cont_docs) >= self.max_cont_docs:
                break
            prefix = list(rec.get("prefix") or [])
            answer = list(rec.get("answer") or [])
            domain = str(rec.get("domain", "continuation") or "continuation")

            orig_ended_with_question = bool(prefix) and self._is_trailing_question_or_prompt(prefix[-1])
            ctx_body = list(prefix)

            # Strip leading short colon-header clauses (e.g. "叙事片段：", "背景资料：", "阅读材料：")
            while ctx_body and len(ctx_body[0].strip()) <= 8 and ctx_body[0].strip().endswith(("：", ":")):
                ctx_body.pop(0)
            if ctx_body:
                m_hdr = re.match(r"^([^，,。！？；：:\s]{1,7}[：:])\s*(.+)$", ctx_body[0].strip())
                if m_hdr and len(m_hdr.group(1)) <= 8 and len(m_hdr.group(2).strip()) >= self.min_q_clause_chars:
                    ctx_body[0] = m_hdr.group(2).strip()

            # Strip trailing question/instruction clauses at the end of prefix
            while ctx_body and self._is_trailing_question_or_prompt(ctx_body[-1]):
                ctx_body.pop()
                orig_ended_with_question = True

            if len(ctx_body) >= min_ctx_body:
                if len(answer) >= 3 and not orig_ended_with_question:
                    ctx_body.extend(answer)
                self.cont_docs.append({"clauses": list(ctx_body), "domain": domain})
            elif len(answer) >= self.min_cont_clauses:
                self.cont_docs.append({"clauses": list(answer), "domain": domain})

        if self.cont_docs:
            self.cont_from_qa_fallback = True
            if self.verbose:
                print(
                    f"📖 [ClauseStream] derived {len(self.cont_docs)} continuation documents "
                    f"from multi-clause QA passages (cont_ratio={self.cont_ratio:.2f})."
                )
        return len(self.cont_docs)

    def _load_builtin_fallback(self, qa_paths, cont_paths):
        print(
            f"⚠️ Warning: [ClauseStream] no QA ('{qa_paths}') or continuation ('{cont_paths}') data found; "
            f"populating built-in tiny fallback records for verification."
        )
        self.used_fallback = True
        for raw in BicameralClauseChainDataset._BUILTIN_FALLBACK_RECORDS:
            parsed = self._parse_qa_item({"domain": raw["domain"], "question": raw["question"],
                                          "answer": raw["answer"]})
            rec = self._make_qa_record(parsed) if parsed else None
            if rec is not None:
                self.qa_records.append(rec)
        for p in _BUILTIN_FALLBACK_PARAGRAPHS:
            self._add_cont_doc(p, min_clauses=min(self.min_cont_clauses, 4))

    # ------------------------------------------------------------------------------------------------
    # IDF / rare spans
    # ------------------------------------------------------------------------------------------------
    @staticmethod
    def _raw_spans(text: str) -> Tuple[set, set]:
        """(char bigrams excluding pure-ASCII-alnum pairs, atomic digit-run/ASCII-word spans with len >= 2)."""
        norm = _normalize_for_bigram_overlap(text or "")
        bigrams = set()
        for i in range(len(norm) - 1):
            a, b = norm[i], norm[i + 1]
            if (a.isascii() and a.isalnum()) or (b.isascii() and b.isalnum()):
                continue  # ASCII alnum is covered by atomic digit / ASCII-word spans (avoids "斩1"-type noise)
            bigrams.add(norm[i:i + 2])
        atomic = set()
        for m in _DIGIT_RUN_RE.findall(text or ""):
            if len(m) >= 2:
                atomic.add(m)
        for m in _ASCII_WORD_RE.findall(text or ""):
            if len(m) >= 2:
                atomic.add(m.lower())
        return bigrams, atomic

    def _build_idf(self):
        rng = random.Random(self.seed + 7919)
        # Reservoir sample of clauses from both sources (document = clause).
        sample: List[str] = []
        seen = 0

        def _offer(c: str):
            nonlocal seen
            seen += 1
            if len(sample) < self.rel_idf_sample:
                sample.append(c)
            else:
                j = rng.randrange(seen)
                if j < self.rel_idf_sample:
                    sample[j] = c

        for rec in self.qa_records:
            for c in rec["prefix"] + rec["answer"]:
                _offer(c)
        for d in self.cont_docs:
            for c in d["clauses"]:
                _offer(c)

        df_bi: Dict[str, int] = {}
        df_at: Dict[str, int] = {}
        for c in sample:
            bi, at = self._raw_spans(c)
            for s in bi:
                df_bi[s] = df_bi.get(s, 0) + 1
            for s in at:
                df_at[s] = df_at.get(s, 0) + 1
        N = max(1, len(sample))
        self.idf_n_docs = N
        self.idf_default = math.log((N + 1) / 1.0) + 1.0  # unseen span = maximally rare
        idf_bi = {s: math.log((N + 1) / (df + 1)) + 1.0 for s, df in df_bi.items()}
        idf_at = {s: math.log((N + 1) / (df + 1)) + 1.0 for s, df in df_at.items()}
        self.idf = dict(idf_bi)
        self.idf.update(idf_at)
        self._atomic_keys = set(idf_at.keys())

        # Occurrence (df)-weighted quantile over bigram IDF values.
        if idf_bi:
            pairs = sorted(((idf_bi[s], df_bi[s]) for s in idf_bi), key=lambda x: x[0])
            total = sum(w for _, w in pairs)
            target = self.rel_min_idf_quantile * total
            acc = 0
            thr = pairs[-1][0]
            for v, w in pairs:
                acc += w
                if acc >= target:
                    thr = v
                    break
            self.rare_bigram_idf_threshold = thr
        else:
            self.rare_bigram_idf_threshold = 0.0

    def rare_spans(self, text: str) -> Dict[str, float]:
        """R(text): rare char bigrams (IDF >= df-weighted quantile threshold) + digit runs / ASCII words (len>=2)."""
        bi, at = self._raw_spans(text)
        out: Dict[str, float] = {}
        thr = self.rare_bigram_idf_threshold
        for s in bi:
            v = self.idf.get(s, self.idf_default)
            if v >= thr:
                out[s] = v
        for s in at:
            out[s] = self.idf.get(s, self.idf_default)
        return out

    def _relevance(self, r_t: Dict[str, float], r_j: Dict[str, float]) -> float:
        denom = sum(r_t.values())
        if denom <= 1e-8:
            return 0.0
        num = sum(v for s, v in r_t.items() if s in r_j)
        return float(num / max(1e-8, denom))

    # ------------------------------------------------------------------------------------------------
    # Index mapping
    # ------------------------------------------------------------------------------------------------
    def set_epoch(self, epoch: int):
        """Rotates index → record/slot mapping across epochs (default 0 keeps outputs fixed per index)."""
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self._length

    def _route(self, idx: int) -> Tuple[int, int]:
        """Deterministic Bresenham interleave: returns (source_id, per-source slot)."""
        r = self.eff_cont_ratio
        if r <= 0.0:
            return 0, idx
        if r >= 1.0:
            return 1, idx
        n_cont_before = math.floor(idx * r)
        is_cont = math.floor((idx + 1) * r) > n_cont_before
        if is_cont:
            return 1, n_cont_before
        return 0, idx - n_cont_before

    def is_continuation_index(self, idx: int) -> bool:
        return self._route(idx)[0] == 1

    def _encode_clause_tokens(self, clause_str: str) -> List[int]:
        # Identical to BicameralClauseChainDataset._encode_clause_tokens ([BOS] chars [EOS], EOS at L-1 if cut).
        ids = list(self.tokenizer.encode(clause_str, add_special_tokens=True))
        if len(ids) > self.seq_len:
            ids = ids[: self.seq_len]
            if self.eos_token_id is not None:
                ids[-1] = self.eos_token_id
        elif len(ids) < self.seq_len:
            ids = ids + [self.pad_token_id] * (self.seq_len - len(ids))
        return ids

    def _qa_layout_for(self, slot: int):
        rec = self.qa_records[(slot + self.epoch * len(self.qa_records)) % len(self.qa_records)]
        clauses = rec["prefix"] + rec["answer"]
        P, A = rec["P"], rec["A"]
        return clauses, P, A, rec["truncated"], rec["domain"]

    def _cont_window_for(self, slot: int):
        g = (slot + self.epoch * self.n_cont_windows) % self.n_cont_windows
        doc_i = bisect.bisect_right(self._cont_slot_cum, g)
        doc = self.cont_docs[doc_i]
        rng = random.Random((self.seed * 1_000_003 + 17) ^ (g * 2_654_435_761 + 0x9E3779B9))
        n = len(doc["clauses"])
        W = min(self.max_clauses, n)
        start = rng.randint(0, n - W)
        max_k = max(self.cont_min_context, min(self.cont_max_context, W - 4))
        k = rng.randint(self.cont_min_context, max_k)
        k = max(1, min(k, W - 1))
        return doc["clauses"][start:start + W], k, doc["domain"]

    # ------------------------------------------------------------------------------------------------
    # Item
    # ------------------------------------------------------------------------------------------------
    def _build_item(self, idx: int, with_debug: bool = False) -> Dict[str, Any]:
        T, L = self.max_clauses, self.seq_len
        src, slot = self._route(idx)
        if src == 0:
            clauses, n_ctx, A, truncated, domain = self._qa_layout_for(slot)
            n_rows = n_ctx + A
        else:
            clauses, n_ctx, domain = self._cont_window_for(slot)
            n_rows = len(clauses)
            truncated = False

        w_main = torch.full((T, L), self.pad_token_id, dtype=torch.long)
        main_valid = torch.zeros(T, dtype=torch.bool)
        target_mask = torch.zeros(T, dtype=torch.bool)
        stop_labels = torch.zeros(T, dtype=torch.float32)
        stop_valid = torch.zeros(T, dtype=torch.bool)
        ctx_rel = torch.zeros(T, T, dtype=torch.float32)
        ctx_rel_valid = torch.zeros(T, dtype=torch.bool)
        sink_label = torch.zeros(T, dtype=torch.float32)

        for t in range(n_rows):
            w_main[t] = torch.tensor(self._encode_clause_tokens(clauses[t]), dtype=torch.long)
        main_valid[:n_rows] = True
        target_mask[n_ctx:n_rows] = True
        target_mask[0] = False

        if src == 0:
            stop_valid[n_ctx:n_rows] = True
            if not truncated:
                stop_labels[n_rows - 1] = 1.0

        spans = [self.rare_spans(c) for c in clauses[:n_rows]]
        for t in range(max(1, n_ctx), n_rows):
            ctx_rel_valid[t] = True
            best = 0.0
            for j in range(t):
                rel = self._relevance(spans[t], spans[j])
                if rel >= _REL_SPARSE_FLOOR:
                    ctx_rel[t, j] = rel
                    best = max(best, rel)
            sink_label[t] = 1.0 if best < self.rel_threshold else 0.0

        item = {
            "w_main": w_main,
            "main_valid": main_valid,
            "target_mask": target_mask,
            "stop_labels": stop_labels,
            "stop_valid": stop_valid,
            "source_id": torch.tensor(src, dtype=torch.long),
            "ctx_rel": ctx_rel,
            "ctx_rel_valid": ctx_rel_valid,
            "sink_label": sink_label,
            "domain": domain,
            "context_text": "".join(clauses[:n_ctx]),
            "target_text": "".join(clauses[n_ctx:n_rows]),
        }
        if with_debug:
            item["_clauses"] = list(clauses[:n_rows])
            item["_spans"] = spans
            item["_truncated"] = truncated
        return item

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx < 0:
            idx += len(self)
        return self._build_item(int(idx))

    def debug_item(self, idx: int) -> Dict[str, Any]:
        """Same as __getitem__ plus `_clauses`, `_spans` (rare spans per row), `_truncated` for inspection."""
        return self._build_item(int(idx), with_debug=True)

    # ------------------------------------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------------------------------------
    def compute_stats(self, max_items: int = 2000) -> Dict[str, Any]:
        n = len(self)
        m = min(n, max_items)
        idxs = sorted(set(int(i * n / m) for i in range(m))) if m > 0 else []
        main_len, tgt_rows, sink_pos, sink_tot, rel_cnt, rel_cnt_thr = [], [], 0, 0, [], []
        n_cont = 0
        for i in idxs:
            it = self._build_item(i)
            n_cont += int(it["source_id"].item())
            main_len.append(int(it["main_valid"].sum()))
            tgt_rows.append(int(it["target_mask"].sum()))
            v = it["ctx_rel_valid"]
            sink_tot += int(v.sum())
            sink_pos += int(it["sink_label"][v].sum().item())
            if v.any():
                rel_cnt.extend((it["ctx_rel"][v] > 0).sum(-1).tolist())
                rel_cnt_thr.extend((it["ctx_rel"][v] >= self.rel_threshold).sum(-1).tolist())
        n_trunc = sum(1 for r in self.qa_records if r["truncated"])
        mean = lambda xs: (sum(xs) / len(xs)) if xs else 0.0
        return {
            "n_qa": len(self.qa_records),
            "n_cont_docs": len(self.cont_docs),
            "est_cont_windows": self.n_cont_windows,
            "len": n,
            "sampled_items": len(idxs),
            "sampled_cont_frac": n_cont / max(1, len(idxs)),
            "mean_main_len": mean(main_len),
            "mean_target_rows": mean(tgt_rows),
            "qa_truncated_pct": 100.0 * n_trunc / max(1, len(self.qa_records)),
            "sink_pos_pct": 100.0 * sink_pos / max(1, sink_tot),
            "mean_rel_ctx_rows": mean(rel_cnt),
            "mean_rel_ctx_rows_ge_thr": mean(rel_cnt_thr),
            "rare_bigram_idf_threshold": self.rare_bigram_idf_threshold,
            "idf_docs": self.idf_n_docs,
        }

    def _print_stats(self):
        s = self.compute_stats()
        print(
            f"✅ ClauseStreamDataset: n_qa={s['n_qa']} n_cont_docs={s['n_cont_docs']} "
            f"est_windows={s['est_cont_windows']} len={s['len']} (cont_ratio eff={self.eff_cont_ratio:.2f}, "
            f"sampled cont frac={s['sampled_cont_frac']:.3f}) T={self.max_clauses} L={self.seq_len} | "
            f"mean main len={s['mean_main_len']:.2f} mean target rows={s['mean_target_rows']:.2f} "
            f"QA truncated={s['qa_truncated_pct']:.1f}% | sink=1 on {s['sink_pos_pct']:.1f}% of target rows | "
            f"mean relevant ctx rows/target row: >0.15={s['mean_rel_ctx_rows']:.2f}, "
            f">={self.rel_threshold}={s['mean_rel_ctx_rows_ge_thr']:.2f} | "
            f"IDF docs={s['idf_docs']} rare-bigram thr={s['rare_bigram_idf_threshold']:.3f} "
            f"(q={self.rel_min_idf_quantile}, unseen idf={self.idf_default:.3f})"
            + (" [QA->CONT FALLBACK]" if self.cont_from_qa_fallback else "")
            + (" [BUILTIN FALLBACK]" if self.used_fallback else "")
        )
        bigrams = [(k, v) for k, v in self.idf.items() if k not in self._atomic_keys]
        if bigrams:
            bigrams.sort(key=lambda kv: kv[1])
            low = ", ".join(f"{k}:{v:.2f}" for k, v in bigrams[:20])
            high = ", ".join(f"{k}:{v:.2f}" for k, v in bigrams[-20:])
            print(f"   ↓ lowest-IDF bigrams (common, NOT rare): {low}")
            print(f"   ↑ highest-IDF bigrams (rare): {high}")
            probes = [b for b in ("的是", "我们", "一个", "可以") if b in self.idf]
            if probes:
                print("   probe: " + ", ".join(
                    f"{b}={self.idf[b]:.2f}({'rare' if self.idf[b] >= self.rare_bigram_idf_threshold else 'common'})"
                    for b in probes))


def create_clause_stream_dataloader(
    qa_paths: Union[str, List[str], None],
    cont_paths: Union[str, List[str], None],
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = False,
    pin_memory: bool = False,
    **dataset_kwargs,
) -> Tuple[ClauseStreamDataset, DataLoader]:
    """Builds (dataset, dataloader); extra kwargs go to `ClauseStreamDataset`. Default collate is sufficient."""
    dataset = ClauseStreamDataset(qa_paths, cont_paths, **dataset_kwargs)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory,
    )
    return dataset, loader
