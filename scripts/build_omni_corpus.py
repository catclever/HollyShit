#!/usr/bin/env python3
"""
scripts/build_omni_corpus.py
============================
Multi-Domain Omni-Corpus Builder for Phase 4 Continuous Flow Matching & 1D Translator.

Downloads/streams high-quality data across 7 major domains (2-3 sources per domain),
extracting up to ~100k samples per source to build an omni-domain training corpus (~1.5M - 2M pairs).

Domains:
  1. Chat & Instruct (对话与指令)
  2. Encyclopedia & Factual (百科与严肃说明文)
  3. Literature & Fiction (严肃文学与小说叙事)
  4. Code & Structured Data (代码工程与结构化数据)
  5. Math & Step-by-Step Logic (数学推导与严谨逻辑)
  6. Classical Chinese & Poetry (文言文与古典诗词)
  7. Bilingual & Translation (中英双语平行语料)

Usage:
  python scripts/build_omni_corpus.py --samples_per_source 100000 --output /content/omni_corpus.jsonl
"""

import os
import sys
import json
import random
import argparse
import re
from typing import Iterator, Tuple, Dict, Any, Optional

def clean_text(text: Optional[str]) -> str:
    if not text or not isinstance(text, str):
        return ""
    text = text.strip()
    # Normalize multiple consecutive spaces/newlines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text

def chunk_long_text(text: str, max_target_len: int = 250) -> Optional[Tuple[str, str]]:
    """
    For non-dialogue continuous texts (like Wikipedia, articles, literature),
    splits text into a natural (Prompt/Prefix -> Target/Continuation) pair.
    """
    text = clean_text(text)
    if len(text) < 40:
        return None
    
    # Try finding sentence delimiter around first 30-80 chars
    delimiters = ["\n\n", "\n", "。", "！", "？", "；", "...", "…"]
    split_idx = -1
    for d in delimiters:
        pos = text.find(d, 20)
        if 20 <= pos <= 100:
            split_idx = pos + len(d)
            break
            
    if split_idx == -1:
        split_idx = min(60, len(text) // 3)
        
    prompt = text[:split_idx].strip()
    target = text[split_idx:split_idx + max_target_len].strip()
    
    if len(prompt) < 5 or len(target) < 15:
        return None
    return prompt, target


# =========================================================================
# Domain Extractors (Streaming from HuggingFace Datasets)
# =========================================================================

def extract_belle_instruct(max_samples: int) -> Iterator[Dict[str, str]]:
    """Domain: Chat & Instruct | Source: BelleGroup/train_1M_CN"""
    from datasets import load_dataset
    print(">>> [Chat] Streaming BelleGroup/train_1M_CN...")
    count = 0
    try:
        ds = load_dataset("BelleGroup/train_1M_CN", split="train", streaming=True)
        for item in ds:
            instr = clean_text(item.get("instruction", ""))
            inp = clean_text(item.get("input", ""))
            out = clean_text(item.get("output", ""))
            
            prompt = f"{instr}\n{inp}".strip() if inp else instr
            if len(prompt) >= 5 and len(out) >= 10:
                yield {"prompt": prompt, "target": out, "domain": "chat_instruct", "source": "belle_1m"}
                count += 1
                if count >= max_samples:
                    break
    except Exception as e:
        print(f"⚠️ Warning during BelleGroup/train_1M_CN stream: {e}")
    print(f"  -> Extracted {count} samples from BelleGroup/train_1M_CN")


def extract_gpt4_alpaca_zh(max_samples: int) -> Iterator[Dict[str, str]]:
    """Domain: Chat & Instruct | Source: silk-road/alpaca-data-gpt4-chinese"""
    from datasets import load_dataset
    print(">>> [Chat] Streaming silk-road/alpaca-data-gpt4-chinese...")
    count = 0
    try:
        ds = load_dataset("silk-road/alpaca-data-gpt4-chinese", split="train", streaming=True)
        for item in ds:
            instr = clean_text(item.get("instruction", ""))
            inp = clean_text(item.get("input", ""))
            out = clean_text(item.get("output", ""))
            prompt = f"{instr}\n{inp}".strip() if inp else instr
            if len(prompt) >= 5 and len(out) >= 10:
                yield {"prompt": prompt, "target": out, "domain": "chat_instruct", "source": "alpaca_gpt4_zh"}
                count += 1
                if count >= max_samples:
                    break
    except Exception as e:
        print(f"⚠️ Warning during alpaca-data-gpt4-chinese stream: {e}")
    print(f"  -> Extracted {count} samples from alpaca-data-gpt4-chinese")


def extract_local_moss(local_path: str, max_samples: int) -> Iterator[Dict[str, str]]:
    """Domain: Chat & Instruct | Source: Existing MOSS dialogues"""
    print(f">>> [Chat] Reading local MOSS dialogues from {local_path}...")
    if not os.path.isfile(local_path):
        print(f"  -> {local_path} not found, skipping local MOSS.")
        return
    count = 0
    with open(local_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                prompt = clean_text(item.get("prompt", ""))
                target = clean_text(item.get("target", ""))
                if len(prompt) >= 5 and len(target) >= 10:
                    yield {"prompt": prompt, "target": target, "domain": "chat_instruct", "source": "local_moss"}
                    count += 1
                    if count >= max_samples:
                        break
            except Exception:
                continue
    print(f"  -> Extracted {count} samples from local MOSS")


def extract_wiki_chinese(max_samples: int) -> Iterator[Dict[str, str]]:
    """Domain: Encyclopedia | Source: pleisto/wikipedia-cn-20230720-filtered"""
    from datasets import load_dataset
    print(">>> [Encyclopedia] Streaming pleisto/wikipedia-cn-20230720-filtered...")
    count = 0
    try:
        ds = load_dataset("pleisto/wikipedia-cn-20230720-filtered", split="train", streaming=True)
        for item in ds:
            title = clean_text(item.get("title", ""))
            completion = clean_text(item.get("completion", ""))
            if not completion and "text" in item:
                completion = clean_text(item["text"])
                
            if title and completion:
                prompt = f"关于【{title}】的百科词条："
                target = completion[:300].strip()
                yield {"prompt": prompt, "target": target, "domain": "encyclopedia", "source": "wiki_cn"}
                count += 1
                if count >= max_samples:
                    break
            else:
                pair = chunk_long_text(completion)
                if pair:
                    yield {"prompt": pair[0], "target": pair[1], "domain": "encyclopedia", "source": "wiki_cn"}
                    count += 1
                    if count >= max_samples:
                        break
    except Exception as e:
        print(f"⚠️ Warning during wikipedia-cn stream: {e}")
    print(f"  -> Extracted {count} samples from wikipedia-cn")


def extract_zhihu_qa(max_samples: int) -> Iterator[Dict[str, str]]:
    """Domain: Encyclopedia / Q&A | Source: wangrui6/Zhihu-KOL or zirui3/zhihu_qa"""
    from datasets import load_dataset
    print(">>> [Encyclopedia/Knowledge] Streaming Zhihu Q&A...")
    count = 0
    dataset_candidates = ["wangrui6/Zhihu-KOL", "zirui3/zhihu_qa"]
    for dname in dataset_candidates:
        if count >= max_samples:
            break
        try:
            ds = load_dataset(dname, split="train", streaming=True)
            for item in ds:
                q = clean_text(item.get("INSTRUCTION", item.get("instruction", item.get("question", item.get("title", "")))))
                a = clean_text(item.get("RESPONSE", item.get("output", item.get("answer", item.get("content", "")))))
                if len(q) >= 5 and len(a) >= 15:
                    yield {"prompt": q, "target": a[:300], "domain": "encyclopedia", "source": dname}
                    count += 1
                    if count >= max_samples:
                        break
        except Exception as e:
            print(f"⚠️ Warning during {dname} stream: {e}")
    print(f"  -> Extracted {count} samples from Zhihu Q&A")


def extract_code_alpaca(max_samples: int) -> Iterator[Dict[str, str]]:
    """Domain: Code & Structured | Source: sahil2801/CodeAlpaca-20k or theblackcat102/evol-codealpaca-v1"""
    from datasets import load_dataset
    print(">>> [Code] Streaming CodeAlpaca...")
    count = 0
    sources = ["sahil2801/CodeAlpaca-20k", "theblackcat102/evol-codealpaca-v1"]
    for sname in sources:
        if count >= max_samples:
            break
        try:
            ds = load_dataset(sname, split="train", streaming=True)
            for item in ds:
                instr = clean_text(item.get("instruction", ""))
                inp = clean_text(item.get("input", ""))
                out = clean_text(item.get("output", ""))
                prompt = f"{instr}\n{inp}".strip() if inp else instr
                if len(prompt) >= 5 and len(out) >= 10:
                    yield {"prompt": prompt, "target": out[:350], "domain": "code", "source": sname}
                    count += 1
                    if count >= max_samples:
                        break
        except Exception as e:
            print(f"⚠️ Warning during {sname} stream: {e}")
    print(f"  -> Extracted {count} samples from CodeAlpaca")


def extract_sql_structured(max_samples: int) -> Iterator[Dict[str, str]]:
    """Domain: Code & Structured | Source: b-mc2/sql-create-context"""
    from datasets import load_dataset
    print(">>> [Code/SQL] Streaming b-mc2/sql-create-context...")
    count = 0
    try:
        ds = load_dataset("b-mc2/sql-create-context", split="train", streaming=True)
        for item in ds:
            prompt_q = clean_text(item.get("question", ""))
            ctx = clean_text(item.get("context", ""))
            answer = clean_text(item.get("answer", ""))
            prompt = f"Schema: {ctx}\nQuery: {prompt_q}"
            if prompt_q and answer:
                yield {"prompt": prompt, "target": answer, "domain": "code", "source": "sql_create_context"}
                count += 1
                if count >= max_samples:
                    break
    except Exception as e:
        print(f"⚠️ Warning during sql-create-context stream: {e}")
    print(f"  -> Extracted {count} samples from SQL Context")


def extract_math_gsm8k_zh(max_samples: int) -> Iterator[Dict[str, str]]:
    """Domain: Math & Logic | Source: meta-math/GSM8K_zh"""
    from datasets import load_dataset
    print(">>> [Math] Streaming meta-math/GSM8K_zh...")
    count = 0
    try:
        ds = load_dataset("meta-math/GSM8K_zh", split="train", streaming=True)
        for item in ds:
            q = clean_text(item.get("question_zh", item.get("question", item.get("instruction", ""))))
            ans = clean_text(item.get("answer_zh", item.get("answer", item.get("output", ""))))
            if len(q) >= 5 and len(ans) >= 10:
                yield {"prompt": q, "target": ans[:350], "domain": "math_logic", "source": "gsm8k_zh"}
                count += 1
                if count >= max_samples:
                    break
    except Exception as e:
        print(f"⚠️ Warning during GSM8K_zh stream: {e}")
    print(f"  -> Extracted {count} samples from GSM8K_zh")


def extract_math_ape210k(max_samples: int) -> Iterator[Dict[str, str]]:
    """Domain: Math & Logic | Source: MU-NLPC/calc-ape210k"""
    from datasets import load_dataset
    print(">>> [Math] Streaming MU-NLPC/calc-ape210k...")
    count = 0
    try:
        ds = load_dataset("MU-NLPC/calc-ape210k", split="train", streaming=True)
        for item in ds:
            q = clean_text(item.get("question_chinese", item.get("question", "")))
            ans = clean_text(item.get("equation", item.get("chain_of_thought", item.get("answer", ""))))
            if len(q) >= 5 and len(ans) >= 5:
                yield {"prompt": q, "target": ans, "domain": "math_logic", "source": "ape210k"}
                count += 1
                if count >= max_samples:
                    break
    except Exception as e:
        print(f"⚠️ Warning during calc-ape210k stream: {e}")
    print(f"  -> Extracted {count} samples from Ape210K")


def extract_classical_chinese(max_samples: int) -> Iterator[Dict[str, str]]:
    """Domain: Classical Chinese | Source: gujilab/chinese-classical-corpus (translate)"""
    from datasets import load_dataset
    print(">>> [Classical Chinese] Streaming gujilab/chinese-classical-corpus (translate)...")
    count = 0
    try:
        ds = load_dataset("gujilab/chinese-classical-corpus", "translate", split="train", streaming=True)
        for item in ds:
            instr = clean_text(item.get("instruction", "文白互译："))
            src = clean_text(item.get("source", ""))
            tgt = clean_text(item.get("target", ""))
            prompt = f"{instr}\n{src}".strip()
            if len(src) >= 4 and len(tgt) >= 4:
                yield {"prompt": prompt, "target": tgt[:300], "domain": "classical_chinese", "source": "classical_corpus"}
                count += 1
                if count >= max_samples:
                    break
    except Exception as e:
        print(f"⚠️ Warning during classical-corpus stream: {e}")
    print(f"  -> Extracted {count} samples from gujilab/chinese-classical-corpus")


def extract_classical_poetry(max_samples: int) -> Iterator[Dict[str, str]]:
    """Domain: Classical Chinese & Poetry | Source: erhwenkuo/poetry-chinese-zhtw or Lifan-Z/Chinese-poetries-txt"""
    from datasets import load_dataset
    print(">>> [Poetry] Streaming classical poetry...")
    count = 0
    candidates = ["Lifan-Z/Chinese-poetries-txt", "erhwenkuo/poetry-chinese-zhtw"]
    for cname in candidates:
        if count >= max_samples:
            break
        try:
            ds = load_dataset(cname, split="train", streaming=True)
            for item in ds:
                title = clean_text(item.get("title", ""))
                author = clean_text(item.get("author", ""))
                content = clean_text(item.get("content", item.get("text", item.get("paragraphs", ""))))
                if isinstance(content, list):
                    content = " ".join(content)
                if content and len(content) >= 10:
                    header = f"古诗词【{title}】（{author}）：" if (title or author) else "诗词鉴赏："
                    pair = chunk_long_text(f"{header}{content}")
                    if pair:
                        yield {"prompt": pair[0], "target": pair[1], "domain": "classical_chinese", "source": cname}
                        count += 1
                        if count >= max_samples:
                            break
        except Exception as e:
            print(f"⚠️ Warning during poetry stream {cname}: {e}")
    print(f"  -> Extracted {count} samples from Classical Poetry")


def extract_opus_bilingual(max_samples: int) -> Iterator[Dict[str, str]]:
    """Domain: Bilingual & Translation | Source: Helsinki-NLP/opus-100 (en-zh)"""
    from datasets import load_dataset
    print(">>> [Bilingual/Translation] Streaming Helsinki-NLP/opus-100 (en-zh)...")
    count = 0
    try:
        ds = load_dataset("Helsinki-NLP/opus-100", "en-zh", split="train", streaming=True)
        for item in ds:
            trans = item.get("translation", {})
            en = clean_text(trans.get("en", ""))
            zh = clean_text(trans.get("zh", ""))
            if len(en) >= 8 and len(zh) >= 4:
                # Randomly alternate en->zh and zh->en to teach bidirectional alignment
                if random.random() < 0.5:
                    prompt = f"Translate English to Chinese: {en}"
                    target = zh
                else:
                    prompt = f"翻译为英文：{zh}"
                    target = en
                yield {"prompt": prompt, "target": target[:300], "domain": "bilingual", "source": "opus_100_en_zh"}
                count += 1
                if count >= max_samples:
                    break
    except Exception as e:
        print(f"⚠️ Warning during opus-100 stream: {e}")
    print(f"  -> Extracted {count} samples from Helsinki-NLP/opus-100")


# =========================================================================
# Main Assembler
# =========================================================================

def build_omni_corpus(
    output_path: str,
    samples_per_source: int = 100000,
    local_moss_path: str = "data/Basic_ZH/moss_dialogues.jsonl",
    shuffle_buffer_size: int = 50000
):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # List of generator functions
    extractors = [
        # Domain 1: Chat & Instruct
        lambda: extract_belle_instruct(samples_per_source),
        lambda: extract_gpt4_alpaca_zh(samples_per_source),
        lambda: extract_local_moss(local_moss_path, samples_per_source),
        
        # Domain 2: Encyclopedia & Factual
        lambda: extract_wiki_chinese(samples_per_source),
        lambda: extract_zhihu_qa(samples_per_source),
        
        # Domain 3: Code & Structured
        lambda: extract_code_alpaca(samples_per_source),
        lambda: extract_sql_structured(samples_per_source),
        
        # Domain 4: Math & Logic
        lambda: extract_math_gsm8k_zh(samples_per_source),
        lambda: extract_math_ape210k(samples_per_source),
        
        # Domain 5: Classical Chinese & Poetry
        lambda: extract_classical_chinese(samples_per_source),
        lambda: extract_classical_poetry(samples_per_source),
        
        # Domain 6: Bilingual & Translation
        lambda: extract_opus_bilingual(samples_per_source),
    ]

    print("==================================================================")
    print(f"🚀 Starting Omni-Corpus Assembly -> {output_path}")
    print(f"🎯 Target Samples per Source: {samples_per_source}")
    print(f"📚 Total Candidate Extractors: {len(extractors)}")
    print("==================================================================")

    total_written = 0
    buffer = []
    
    with open(output_path, "w", encoding="utf-8") as out_f:
        for idx, ext_fn in enumerate(extractors, 1):
            print(f"\n[{idx}/{len(extractors)}] Executing source extractor...")
            source_count = 0
            for item in ext_fn():
                buffer.append(item)
                source_count += 1
                
                # When buffer exceeds threshold, shuffle and write out
                if len(buffer) >= shuffle_buffer_size:
                    random.shuffle(buffer)
                    for b_item in buffer:
                        out_f.write(json.dumps(b_item, ensure_ascii=False) + "\n")
                    total_written += len(buffer)
                    print(f"  💾 Flushed {len(buffer)} samples (Total written: {total_written})...")
                    buffer.clear()
                    out_f.flush()
            print(f"  -> Extractor [{idx}] collected {source_count} samples.")

        # Flush remaining buffer
        if buffer:
            random.shuffle(buffer)
            for b_item in buffer:
                out_f.write(json.dumps(b_item, ensure_ascii=False) + "\n")
            total_written += len(buffer)
            buffer.clear()
            out_f.flush()

    print("\n==================================================================")
    print(f"🎉 Omni-Corpus Assembly Complete!")
    print(f"📦 Total Samples Saved: {total_written}")
    print(f"📁 File: {output_path}")
    print(f"📊 Size: {os.path.getsize(output_path) / (1024 * 1024):.2f} MB")
    print("==================================================================")


def main():
    parser = argparse.ArgumentParser(description="Build Multi-Domain Omni Corpus for Continuous Flow Matching")
    parser.add_argument("--samples_per_source", type=int, default=100000, help="Max samples extracted per data source")
    parser.add_argument("--output", type=str, default="/content/omni_corpus.jsonl", help="Output JSONL file path")
    parser.add_argument("--local_moss", type=str, default="data/Basic_ZH/moss_dialogues.jsonl", help="Local MOSS jsonl path")
    parser.add_argument("--shuffle_buffer", type=int, default=50000, help="Buffer size for randomized writing")
    args = parser.parse_args()

    build_omni_corpus(
        output_path=args.output,
        samples_per_source=args.samples_per_source,
        local_moss_path=args.local_moss,
        shuffle_buffer_size=args.shuffle_buffer
    )

if __name__ == "__main__":
    main()
