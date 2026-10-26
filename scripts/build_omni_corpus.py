#!/usr/bin/env python3
"""
scripts/build_omni_corpus.py
============================
Backward-Compatibility Shim for `scripts/build_omni_knowledge_corpus.py`.

Notice:
  This script has been split into two dedicated tools:
    1. `scripts/build_omni_knowledge_corpus.py` -> produces `omni_knowledge_corpus.jsonl`
       (8 grounded domains: Chat, Encyclopedia, Literature, Code, Math, Classical, News, Bilingual)
    2. `scripts/build_moss_dialogues.py` -> produces `moss_dialogues.jsonl`
       (Dedicated MOSS dialogue stream/local-raw processing)

Executing this script automatically delegates to `build_omni_knowledge_corpus.py`.
Any legacy `--local_moss` argument is gracefully ignored.
"""

import os
import sys
import argparse
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Re-export key utilities for backward compatibility
from scripts.build_omni_knowledge_corpus import (
    build_omni_knowledge_corpus,
    clean_text,
    is_non_cheating_grounded_pair,
    has_substantive_overlap,
    extract_factual_clauses,
    try_split_structured_prompt,
    try_split_classical_text,
    build_grounded_instruct_pair,
    build_mrc_grounded_pair,
    build_grounded_wiki_pair,
    build_grounded_zhihu_pair,
    build_grounded_literature_pair,
    build_grounded_ape210k_pair,
    build_grounded_poetry_pair,
    build_grounded_news_pair,
)


def build_omni_corpus(
    output_path: str = "/content/omni_knowledge_corpus.jsonl",
    samples_per_source: int = 100000,
    local_moss_path: Optional[str] = None,
    shuffle_buffer_size: int = 50000,
    seed: int = 42,
    **kwargs,
):
    """
    Backward-compatibility wrapper for `build_omni_knowledge_corpus`.
    Ignores `local_moss_path` if provided.
    """
    if local_moss_path:
        print(
            f"ℹ️ Notice: 'local_moss_path={local_moss_path}' is ignored in omni_knowledge_corpus. "
            "Please use `scripts/build_moss_dialogues.py` to build `moss_dialogues.jsonl`."
        )
    return build_omni_knowledge_corpus(
        output_path=output_path,
        samples_per_source=samples_per_source,
        shuffle_buffer_size=shuffle_buffer_size,
        seed=seed,
    )


def main():
    print("ℹ️ Notice: scripts/build_omni_corpus.py has been split. Delegating to scripts/build_omni_knowledge_corpus.py ...")

    parser = argparse.ArgumentParser(
        description="Build Grounded 8-Domain Omni Corpus for Phase 4 Continuous Flow Matching (Legacy Shim)"
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
        "--local_moss",
        type=str,
        default=None,
        help="[Deprecated/Ignored] Local MOSS jsonl path (MOSS is now built via scripts/build_moss_dialogues.py)",
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

    if args.local_moss:
        print(f"ℹ️ Note: --local_moss '{args.local_moss}' is ignored. Use scripts/build_moss_dialogues.py for MOSS data.")

    build_omni_corpus(
        output_path=args.output,
        samples_per_source=args.samples_per_source,
        local_moss_path=args.local_moss,
        shuffle_buffer_size=args.shuffle_buffer,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
