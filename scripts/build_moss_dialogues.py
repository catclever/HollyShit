#!/usr/bin/env python3
"""
scripts/build_moss_dialogues.py
===============================
Dedicated MOSS Dialogue Corpus Builder.

Produces: `moss_dialogues.jsonl`

Streams raw MOSS conversational data from HuggingFace (`fnlp/moss-003-sft-data` or
`OpenMOSS-Team/moss-003-sft-data`), or parses a local raw JSON/JSONL file (`--local_raw`),
strips MOSS-specific tokens (`<|Inner Thoughts|>`, `<|Commands|>`, `<|Human|>`, `<|MOSS|>`),
cleans synthetic bracket tags, and outputs clean `(prompt, target)` dialogue pairs.

Output format per line:
  {
    "prompt": "<User message or contextual dialogue history>",
    "target": "<Cleaned assistant response>",
    "domain": "dialogue",
    "source": "moss",
    "category": "<category_if_available>",
    "conversation_id": "<conv_id_if_available>"
  }

Usage:
  # Stream from Hugging Face:
  python scripts/build_moss_dialogues.py \\
      --output data/Basic_ZH/moss_dialogues.jsonl \\
      --max_samples 100000

  # Or process a local raw file:
  python scripts/build_moss_dialogues.py \\
      --local_raw /path/to/moss_raw.jsonl \\
      --output data/Basic_ZH/moss_dialogues.jsonl \\
      --max_samples 100000
"""

import os
import re
import sys
import json
import argparse
from typing import Iterator, List, Tuple, Dict, Any, Optional, Union

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =========================================================================
# Text Cleaning & Normalization for MOSS
# =========================================================================

_STRIP_PUNCT_RE = re.compile(
    r"^[\s,，.。!！?？;；:：、\"'“”‘’()（）\[\]【】]+|"
    r"[\s,，.。!！?？;；:：、\"'“”‘’()（）\[\]【】]+$"
)
_LEADING_TAG_RE = re.compile(r"^\s*【[^】]{1,20}】[：:\s]*")


def clean_text(text: Optional[str]) -> str:
    """
    Cleans whitespace and strips artificial bracket tags.
    """
    if not text or not isinstance(text, str):
        return ""
    text = text.strip()
    while True:
        new_text = _LEADING_TAG_RE.sub("", text).strip()
        if new_text == text:
            break
        text = new_text
    text = re.sub(r"【([^】]*)】", r"\1", text)
    text = text.replace("【", "").replace("】", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_moss_text(text: Optional[str]) -> str:
    """
    Strips MOSS special control tokens, thoughts, and command blocks:
      - `<|Inner Thoughts|>: ... <eoc>`
      - `<|Commands|>: ... <eoc>`
      - `<|Human|>:`, `<|MOSS|>:`, `<eoh>`, `<eom>`, `<eoc>`
    """
    if not text or not isinstance(text, str):
        return ""
    # Remove thoughts and command blocks
    text = re.sub(r"<\|Inner Thoughts\|>:.*?<eoc>", "", text, flags=re.DOTALL)
    text = re.sub(r"<\|Commands\|>:.*?<eoc>", "", text, flags=re.DOTALL)
    # Strip remaining special tags
    text = re.sub(
        r"<\|Human\|>:|<\|MOSS\|>:|<\|Commands\|>:|<\|Inner Thoughts\|>:|<eoh>|<eom>|<eoc>",
        "",
        text,
    )
    return clean_text(text)


def _ensure_sentence_end_punct(text: str) -> str:
    text = clean_moss_text(text)
    if not text:
        return text
    if re.search(r"[。！？.!?;；\"'”’）\)]$", text):
        return text
    has_cjk = bool(re.search(r"[\u4e00-\u9fff]", text))
    text = re.sub(r"[,，、:：]+$", "", text).strip()
    return text + ("。" if has_cjk else ".")


# =========================================================================
# Turn Extraction Helpers
# =========================================================================

def extract_moss_turns_from_item(
    item: Union[Dict[str, Any], str],
    max_prompt_chars: int = 240,
    max_target_chars: int = 240,
) -> List[Dict[str, Any]]:
    """
    Extracts one or more clean `(prompt, target)` dialogue pairs from a MOSS record.
    Supports:
      - Multi-turn `conversation` list (`[{"human": ..., "assistant": ...}, ...]`)
      - OpenAI / ShareGPT style `messages` or `chat`
      - Single-turn direct fields (`human`/`assistant`, `prompt`/`response`, `instruction`/`output`)
      - Raw text containing `<|Human|>: ... <|MOSS|>: ...`
    """
    results: List[Dict[str, Any]] = []

    if isinstance(item, str):
        raw_str = item.strip()
        if not raw_str:
            return []
        # Attempt JSON decoding if the string looks like serialized JSON
        if raw_str.startswith("{") and raw_str.endswith("}"):
            try:
                parsed = json.loads(raw_str)
                return extract_moss_turns_from_item(parsed, max_prompt_chars, max_target_chars)
            except Exception:
                pass
        # Fallback regex split on raw string
        if "<|Human|>:" in raw_str and "<|MOSS|>:" in raw_str:
            turns_raw = re.split(r"<\|Human\|>:", raw_str)
            for t_raw in turns_raw:
                if not t_raw.strip() or "<|MOSS|>:" not in t_raw:
                    continue
                parts = t_raw.split("<|MOSS|>:", 1)
                u_text = clean_moss_text(parts[0])
                a_text = clean_moss_text(parts[1])
                if len(u_text) >= 2 and len(a_text) >= 4 and u_text != a_text:
                    results.append({
                        "prompt": u_text[:max_prompt_chars],
                        "target": _ensure_sentence_end_punct(a_text[:max_target_chars]),
                        "domain": "dialogue",
                        "source": "moss",
                    })
            return results
        return []

    if not isinstance(item, dict):
        return []

    conv_id = item.get("conversation_id", item.get("id", None))
    category = item.get("category", item.get("task", None))

    # Case 1: Standard MOSS `conversation` list of turns
    if "conversation" in item and isinstance(item["conversation"], list):
        turns = item["conversation"]
        context_history: List[Tuple[str, str]] = []
        for turn_idx, turn in enumerate(turns):
            if not isinstance(turn, dict):
                continue
            u_raw = turn.get("human", turn.get("user", turn.get("input", turn.get("prompt", ""))))
            a_raw = turn.get("assistant", turn.get("moss", turn.get("bot", turn.get("output", turn.get("response", "")))))
            u_clean = clean_moss_text(u_raw)
            a_clean = clean_moss_text(a_raw)
            if len(u_clean) < 2 or len(a_clean) < 4 or u_clean == a_clean:
                continue

            # Build prompt with recent context if available
            if not context_history:
                prompt = u_clean[:max_prompt_chars]
            else:
                last_u, last_a = context_history[-1]
                cand_prompt = f"{last_u}\n{last_a}\n{u_clean}"
                if len(cand_prompt) <= max_prompt_chars:
                    prompt = cand_prompt
                else:
                    budget_for_prev = max(30, max_prompt_chars - len(u_clean) - 10)
                    prev_summary = f"{last_u[:budget_for_prev // 2]}...{last_a[:budget_for_prev // 2]}"
                    cand_prompt2 = f"{prev_summary}\n{u_clean}"
                    prompt = cand_prompt2[:max_prompt_chars] if len(cand_prompt2) <= max_prompt_chars else u_clean[:max_prompt_chars]

            target = _ensure_sentence_end_punct(a_clean[:max_target_chars])
            turn_record = {
                "prompt": prompt,
                "target": target,
                "domain": "dialogue",
                "source": "moss",
            }
            if conv_id is not None:
                turn_record["conversation_id"] = conv_id
            if category is not None:
                turn_record["category"] = category
            turn_record["turn"] = turn_idx

            results.append(turn_record)
            context_history.append((u_clean, a_clean))

        if results:
            return results

    # Case 2: OpenAI / ShareGPT style `messages` or `chat`
    for list_key in ["messages", "chat", "turns", "dialogue"]:
        if list_key in item and isinstance(item[list_key], list):
            turns = item[list_key]
            curr_user = ""
            for t_idx, msg in enumerate(turns):
                if not isinstance(msg, dict):
                    continue
                role = str(msg.get("role", msg.get("from", ""))).lower()
                content = clean_moss_text(msg.get("content", msg.get("value", msg.get("text", ""))))
                if role in ["user", "human"]:
                    curr_user = content
                elif role in ["assistant", "bot", "moss", "model", "gpt"] and curr_user:
                    if len(curr_user) >= 2 and len(content) >= 4 and curr_user != content:
                        turn_record = {
                            "prompt": curr_user[:max_prompt_chars],
                            "target": _ensure_sentence_end_punct(content[:max_target_chars]),
                            "domain": "dialogue",
                            "source": "moss",
                        }
                        if conv_id is not None:
                            turn_record["conversation_id"] = conv_id
                        if category is not None:
                            turn_record["category"] = category
                        turn_record["turn"] = len(results)
                        results.append(turn_record)
                    curr_user = ""
            if results:
                return results

    # Case 3: Single-turn key pairs
    pair_candidates = [
        ("human", "assistant"),
        ("prompt", "response"),
        ("prompt", "target"),
        ("instruction", "output"),
        ("query", "response"),
    ]
    for k_p, k_t in pair_candidates:
        if k_p in item and k_t in item and item[k_p] and item[k_t]:
            inp = clean_moss_text(item.get("input", item.get("context", "")))
            p_val = clean_moss_text(item[k_p])
            t_val = clean_moss_text(item[k_t])
            if inp:
                p_val = f"{inp[:80]}\n{p_val}"
            if len(p_val) >= 2 and len(t_val) >= 4 and p_val != t_val:
                rec = {
                    "prompt": p_val[:max_prompt_chars],
                    "target": _ensure_sentence_end_punct(t_val[:max_target_chars]),
                    "domain": "dialogue",
                    "source": "moss",
                }
                if conv_id is not None:
                    rec["conversation_id"] = conv_id
                if category is not None:
                    rec["category"] = category
                results.append(rec)
                return results

    return results


# =========================================================================
# Stream Sources (Hugging Face & Local Raw)
# =========================================================================

def stream_moss_from_hf(
    source_repo: str = "fnlp/moss-003-sft-data",
    max_samples: int = 100000,
) -> Iterator[Dict[str, Any]]:
    """
    Streams from HuggingFace dataset `source_repo` with fallback to `OpenMOSS-Team/moss-003-sft-data`.
    """
    from datasets import load_dataset
    repos_to_try = [source_repo]
    if "moss-003-sft-data" in source_repo and "OpenMOSS-Team" not in source_repo:
        repos_to_try.append("OpenMOSS-Team/moss-003-sft-data")

    count = 0
    for repo_name in repos_to_try:
        print(f">>> [MOSS] Attempting to stream from HuggingFace: {repo_name} ...")
        try:
            ds = load_dataset(repo_name, split="train", streaming=True)
            for item in ds:
                turns = extract_moss_turns_from_item(item)
                for turn in turns:
                    yield turn
                    count += 1
                    if count >= max_samples:
                        print(f"  -> Reached target {count} samples from {repo_name}.")
                        return
            print(f"  -> Finished stream from {repo_name} with {count} samples.")
            if count > 0:
                return
        except Exception as e:
            print(f"⚠️ Warning during stream from {repo_name}: {e}")

    print(f"  -> Total extracted {count} samples from Hugging Face MOSS repos.")


def stream_moss_from_local(
    local_raw_path: str,
    max_samples: int = 100000,
) -> Iterator[Dict[str, Any]]:
    """
    Reads from a local raw JSON or JSONL file.
    """
    print(f">>> [MOSS] Reading local raw MOSS file: {local_raw_path} ...")
    if not os.path.isfile(local_raw_path):
        print(f"❌ Error: Local file not found: {local_raw_path}")
        return

    count = 0
    # Try reading as full JSON first (e.g. JSON array)
    try:
        with open(local_raw_path, "r", encoding="utf-8") as f:
            first_char = f.read(1).strip()
            f.seek(0)
            if first_char == "[":
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        turns = extract_moss_turns_from_item(item)
                        for turn in turns:
                            yield turn
                            count += 1
                            if count >= max_samples:
                                print(f"  -> Reached target {count} samples from local JSON array.")
                                return
                    print(f"  -> Finished local JSON array with {count} samples.")
                    return
    except Exception:
        # Fall through to JSONL parsing
        pass

    # Read line-by-line as JSONL
    with open(local_raw_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                item = json.loads(line_str)
                turns = extract_moss_turns_from_item(item)
            except Exception:
                turns = extract_moss_turns_from_item(line_str)

            for turn in turns:
                yield turn
                count += 1
                if count >= max_samples:
                    print(f"  -> Reached target {count} samples from local JSONL.")
                    return

    print(f"  -> Finished local raw processing with {count} samples.")


# =========================================================================
# Main Builder
# =========================================================================

def build_moss_dialogues(
    output_path: str = "data/Basic_ZH/moss_dialogues.jsonl",
    max_samples: int = 100000,
    source_repo: str = "fnlp/moss-003-sft-data",
    local_raw: Optional[str] = None,
):
    """
    Builds the cleaned MOSS dialogues dataset and saves it to `output_path`.
    """
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print("==================================================================")
    print(f"🚀 Starting MOSS Dialogue Turn Corpus Assembly -> {output_path}")
    print(f"🎯 Target Samples: {max_samples}")
    if local_raw:
        print(f"📂 Source Mode: Local Raw File ({local_raw})")
    else:
        print(f"🌐 Source Mode: HuggingFace Stream ({source_repo})")
    print("==================================================================")

    streamer = (
        stream_moss_from_local(local_raw, max_samples=max_samples)
        if local_raw
        else stream_moss_from_hf(source_repo=source_repo, max_samples=max_samples)
    )

    total_written = 0
    with open(output_path, "w", encoding="utf-8") as out_f:
        for item in streamer:
            out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
            total_written += 1
            if total_written % 10000 == 0:
                print(f"  💾 Written {total_written} dialogue turns...")

    print("\n==================================================================")
    print("🎉 MOSS Dialogue Corpus Assembly Complete!")
    print(f"📦 Total Samples Saved: {total_written}")
    print(f"📁 Output: {output_path}")
    if os.path.exists(output_path):
        print(f"📊 Size: {os.path.getsize(output_path) / (1024 * 1024):.2f} MB")
    print("==================================================================")


def main():
    parser = argparse.ArgumentParser(
        description="Build MOSS Cleaned Dialogue Turns Corpus (moss_dialogues.jsonl)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/Basic_ZH/moss_dialogues.jsonl",
        help="Output JSONL file path (default: data/Basic_ZH/moss_dialogues.jsonl)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=100000,
        help="Maximum number of dialogue turns to extract (default: 100000)",
    )
    parser.add_argument(
        "--source_repo",
        type=str,
        default="fnlp/moss-003-sft-data",
        help="HuggingFace repository to stream from (default: fnlp/moss-003-sft-data)",
    )
    parser.add_argument(
        "--local_raw",
        type=str,
        default=None,
        help="Path to local raw JSON or JSONL file to process instead of HF streaming",
    )
    args = parser.parse_args()

    build_moss_dialogues(
        output_path=args.output,
        max_samples=args.max_samples,
        source_repo=args.source_repo,
        local_raw=args.local_raw,
    )


if __name__ == "__main__":
    main()
