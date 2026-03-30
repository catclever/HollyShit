"""
【脚本功能】：混合字级/字节级神圣词表构建器 (Hybrid Char-Byte Vocab Builder)
【使用场景】：初始化阶段。扫描所有训练素材，统计出频率最高的 TopN 个非 ASCII 中文字符（由于英文字符被退化为 0-255 的基础字节处理），并与特殊标志符合并，生成极其紧凑的 8000 维中文词库。
【用法示例】：`python tools/build_char_vocab.py`
"""
import os
import json
import collections
import pandas as pd
from tqdm import tqdm

def build_vocabulary(
    parquet_path="data/Basic_ZH/chunked_mixed_wiki.parquet",
    output_path="data/Basic_ZH/char_vocab.json",
    vocab_size=8000
):
    print(f"1. Loading dataset from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    text_chunks = df['chunks'].explode().dropna().tolist()
    print(f"   => Loaded {len(text_chunks)} text samples.")

    print("2. Scanning characters (Ignoring ASCII which is handled by raw Bytes)...")
    counter = collections.Counter()
    
    for text in tqdm(text_chunks, desc="Counting frequencies"):
        for char in text:
            # If the character's unicode point is < 128, it's pure ASCII.
            # It will be natively handled by the 0-255 Byte Foundation.
            if ord(char) < 128:
                continue
            counter[char] += 1

    print(f"   => Found {len(counter)} unique Non-ASCII characters (Chinese/Full-width).")

    # We need to reserve space for:
    # - 256 Raw Bytes (0-255)
    # - 4 Special Tokens (PAD, UNK, BOS, EOS)
    reserved_spaces = 256 + 4
    available_slots = vocab_size - reserved_spaces
    print(f"3. Taking the Top {available_slots} most frequent non-ASCII characters...")

    top_chars = [char for char, freq in counter.most_common(available_slots)]

    print(f"4. Assembling the Ultimate Hybrid Dictionary...")
    
    # 结构: [256个字节] + [4个特殊符号] + [其余高频汉字]
    # 我们用特定的字符串表示 Byte 和 Special Tokens 来避免 JSON 崩溃。
    vocab_dict = {}
    current_id = 0

    # 4.1 加入 256 个地基字节
    for b in range(256):
        vocab_dict[f"<0x{b:02X}>"] = current_id
        current_id += 1

    # 4.2 加入特殊符号
    special_tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
    for st in special_tokens:
        vocab_dict[st] = current_id
        current_id += 1

    # 4.3 加入高频汉字/中文标点
    for char in top_chars:
        # 防死锁保护（通常不可能，因为我们过滤了 ASCII）
        if char not in vocab_dict:
            vocab_dict[char] = current_id
            current_id += 1

    real_vocab_size = len(vocab_dict)
    print(f"5. Saving Vocabulary (Size: {real_vocab_size}) to {output_path}...")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(vocab_dict, f, ensure_ascii=False, indent=2)

    print("=========================================")
    print(f"          VOCABULARY BUILT!              ")
    print(f"      Total Dictionary Size: {real_vocab_size}")
    print("=========================================")
    print("Sample entries:")
    sample_keys = list(vocab_dict.keys())
    print(f" - Byte examples: {sample_keys[0]} -> {vocab_dict[sample_keys[0]]}")
    print(f" - Special examples: {special_tokens[0]} -> {vocab_dict[special_tokens[0]]}")
    print(f" - Top character 1: '{top_chars[0]}' -> {vocab_dict[top_chars[0]]}")
    print(f" - Top character 2: '{top_chars[1]}' -> {vocab_dict[top_chars[1]]}")

if __name__ == "__main__":
    build_vocabulary()
