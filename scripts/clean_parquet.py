"""
【脚本功能】：针对遗留脏数据的靶向清洗器
【使用场景】：Phase 0 数据准备阶段。专门用于洗掉旧维基数据中带有“答案：分段：”这一类被大模型生成的结构污染的部分，将它们复原为干净的连续自然段。
【用法示例】：`python scripts/clean_parquet.py`
"""
import pandas as pd
import re
import os

def clean_target_segments(text):
    """
    Looks specifically for the noisy format containing '答案：' and '分段：'.
    If found, it cleans it by extracting only the segments and joining them
    with newlines, making it look like normal multi-turn dialogue.
    All other structured or unstructured text (Wiki, normal dialogues) 
    is returned EXACTLY as it was.
    """
    if not isinstance(text, str):
        return text
        
    # Check if this holds the specific dirty structure
    if '答案：' in text and '分段：' in text:
        parts = text.split('分段：')
        # Part 0 is the space-separated prompt + "答案：", discard it
        # Extract the pure sentences
        segments = [p.strip() for p in parts[1:]]
        segments = [s for s in segments if s]
        
        # Join back exactly like normal newline-separated dialogue
        return '\n'.join(segments)
        
    # For everything else, touch nothing!
    return text

def main():
    input_path = 'data/Basic_ZH/mixed_wiki_full_1to1.parquet'
    output_path = 'data/Basic_ZH/cleaned_mixed_wiki.parquet'
    
    print(f"Loading data from {input_path}...")
    df = pd.read_parquet(input_path)
    print(f"Loaded {len(df)} rows.")
    
    print("Selectively cleaning ONLY the '分段：' format data...")
    # Apply the targeted cleaning
    df['text'] = df['text'].apply(clean_target_segments)
    
    print(f"Sample of data after targeted cleaning:")
    for i, txt in enumerate(df['text'].head(5)):
        print(f"\n--- Row {i} ---")
        # Print first 200 chars to verify
        print(txt[:200])
        
    print(f"\nSaving exactly {len(df)} rows back to {output_path}...")
    df.to_parquet(output_path, engine='pyarrow', index=False)
    print("Done! The dataset retains its original mixed multi-modal text shape.")

if __name__ == '__main__':
    main()
