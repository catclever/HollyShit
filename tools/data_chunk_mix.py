"""
【脚本功能】：全领域语料（Omni-Domain）万能装甲混合器
【使用场景】：Phase 0 数据准备。它能嗅探目录下被爬虫拉下来的所有垂直领域数据（如小红书、法律、金融等 JSONL），自动识别其所属科目并执行对应的前缀模板注入，最后与原本的维基打底数据集完美缝合并保存为大一统 Parquet。
【用法示例】：`python tools/data_chunk_mix.py --spice_dir ./raw_spices --base_parquet old.parquet --output_parquet omni.parquet`
"""
import os
import sys
import json
import glob
import argparse
import pandas as pd
from tqdm import tqdm

# Ensure scripts dir is importable
try:
    from scripts.chunk_and_eos import intelligent_chunking
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from scripts.chunk_and_eos import intelligent_chunking

def parse_cqia(file_path, prefix_name):
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc=f"Parsing {prefix_name}"):
            data = json.loads(line)
            # COIG-CQIA format: instruction, output
            text = f"【网友提问】：{data.get('instruction', '')}\n【高赞神回复】：{data.get('output', '')}"
            chunks = intelligent_chunking(text)
            if chunks:
                records.append({
                    "source": f"spice_cqia_{prefix_name}",
                    "chunks": chunks,
                    "chunk_count": len(chunks)
                })
    return records

def parse_finance(file_path):
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Parsing Finance"):
            data = json.loads(line)
            # COIG-CQIA finance format
            text = f"【财经咨询】：{data.get('instruction', '')}\n【专业分析】：{data.get('output', '')}"
            chunks = intelligent_chunking(text)
            if chunks:
                records.append({
                    "source": "spice_finance",
                    "chunks": chunks,
                    "chunk_count": len(chunks)
                })
    return records

def parse_medical(file_path):
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Parsing Medical"):
            data = json.loads(line)
            # medical finetune format usually has "instruction", "output", and sometimes "input"
            req = data.get('instruction', '') + " " + data.get('input', '')
            text = f"【患者提问】：{req.strip()}\n【主任医生】：{data.get('output', '')}"
            chunks = intelligent_chunking(text)
            if chunks:
                records.append({
                    "source": "spice_medical",
                    "chunks": chunks,
                    "chunk_count": len(chunks)
                })
    return records

def parse_legal(file_path):
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Parsing Legal"):
            data = json.loads(line)
            # Chinese-Legal-Case-Classification format: 'instruction' and 'input'
            fact = data.get('instruction', '') + "\n" + data.get('input', '')
            if len(fact.strip()) > 10:
                text = f"【法庭纪要】：{fact}"
                chunks = intelligent_chunking(text)
                if chunks:
                    records.append({
                        "source": "spice_legal",
                        "chunks": chunks,
                        "chunk_count": len(chunks)
                    })
    return records

def main():
    parser = argparse.ArgumentParser(description="Omni-Domain Spice Transformer & Fuser")
    parser.add_argument("--spice_dir", type=str, required=True, help="Directory containing raw JSONL spices")
    parser.add_argument("--base_parquet", type=str, required=True, help="Path to the base Mamba parquet")
    parser.add_argument("--output_parquet", type=str, required=True, help="Path to save the fused output")
    args = parser.parse_args()
    
    print("=== Phase 0.0b: Omni-Domain Transformation & Fusion ===")
    
    all_records = []
    
    jsonl_files = glob.glob(os.path.join(args.spice_dir, "*.jsonl"))
    if not jsonl_files:
        print(f"[WARNING] No .jsonl files found in {args.spice_dir}!")
        
    for cf in jsonl_files:
        basename = os.path.basename(cf)
        if basename.startswith("m-a-p_COIG-CQIA_"):
            subset_name = basename.replace("m-a-p_COIG-CQIA_", "").replace(".jsonl", "")
            if subset_name == "finance":
                all_records.extend(parse_finance(cf))
            else:
                all_records.extend(parse_cqia(cf, subset_name))
        elif basename.startswith("shibing624_medical"):
            all_records.extend(parse_medical(cf))
        elif basename.startswith("gehits_Chinese-Legal"):
            all_records.extend(parse_legal(cf))
        else:
            print(f"[SKIP] Unknown mapping logic for file: {basename}")
        
    print(f"\nExtracted {len(all_records)} valid formatted Spices!")
    
    spice_df = pd.DataFrame(all_records)
    if not spice_df.empty:
        print("Spice columns:", spice_df.columns.tolist())
    
    print(f"\nLoading base world-model chunks from {args.base_parquet}...")
    try:
        base_df = pd.read_parquet(args.base_parquet)
        print(f"Base rows: {len(base_df)}")
    except Exception as e:
        print(f"[ERROR] Failed to load base parquet: {e}")
        return
    
    if not spice_df.empty:
        print("\nFusing Universes...")
        fusion_df = pd.concat([base_df, spice_df], ignore_index=True)
    else:
        print("\nNo spices fused. Saving base directly.")
        fusion_df = base_df
    
    print(f"Total rows in fusion Universe: {len(fusion_df)}")
    
    # Save!
    print(f"\nSaving Universal World Model Data to {args.output_parquet}...")
    fusion_df.to_parquet(args.output_parquet, engine='pyarrow')
    print("[SUCCESS] Done!")
    
    print("\n--- Quick Sanity Check Sample ---")
    print(fusion_df.iloc[-1]['chunks'][:3])

if __name__ == '__main__':
    main()
