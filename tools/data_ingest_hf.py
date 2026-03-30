"""
【脚本功能】：HuggingFace 防封锁原子拉取引擎 (Robust Dataset Ingester)
【使用场景】：Phase 0 数据准备。放弃了脆弱的流式读取，直接调用底层 `huggingface_hub` 原生的 Byte-level 断点续传技术。即使在不开代理、网络频繁闪断的环境下，也能像牛皮糖一样把海外极其珍贵的数百兆领域数据集一块块抗回本地。
【用法示例】：`python tools/data_ingest_hf.py --recipe data/Basic_ZH/omni_spices_recipe.json --output_dir data/Basic_ZH/raw_spices`
"""
import os
import json
import argparse
from datasets import load_dataset
from tqdm import tqdm

def save_jsonl(records: list[dict], output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[SUCCESS] Saved {len(records)} raw records to {output_path}")

def download_hf_dataset(dataset_name: str, config_name: str, split: str, num_samples: int, output_dir: str, output_file: str):
    print(f"\n--- Downloading {dataset_name} ({config_name if config_name else 'default'}) ---")
    
    import time
    import os
    final_path = os.path.join(output_dir, output_file)
    os.makedirs(output_dir, exist_ok=True)
    
    max_retries = 100
    for attempt in range(max_retries):
        existing_count = 0
        if os.path.exists(final_path):
            with open(final_path, 'r', encoding='utf-8') as f:
                existing_count = sum(1 for _ in f)
                
        if existing_count >= num_samples:
            print(f"[SUCCESS] {dataset_name} ({config_name}) already fully downloaded ({existing_count} records).")
            break
            
        print(f"Downloading {dataset_name} (Attempt {attempt+1}) using robust native HF byte-level caching")
        
        try:
            if config_name:
                ds = load_dataset(dataset_name, config_name, split=split, trust_remote_code=True)
            else:
                ds = load_dataset(dataset_name, split=split, trust_remote_code=True)
                
            records = []
            for i, row in enumerate(tqdm(ds, total=num_samples, desc=f"Extracting: {dataset_name}")):
                if i >= num_samples:
                    break
                records.append(row)
                
            # Atomic rewrite
            save_jsonl(records, final_path)
            break # Success, exit retry loop
            
        except Exception as e:
            print(f"[ERROR] Connection dropped: {e}")
            print("Retrying from breakpoint in 5 seconds (native HF cache will resume partial files automatically)...")
            time.sleep(5)

def main():
    parser = argparse.ArgumentParser(description="Generic HuggingFace Dataset Downloader")
    parser.add_argument("--recipe", type=str, help="Path to a JSON file containing a list of dataset configs to download in bulk.")
    parser.add_argument("--output_dir", type=str, default="data/Basic_ZH/raw_spices", help="Global directory to save all downloaded dataset files.")
    
    # Standalone specific parameters (ignored if --recipe is used)
    parser.add_argument("--dataset", type=str, help="HF dataset name (e.g. BAAI/COIG-CQIA)")
    parser.add_argument("--config", type=str, default=None, help="Dataset config/subset name (e.g. xiaohongshu)")
    parser.add_argument("--split", type=str, default="train", help="Dataset split (default: train)")
    parser.add_argument("--num_samples", type=int, default=10000, help="Number of samples to stream and save")
    parser.add_argument("--output_file", type=str, help="Output JSONL filename (e.g. raw_legal.jsonl)")
    
    args = parser.parse_args()
    
    # Ensure the global output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Global Output Directory is set to: {args.output_dir}")
    
    if args.recipe:
        print(f"Loading batch download recipe from {args.recipe}...")
        with open(args.recipe, "r", encoding="utf-8") as f:
            recipe_list = json.load(f)
            
        for job in recipe_list:
            download_hf_dataset(
                dataset_name=job.get('dataset'),
                config_name=job.get('config'),
                split=job.get('split', 'train'),
                num_samples=job.get('num_samples', 10000),
                output_dir=args.output_dir,
                output_file=job.get('output_file')
            )
        print("\n[SUCCESS] Entire batch recipe completed!")
    elif args.dataset and args.output_file:
        download_hf_dataset(
            dataset_name=args.dataset,
            config_name=args.config,
            split=args.split,
            num_samples=args.num_samples,
            output_dir=args.output_dir,
            output_file=args.output_file
        )
    else:
        print("Please provide either --recipe <json_file> OR --dataset <name> and --output_file <filename>.")

if __name__ == "__main__":
    main()
