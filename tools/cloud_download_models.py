import os
import argparse
try:
    from modelscope import snapshot_download as ms_snapshot_download
except ImportError:
    print("Please install modelscope first: pip install modelscope")
    exit(1)
try:
    from huggingface_hub import snapshot_download as hf_snapshot_download
except ImportError:
    hf_snapshot_download = None

MODEL_ALIASES = {
    "bge": "BAAI/bge-m3",
    "qwen": "iic/gte_Qwen2-7B-instruct",
    "youtu": "tencent/Youtu-Embedding",
    "tencent/youtu": "tencent/Youtu-Embedding",
    "tencent/youtu-embedding": "tencent/Youtu-Embedding",
}

def main():
    parser = argparse.ArgumentParser(description="Flash-Speed Model Downloader for Cloud Rentals (ModelScope Native)")
    parser.add_argument("--model", type=str, required=True, help="Which model to download: 'bge', 'qwen' or full repo id")
    parser.add_argument("--cache_dir", type=str, default="./models", help="Directory to save the massive model shards")
    parser.add_argument("--hf_mirror", type=str, default=os.environ.get("HF_MIRROR", os.environ.get("HF_ENDPOINT", "")), help="HF 镜像地址，优先级: --hf_mirror > HF_MIRROR > HF_ENDPOINT")
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    
    ms_repo = None
    hf_repo = None
    model_input = args.model.strip()
    model_norm = model_input.lower()
    resolved = MODEL_ALIASES.get(model_norm, model_input)
    if resolved in {"BAAI/bge-m3", "iic/gte_Qwen2-7B-instruct"}:
        ms_repo = resolved
    elif "/" in resolved:
        hf_repo = resolved
    else:
        ms_repo = resolved

    target_repo = ms_repo or hf_repo
    print(f"\n[INIT] Engaging Cloud-Native Downloader for {target_repo}...")
    print(f"Targeting Local Cache Directory: {os.path.abspath(args.cache_dir)}\n")

    if ms_repo is not None:
        try:
            model_dir = ms_snapshot_download(ms_repo, cache_dir=args.cache_dir)
            print(f"\n[SUCCESS] Model completely downloaded and cached at: {model_dir}")
            print("You are cleared to engage the Extraction Engine!")
            return
        except Exception as e:
            print(f"[WARN] ModelScope download failed for {ms_repo}: {e}")
            if "/" in ms_repo:
                hf_repo = ms_repo

    if hf_repo is not None:
        if hf_snapshot_download is None:
            print("\n[FATAL] huggingface_hub is required for HF fallback. Install with: pip install -U huggingface_hub")
            return
        local_dir = os.path.join(args.cache_dir, hf_repo)
        hf_kwargs = {"repo_id": hf_repo, "local_dir": local_dir, "local_dir_use_symlinks": False}
        hf_mirror = (args.hf_mirror or "").strip().rstrip("/")
        if hf_mirror:
            hf_kwargs["endpoint"] = hf_mirror
            print(f"[HF] Using mirror endpoint: {hf_mirror}")
        try:
            model_dir = hf_snapshot_download(**hf_kwargs)
        except Exception as e:
            print(f"\n[FATAL] HF download failed for {hf_repo}: {e}")
            if model_norm in {"youtu", "tencent/youtu", "tencent/youtu-embedding"}:
                print("可用仓库ID请使用: tencent/Youtu-Embedding")
            return
        print(f"\n[SUCCESS] Model completely downloaded and cached at: {model_dir}")
        print("You are cleared to engage the Extraction Engine!")
        return

    print(f"\n[FATAL] Model Fetch failed. Unsupported model id: {args.model}")

if __name__ == "__main__":
    main()
