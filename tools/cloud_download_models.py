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
    parser.add_argument("--cache_dir", type=str, default=None, help="Directory to save the massive model shards. Priorities: 1. This arg, 2. HF_HOME/hub, 3. ./models")
    parser.add_argument("--hf_mirror", type=str, default=os.environ.get("HF_MIRROR", os.environ.get("HF_ENDPOINT", "")), help="HF 镜像地址，优先级: --hf_mirror > HF_MIRROR > HF_ENDPOINT")
    args = parser.parse_args()

    # Cache Directory Resolution Logic
    if args.cache_dir:
        final_cache_dir = args.cache_dir
    else:
        hf_home = os.environ.get("HF_HOME")
        if hf_home:
            final_cache_dir = os.path.join(hf_home, "hub") if not hf_home.endswith("hub") else hf_home
        else:
            default_hf_hub = os.path.expanduser("~/.cache/huggingface/hub")
            # If ~/.cache/huggingface exists, assume the user uses the standard HF cache
            if os.path.exists(os.path.expanduser("~/.cache/huggingface")):
                final_cache_dir = default_hf_hub
            else:
                final_cache_dir = "./models"

    os.makedirs(final_cache_dir, exist_ok=True)
    
    model_input = args.model.strip()
    model_norm = model_input.lower()
    target_repo = MODEL_ALIASES.get(model_norm, model_input)

    print(f"\n[INIT] Engaging Cloud-Native Downloader for {target_repo}...")
    print(f"Targeting Local Cache Directory: {os.path.abspath(final_cache_dir)}\n")

    # 1. First Attempt: ModelScope (Fastest in CN)
    print("[1/2] Attempting ModelScope Download...")
    try:
        model_dir = ms_snapshot_download(target_repo, cache_dir=final_cache_dir)
        print(f"\n[SUCCESS] Model completely downloaded via ModelScope at: {model_dir}")
        print("You are cleared to engage the Extraction Engine!")
        return
    except Exception as e:
        print(f"[WARN] ModelScope download failed (or model doesn't exist) for {target_repo}.")
        print(f"       Reason: {e}")
        print("       -> Falling back to HuggingFace Mirror...")

    # 2. Second Attempt: HuggingFace (Fallback)
    if hf_snapshot_download is None:
        print("\n[FATAL] huggingface_hub is required for HF fallback. Install with: pip install -U huggingface_hub")
        return
    
    # HuggingFace naturally manages its own cache structures (e.g., models--repo--id)
    hf_kwargs = {"repo_id": target_repo, "cache_dir": final_cache_dir, "local_dir_use_symlinks": False}
    hf_mirror = (args.hf_mirror or "").strip().rstrip("/")
    if hf_mirror:
        hf_kwargs["endpoint"] = hf_mirror
        print(f"[HF] Using mirror endpoint: {hf_mirror}")
    try:
        model_dir = hf_snapshot_download(**hf_kwargs)
        print(f"\n[SUCCESS] Model completely downloaded via HuggingFace at: {model_dir}")
        print("You are cleared to engage the Extraction Engine!")
        return
    except Exception as e:
        print(f"\n[FATAL] HF download failed for {target_repo}: {e}")
        if model_norm in {"youtu", "tencent/youtu", "tencent/youtu-embedding"}:
            print("可用仓库ID请使用: tencent/Youtu-Embedding")
        return

if __name__ == "__main__":
    main()
