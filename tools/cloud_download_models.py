"""
【脚本功能】：云端算力节点的高速全能模型走私器 (Universal Multi-Backend Downloader)
【使用场景】：Phase 0.1 云端算力部署。绕过国外 GFW 限速，直接从中国镜像站（ModelScope 阿里的百兆内网）拉取动辄十几 GB 的超大模型。专门为了在新租借的 AutoDL / 恒源云 服务器上秒速部署而生。
【用法示例】：`python tools/cloud_download_models.py --repo_id qwen`
"""
import os
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Universal Flash-Speed Model Downloader for Cloud Instances")
    parser.add_argument("--repo_id", type=str, required=True, 
                        help="Repository ID (e.g. 'iic/gte_Qwen2-7B-instruct') or a registered shortcut ('qwen', 'bge').")
    parser.add_argument("--cache_dir", type=str, default="./models", help="Local directory to shard the massive weights into.")
    parser.add_argument("--backend", type=str, choices=["modelscope", "huggingface"], default="modelscope", 
                        help="Backend registry. Mainland machines strongly prefer 'modelscope' (100MB/s). Global use 'huggingface'.")
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    
    # Internal Alias Mapping (Convenience Shortcuts)
    model_mapping = {
        "bge": ("BAAI/bge-m3", "BAAI/bge-m3"), # (ModelScope ID, HuggingFace ID)
        "qwen": ("iic/gte_Qwen2-7B-instruct", "Alibaba-NLP/gte-Qwen2-7B-instruct")
    }
    
    ms_repo = args.repo_id
    hf_repo = args.repo_id
    
    if args.repo_id.lower() in model_mapping:
        ms_repo, hf_repo = model_mapping[args.repo_id.lower()]
        
    target_repo = ms_repo if args.backend == "modelscope" else hf_repo
        
    print(f"\n[INIT] Engaging {args.backend.upper()} Carrier Node target: {target_repo}...")
    print(f"Targeting Local Deep-Cache: {os.path.abspath(args.cache_dir)}\n")
    
    try:
        if args.backend == "modelscope":
            from modelscope import snapshot_download
            model_dir = snapshot_download(target_repo, cache_dir=args.cache_dir)
        else:
            from huggingface_hub import snapshot_download
            model_dir = snapshot_download(repo_id=target_repo, cache_dir=args.cache_dir)
            
        print(f"\n[SUCCESS] Matrix securely established and cached at: {model_dir}")
        print("You are cleared to engage the Extraction Engine!")
    except ImportError:
        print(f"\n[FATAL] Missing required library: pip install {args.backend}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[FATAL] Multi-threaded Fetch failed. Drop/Error details:\n{e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
