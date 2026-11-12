import argparse
import glob
import json
import math
import os
import re
import shutil
import sys
import threading
import time
import zipfile
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Auto-resolve the parent workspace root and force it to the FRONT of the import path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir in sys.path and os.path.exists(os.path.join(script_dir, "model.py")):
    sys.path = [p for p in sys.path if p != script_dir] + [script_dir]

from training.data.alignment_dataloader import Phase1_ChunkedNpzDataLoader, Phase1_MultiEmbDataLoader
from training.core.char_tokenizer import CharTokenizer
from training.core.checkpoint import Checkpointer
from model.config import WeakDecoderConfig
from model.phase1_charencoder_cuda import (
    TinyCharEncoderCUDA,
    load_mlx_safetensors_into_torch,
)
from model.god_encoder_cuda import GodEncoderCUDA
from model.adapter_cuda import SensoryFuserCUDA
from training.losses.char_encoder_v2_loss import (
    AsymmetricNextClauseHeadCUDA,
    CharSubspaceAnchorHeadCUDA,
    OrderDirectionHeadCUDA,
    compute_phase1_v2_distill_loss,
    evaluate_phase1_v2_health,
)


def is_colab_drive_mounted(
    mount_root: str = "/content/drive",
    mounts_file: str = "/proc/mounts",
) -> bool:
    """
    Returns True iff `mount_root` is actively mounted as a filesystem (`os.path.ismount`
    or `mounts_file` contains a mountpoint equal to `mount_root` or under `mount_root + "/"`).
    """
    if not mount_root:
        return False
    norm_root = os.path.abspath(mount_root).rstrip("/") or "/"
    real_root = os.path.realpath(mount_root).rstrip("/") or "/"
    root_variants = {norm_root, real_root}

    try:
        if os.path.exists(mount_root) and os.path.ismount(mount_root):
            return True
    except OSError:
        pass

    if mounts_file and os.path.isfile(mounts_file):
        try:
            with open(mounts_file, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) < 2:
                        continue
                    mp = parts[1].replace("\\040", " ").rstrip("/") or "/"
                    for r in root_variants:
                        if mp == r or mp.startswith(r + "/"):
                            return True
        except OSError:
            pass

    return False


def _collect_resolved_target_paths(target_path: str) -> List[str]:
    abs_target = os.path.abspath(target_path)
    candidates = [abs_target, os.path.realpath(abs_target)]

    cur = abs_target
    suffix_parts: List[str] = []
    while cur and cur != os.path.dirname(cur):
        if os.path.islink(cur):
            try:
                raw_link = os.readlink(cur)
                link_abs = (
                    os.path.abspath(raw_link)
                    if os.path.isabs(raw_link)
                    else os.path.abspath(os.path.join(os.path.dirname(cur), raw_link))
                )
                joined_abs = (
                    os.path.abspath(os.path.join(link_abs, *reversed(suffix_parts)))
                    if suffix_parts
                    else link_abs
                )
                candidates.extend(
                    [link_abs, os.path.realpath(cur), joined_abs, os.path.realpath(joined_abs)]
                )
            except OSError:
                pass
        suffix_parts.append(os.path.basename(cur))
        cur = os.path.dirname(cur)

    seen = set()
    ordered: List[str] = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    return ordered


def assert_colab_drive_mounted(
    target_path: str,
    mount_root: str = "/content/drive",
    mounts_file: str = "/proc/mounts",
) -> None:
    """
    Hard safety gate preventing silent checkpoint loss on unmounted Colab runtimes:
    If `target_path` (or any symlink ancestor's resolved target) points inside `mount_root`,
    verifies `is_colab_drive_mounted(mount_root, mounts_file=mounts_file)` and immediately
    raises `RuntimeError` if `mount_root` is NOT mounted.
    """
    if not target_path or not mount_root:
        return

    norm_root = os.path.abspath(mount_root).rstrip("/") or "/"
    real_root = os.path.realpath(mount_root).rstrip("/") or "/"
    root_variants = {norm_root, real_root}

    resolved_candidates = _collect_resolved_target_paths(target_path)
    matched_drive_paths: List[str] = []
    for cand in resolved_candidates:
        cand_norm = cand.rstrip("/") or "/"
        for r in root_variants:
            if cand_norm == r or cand_norm.startswith(r + "/"):
                matched_drive_paths.append(cand)
                break

    if not matched_drive_paths:
        return

    if not is_colab_drive_mounted(mount_root=mount_root, mounts_file=mounts_file):
        raise RuntimeError(
            f"❌ [Colab Mount Guard] 致命安全拦截：目标路径 {target_path!r} "
            f"(解析路径: {matched_drive_paths[0]!r}) 指向 Google Drive 挂载区 {mount_root!r}，"
            f"但检测到 {mount_root!r} 当前并未挂载！(os.path.ismount 与 {mounts_file} 均未命中)\n"
            f"👉 已拒绝在 Colab 容器本地临时 SSD 创建伪目录写入权重（断开运行时将导致权重永久丢失）。\n"
            f"👉 解决方法：请先在 Colab 单元格执行 `from google.colab import drive; drive.mount('/content/drive')` 并完成授权！"
        )


def _get_hf_token() -> Optional[str]:
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or ""
    tok = tok.strip()
    return tok if tok else None


def _cleanup_hf_download_metadata(file_path: str, roots: Sequence[Optional[str]]) -> None:
    if not file_path:
        return
    abs_p = os.path.abspath(file_path)
    seen_roots = set()
    for root in roots:
        if not root:
            continue
        abs_root = os.path.abspath(root)
        if abs_root in seen_roots:
            continue
        seen_roots.add(abs_root)
        prefix = abs_root.rstrip(os.sep) + os.sep
        if not abs_p.startswith(prefix):
            continue
        rel = os.path.relpath(abs_p, abs_root)
        dl_meta_dir = os.path.join(
            abs_root, ".cache", "huggingface", "download", os.path.dirname(rel)
        )
        base_fn = os.path.basename(rel)
        if not os.path.isdir(dl_meta_dir):
            continue
        try:
            for fn in os.listdir(dl_meta_dir):
                is_match = fn in {
                    f"{base_fn}.metadata",
                    f"{base_fn}.incomplete",
                    f"{base_fn}.lock",
                } or (
                    fn.startswith(base_fn + ".")
                    and fn.endswith((".metadata", ".incomplete", ".lock"))
                )
                if is_match:
                    try:
                        os.remove(os.path.join(dl_meta_dir, fn))
                    except OSError:
                        pass
        except OSError:
            pass


def _download_hf_dataset_chunk_file(
    repo_id: str,
    filename: str,
    local_dir: str,
    cache_dir: Optional[str] = None,
) -> str:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise RuntimeError("请安装 huggingface_hub: pip install huggingface_hub")

    hf_token = _get_hf_token()
    os.makedirs(local_dir, exist_ok=True)
    try:
        return hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            local_dir=local_dir,
            token=hf_token,
        )
    except TypeError:
        fallback_cache = cache_dir or local_dir
        os.makedirs(fallback_cache, exist_ok=True)
        return hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            cache_dir=fallback_cache,
            token=hf_token,
        )


def mirror_checkpoint_to_hf_hub(
    ckpt_dir: str,
    hf_ckpt_repo: Optional[str],
    include_optimizer: bool = False,
) -> bool:
    """
    Mirrors a saved Phase 1 v2 checkpoint directory to a private Hugging Face model repository
    under `p1_v2/<basename(ckpt_dir)>`. Non-fatal on network/API errors so training never crashes.
    """
    if not hf_ckpt_repo or not hf_ckpt_repo.strip():
        return False
    repo_id = hf_ckpt_repo.strip()
    try:
        from huggingface_hub import HfApi

        hf_token = _get_hf_token()
        api = HfApi(token=hf_token)
        api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
        folder_name = os.path.basename(os.path.normpath(ckpt_dir))
        ignore_patterns = [] if include_optimizer else ["optimizer.pt"]
        api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=ckpt_dir,
            path_in_repo=f"p1_v2/{folder_name}",
            ignore_patterns=ignore_patterns,
        )
        print(
            f"[HF-Backup] ☁️ Mirrored Phase 1 v2 checkpoint to https://huggingface.co/{repo_id}/tree/main/p1_v2/{folder_name}"
        )
        return True
    except Exception as e:
        print(f"[HF-Backup] ⚠️ Failed to mirror checkpoint {ckpt_dir} to {repo_id}: {e}")
        return False


def custom_lr_schedule(global_step: int, max_lr: float, warmup_steps: int):
    if warmup_steps <= 0:
        return max_lr
    if global_step < warmup_steps:
        return max_lr * (global_step / float(warmup_steps))
    return max_lr


def load_optional_config(config_file: str):
    if not config_file:
        return {}
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found: {config_file}")
    with open(config_file, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_path_maybe_relative(path_value: str, base_dir: str):
    if not path_value:
        return path_value
    if os.path.isabs(path_value):
        return path_value
    candidate = os.path.join(base_dir, path_value)
    if os.path.exists(candidate):
        return candidate
    return path_value


def _load_ckpt_state_dict(path: str) -> dict:
    if path.endswith(".pt") or path.endswith(".pth"):
        return torch.load(path, map_location="cpu", weights_only=True)
    from safetensors.torch import load_file
    return load_file(path)


def infer_teacher_arch_from_ckpt(p0_ckpt: str):
    fuser_candidates = [
        os.path.join(p0_ckpt, "sense_fuser.safetensors"),
        os.path.join(p0_ckpt, "sense_adapter.safetensors"),
        os.path.join(p0_ckpt, "sense_fuser.pt"),
        os.path.join(p0_ckpt, "sense_adapter.pt"),
    ]
    fuser_path = next((p for p in fuser_candidates if os.path.exists(p)), None)
    if fuser_path is None:
        raise FileNotFoundError(f"Missing sense_fuser/sense_adapter checkpoint under: {p0_ckpt}")

    god_candidates = [
        os.path.join(p0_ckpt, "god_encoder.safetensors"),
        os.path.join(p0_ckpt, "god_encoder.pt"),
    ]
    god_path = next((p for p in god_candidates if os.path.exists(p)), None)
    if god_path is None:
        raise FileNotFoundError(f"Missing god_encoder checkpoint under: {p0_ckpt}")

    fuser_state = _load_ckpt_state_dict(fuser_path)
    god_state = _load_ckpt_state_dict(god_path)

    route_dims = {}
    d_model = None
    for key, tensor in fuser_state.items():
        m = re.match(r"adapters\.(\d+)\.(?:net\.layers\.0|fc1)\.weight$", key)
        if m:
            idx = int(m.group(1))
            route_dims[idx] = int(tensor.shape[1])
            d_model = int(tensor.shape[0])

    if not route_dims:
        raise RuntimeError(f"Could not infer adapter routes from {fuser_path}")

    emb_dims = [route_dims[i] for i in sorted(route_dims.keys())]

    z_dim = None
    for key, tensor in god_state.items():
        if key in {"net.layers.2.weight", "fc2.weight"}:
            z_dim = int(tensor.shape[0])
            if d_model is None:
                d_model = int(tensor.shape[1])
            break

    if z_dim is None:
        raise RuntimeError(f"Could not infer z_dim from {god_path}")
    if d_model is None:
        raise RuntimeError(f"Could not infer d_model from {fuser_path}/{god_path}")

    return fuser_path, god_path, emb_dims, d_model, z_dim


def resolve_emb_paths(args, ds_config: dict, expected_routes: int):
    if args.emb_paths:
        emb_paths = args.emb_paths
    elif ds_config.get("emb_paths"):
        emb_paths = ds_config["emb_paths"]
    else:
        emb_paths = sorted(glob.glob(os.path.join(args.emb_dir, "*embeddings*.npy")))

    if len(emb_paths) != expected_routes:
        msg = (
            f"Embedding route mismatch: expected {expected_routes} from checkpoint, "
            f"but got {len(emb_paths)} embedding files. paths={emb_paths}"
        )
        if args.strict_route_check:
            raise ValueError(msg)
        if len(emb_paths) < expected_routes:
            raise ValueError(msg + " | Non-strict cannot recover from fewer routes.")
        print(f"[Warn] {msg} | Non-strict mode: truncating to first {expected_routes} paths.")
        emb_paths = emb_paths[:expected_routes]
    return emb_paths


def build_dataloader(args, tokenizer, ds_config: dict, expected_routes: int):
    ds_config = ds_config or {}
    has_chunked_conf = bool(ds_config.get("base_models") and ds_config.get("chunk_name_patterns"))
    config_base_dir = os.path.dirname(os.path.abspath(args.config_file)) if args.config_file else os.getcwd()
    parquet_path = args.parquet_path or ds_config.get("parquet_path", "data/Basic_ZH/chunked_mixed_wiki.parquet")
    parquet_path = resolve_path_maybe_relative(parquet_path, config_base_dir)

    if has_chunked_conf:
        models = ds_config["base_models"]
        chunk_patterns = ds_config["chunk_name_patterns"]

        raw_hf_arg = getattr(args, "hf_repo_id", None)
        hf_repo_id = (
            raw_hf_arg if raw_hf_arg is not None else ds_config.get("huggingface_repo_id")
        )
        if isinstance(hf_repo_id, str):
            hf_repo_id = hf_repo_id.strip()
        if not hf_repo_id:
            hf_repo_id = None

        raw_ms_arg = getattr(args, "ms_repo_id", None)
        ms_repo_id = (
            raw_ms_arg if raw_ms_arg is not None else ds_config.get("modelscope_repo_id")
        )
        if isinstance(ms_repo_id, str):
            ms_repo_id = ms_repo_id.strip()
        if not ms_repo_id:
            ms_repo_id = None

        effective_ms_repo_id = None if hf_repo_id else ms_repo_id

        if len(models) != expected_routes:
            msg = (
                f"Model route mismatch: checkpoint expects {expected_routes}, "
                f"but config has {len(models)} base_models={models}"
            )
            if args.strict_route_check:
                raise ValueError(msg)
            if len(models) < expected_routes:
                raise ValueError(msg + " | Non-strict cannot recover from fewer routes.")
            print(f"[Warn] {msg} | Non-strict mode: truncating to first {expected_routes} models.")
            models = models[:expected_routes]
            chunk_patterns = {k: v for k, v in chunk_patterns.items() if k in models}

        src_desc = (
            f"hf_repo_id={hf_repo_id}"
            if hf_repo_id
            else (
                f"ms_repo_id={effective_ms_repo_id}"
                if effective_ms_repo_id
                else f"local_npz_dir={args.data_dir}"
            )
        )
        print(f"[Data] Using Phase1_ChunkedNpzDataLoader ({len(models)} routes, {src_desc}).")
        return Phase1_ChunkedNpzDataLoader(
            parquet_path=parquet_path,
            models=models,
            chunk_patterns=chunk_patterns,
            tokenizer=tokenizer,
            ms_repo_id=effective_ms_repo_id,
            chunk_size=args.chunk_size,
            local_npz_dir=args.data_dir,
            cache_dir=args.download_cache_dir,
            batch_size=args.batch_size,
            max_seq_len=args.max_seq_len,
            shuffle=True,
            lazy_start=True,
            backend="torch",
            hf_repo_id=hf_repo_id,
        )

    emb_paths = resolve_emb_paths(args, ds_config, expected_routes)
    print(f"[Data] Using Phase1_MultiEmbDataLoader ({len(emb_paths)} routes): {emb_paths}")
    return Phase1_MultiEmbDataLoader(
        parquet_path=parquet_path,
        emb_paths=emb_paths,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        backend="torch",
    )


def attach_chunk_download_router(loader, local_data_dir: str, download_cache_dir: str):
    if not hasattr(loader, "models") or not hasattr(loader, "chunk_patterns"):
        return loader

    local_data_dir = os.path.abspath(local_data_dir)
    download_cache_dir = os.path.abspath(download_cache_dir)
    loader.npz_dir = local_data_dir
    loader.cache_dir = download_cache_dir
    _cand_locks: Dict[str, threading.Lock] = {}
    _cand_locks_guard = threading.Lock()
    _last_temp_cleanup_at = 0.0

    def _cleanup_stale_temp_artifacts(now_ts: float = None, stale_seconds: int = 600):
        nonlocal _last_temp_cleanup_at
        if now_ts is None:
            now_ts = time.time()
        if now_ts - _last_temp_cleanup_at < 30:
            return
        _last_temp_cleanup_at = now_ts

        roots = {
            os.path.join(local_data_dir, "._____temp"),
            os.path.join(download_cache_dir, "._____temp"),
            os.path.join(local_data_dir, ".cache", "huggingface", "download"),
            os.path.join(download_cache_dir, ".cache", "huggingface", "download"),
            local_data_dir,
            download_cache_dir,
        }
        for root in roots:
            if not root or not os.path.exists(root):
                continue
            for dp, _, fs in os.walk(root, topdown=False):
                for fn in fs:
                    fp = os.path.join(dp, fn)
                    try:
                        age = now_ts - os.path.getmtime(fp)
                    except OSError:
                        continue
                    if age < stale_seconds:
                        continue
                    is_temp = (
                        fn.endswith(".incomplete")
                        or fn.endswith(".tmp")
                        or fn.endswith(".partial")
                        or ".tmp." in fn
                    )
                    if is_temp:
                        try:
                            os.remove(fp)
                        except OSError:
                            pass

    def _get_cand_lock(cand_name: str):
        with _cand_locks_guard:
            if cand_name not in _cand_locks:
                _cand_locks[cand_name] = threading.Lock()
            return _cand_locks[cand_name]

    def _resolve_npz_path(cand_name: str):
        _cleanup_stale_temp_artifacts()
        local_path = os.path.join(local_data_dir, cand_name)
        cand_lock = _get_cand_lock(cand_name)
        with cand_lock:
            if os.path.exists(local_path):
                return local_path

            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            hf_repo = getattr(loader, "hf_repo_id", None)
            ms_repo = getattr(loader, "ms_repo_id", None)
            if hf_repo:
                last_err = None
                downloaded_path = None
                for _ in range(3):
                    try:
                        downloaded_path = _download_hf_dataset_chunk_file(
                            hf_repo,
                            cand_name,
                            local_dir=local_data_dir,
                            cache_dir=download_cache_dir,
                        )
                        break
                    except Exception as e:
                        last_err = e
                        time.sleep(0.5)
                if downloaded_path is None:
                    raise last_err
                downloaded_path = os.path.abspath(downloaded_path)
                _cleanup_hf_download_metadata(
                    local_path, [local_data_dir, download_cache_dir]
                )
            elif ms_repo:
                from modelscope.hub.file_download import dataset_file_download
                last_err = None
                downloaded_path = None
                for _ in range(3):
                    try:
                        downloaded_path = dataset_file_download(
                            ms_repo, cand_name, cache_dir=download_cache_dir
                        )
                        break
                    except FileNotFoundError as e:
                        last_err = e
                        time.sleep(0.5)
                if downloaded_path is None:
                    raise last_err
                downloaded_path = os.path.abspath(downloaded_path)
            else:
                downloaded_path = os.path.join(local_data_dir, cand_name)
                if not os.path.exists(downloaded_path):
                    raise FileNotFoundError(f"本地目标块缺失：{downloaded_path}")

            if downloaded_path != os.path.abspath(local_path):
                is_link = os.path.islink(downloaded_path)
                real_src = os.path.realpath(downloaded_path) if is_link else None
                if is_link:
                    try:
                        os.remove(downloaded_path)
                    except OSError:
                        pass
                    if not real_src or not os.path.exists(real_src):
                        raise FileNotFoundError(
                            f"Broken HF cache symlink {downloaded_path} -> {real_src}"
                        )
                    try:
                        os.replace(real_src, local_path)
                    except OSError:
                        shutil.copy2(real_src, local_path)
                        try:
                            os.remove(real_src)
                        except OSError:
                            pass
                else:
                    try:
                        os.replace(downloaded_path, local_path)
                    except OSError:
                        shutil.copy2(downloaded_path, local_path)
                        try:
                            os.remove(downloaded_path)
                        except OSError:
                            pass
                _cleanup_hf_download_metadata(
                    downloaded_path, [local_data_dir, download_cache_dir]
                )
            return local_path

    loader._resolve_npz_path = _resolve_npz_path
    return loader


def expected_checkpoint_dir(out_dir: str, prefix: str, step: int, is_emergency: bool = False):
    base_name = f"step_{step}_emergency" if is_emergency else f"step_{step}"
    folder_name = f"{prefix}_{base_name}" if prefix else base_name
    return os.path.join(out_dir, folder_name)


def save_char_subspace_stats(
    student: TinyCharEncoderCUDA,
    out_dir: str,
    prefix: str,
    step: int,
    is_emergency: bool = False,
):
    """
    Saves `char_subspace_stats.pt` containing the native orthogonal 32-D basis and
    valid-token mean offset for downstream Phase 4/5 (`HiPPOOracleCUDA` / Translator / Left Brain).
    """
    assert_colab_drive_mounted(out_dir)
    ckpt_dir = expected_checkpoint_dir(out_dir, prefix, step, is_emergency=is_emergency)
    assert_colab_drive_mounted(ckpt_dir)
    os.makedirs(ckpt_dir, exist_ok=True)
    stats_path = os.path.join(ckpt_dir, "char_subspace_stats.pt")
    w_sub = student.char_subspace_proj.weight.detach().cpu()  # (d_sub, d_model)
    payload = {
        "char_subspace_proj": w_sub,
        "subspace_proj": w_sub.t().contiguous(),  # (d_model, d_sub) for HiPPOOracleCUDA
        "char_token_mean": student.char_token_mean.detach().cpu(),
        "d_sub": student.d_sub,
        "d_model": student.d_model,
        "pad_token_id": student.pad_token_id,
        "global_step": int(step),
    }
    torch.save(payload, stats_path)
    print(f"[Checkpoint] Saved char_subspace_stats.pt to {stats_path}")
    return stats_path


def save_torch_optimizer_state(optimizer, out_dir: str, prefix: str, step: int, is_emergency: bool = False):
    assert_colab_drive_mounted(out_dir)
    ckpt_dir = expected_checkpoint_dir(out_dir, prefix, step, is_emergency=is_emergency)
    assert_colab_drive_mounted(ckpt_dir)
    os.makedirs(ckpt_dir, exist_ok=True)
    opt_path = os.path.join(ckpt_dir, "optimizer.pt")
    torch.save(optimizer.state_dict(), opt_path)
    print(f"[Checkpoint] Saved optimizer state to {opt_path}")


def get_distill_parser():
    parser = argparse.ArgumentParser(
        description="Phase 1 v2 Dual-Head Masked Character Encoder Distillation (Zero WeakDecoder CE)"
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=5000)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--save_steps", type=int, default=10000)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--out_dir", type=str, default="checkpoints/distilled_v2")
    parser.add_argument("--ckpt_prefix", type=str, default="charencoder_pt_v2")
    parser.add_argument("--p0_ckpt", type=str, required=True, help="Frozen Phase 0 checkpoint to distill from")
    parser.add_argument("--config_file", type=str, default=None, help="Optional dataset config JSON")
    parser.add_argument(
        "--hf_repo_id",
        type=str,
        default=None,
        help="Optional Hugging Face dataset repo ID overriding dataset_config.json (e.g. catclever/emb_npy)",
    )
    parser.add_argument(
        "--ms_repo_id",
        type=str,
        default=None,
        help="Optional ModelScope dataset repo ID override",
    )
    parser.add_argument(
        "--hf_ckpt_repo",
        type=str,
        default=None,
        help="Optional Hugging Face model repo ID to mirror saved checkpoints (e.g. user/hollyshit-checkpoints)",
    )
    parser.add_argument("--emb_paths", nargs="*", default=None, help="Explicit embedding .npy paths")
    parser.add_argument("--emb_dir", type=str, default="data/Basic_ZH/embs/hy-tmp")
    parser.add_argument("--parquet_path", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default="./embs")
    parser.add_argument("--download_cache_dir", type=str, default=None)
    parser.add_argument("--chunk_size", type=int, default=500000)
    parser.add_argument("--strict_route_check", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prefetch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--download_only_prefetch", action="store_true")
    parser.add_argument("--student_d_model", type=int, default=1024)
    parser.add_argument("--student_n_heads", type=int, default=8)
    parser.add_argument("--student_n_layers", type=int, default=6)
    parser.add_argument("--student_d_sub", type=int, default=32)
    parser.add_argument("--order_margin", type=float, default=0.35)
    parser.add_argument("--w_distill", type=float, default=2.0)
    parser.add_argument("--w_rel", type=float, default=2.0)
    parser.add_argument("--w_order", type=float, default=2.0)
    parser.add_argument("--w_next", type=float, default=0.5)
    parser.add_argument("--w_char_iso", type=float, default=2.0)
    parser.add_argument("--w_char_id", type=float, default=1.0)
    parser.add_argument("--ema_momentum", type=float, default=0.05)
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="auto",
        choices=["auto", "bf16", "fp16", "fp32"],
        help="CUDA mixed precision mode: 'auto' uses bf16 on Ampere+ (L4/A100) and fp16 on T4",
    )
    return parser


def main():
    parser = get_distill_parser()
    args = parser.parse_args()

    # Hard Drive mount guard at second 0 (before Checkpointer / dataloaders / models!)
    assert_colab_drive_mounted(args.out_dir)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("\n🔥 [Hardware] CUDA cluster detected. Booting NVIDIA PyTorch Amp Backend...")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("\n🍏 [Hardware] Apple Silicon detected. Booting PyTorch MPS Accelerated Backend...")
    else:
        device = torch.device("cpu")
        print("\n🐢 [Hardware] CPU only detected.")

    ds_config = load_optional_config(args.config_file)
    if args.download_cache_dir is None:
        args.download_cache_dir = args.data_dir

    config = WeakDecoderConfig()
    config.max_seq_len = args.max_seq_len

    tokenizer = CharTokenizer()
    pad_token_id = getattr(tokenizer, "pad_token_id", 0)

    fuser_path, god_path, emb_dims, d_model, z_dim = infer_teacher_arch_from_ckpt(args.p0_ckpt)
    print(
        f"\n[Teacher Arch] routes={len(emb_dims)} emb_dims={emb_dims} "
        f"d_model={d_model} z_dim={z_dim}"
    )

    dataloader = build_dataloader(args, tokenizer, ds_config, expected_routes=len(emb_dims))
    dataloader = attach_chunk_download_router(dataloader, args.data_dir, args.download_cache_dir)

    # 1. Instantiate the Frozen Teacher Replicas (SensoryFuser + GodEncoder; NO WeakDecoder!)
    fuser = SensoryFuserCUDA(emb_dims, d_model).to(device)
    god_encoder = GodEncoderCUDA(d_model, z_dim).to(device)

    print(f"\n[Teacher] Loading Frozen Phase 0 Teacher from {args.p0_ckpt} (Zero WeakDecoder CE)...")
    load_mlx_safetensors_into_torch(fuser, fuser_path)
    load_mlx_safetensors_into_torch(god_encoder, god_path)
    fuser.eval()
    god_encoder.eval()
    for p in fuser.parameters():
        p.requires_grad = False
    for p in god_encoder.parameters():
        p.requires_grad = False

    # 2. Instantiate Phase 1 v2 Dual-Head Student Encoder + Auxiliary Heads
    print("[Student] Initializing Phase 1 v2 Dual-Head Masked TinyCharEncoderCUDA...")
    student = TinyCharEncoderCUDA(
        vocab_size=config.vocab_size,
        d_model=args.student_d_model,
        n_heads=args.student_n_heads,
        n_layers=args.student_n_layers,
        max_seq_len=config.max_seq_len,
        z_dim=z_dim,
        d_sub=args.student_d_sub,
        pad_token_id=pad_token_id,
    ).to(device)

    order_head = OrderDirectionHeadCUDA(z_dim=z_dim).to(device)
    next_clause_head = AsymmetricNextClauseHeadCUDA(z_dim=z_dim).to(device)
    char_anchor_head = CharSubspaceAnchorHeadCUDA(
        d_sub=args.student_d_sub, vocab_size=config.vocab_size
    ).to(device)

    trainable_params = (
        list(student.parameters())
        + list(order_head.parameters())
        + list(next_clause_head.parameters())
        + list(char_anchor_head.parameters())
    )

    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

    if device.type == "cuda" and args.mixed_precision != "fp32":
        bf16_ok = bool(hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported())
        if args.mixed_precision == "bf16" or (args.mixed_precision == "auto" and bf16_ok):
            use_amp = True
            amp_dtype = torch.bfloat16
            use_scaler = False
            print("⚡ [AMP] Using CUDA bfloat16 mixed precision (overflow-free, GradScaler disabled).")
        else:
            use_amp = True
            amp_dtype = torch.float16
            use_scaler = True
            print("⚡ [AMP] Using CUDA float16 mixed precision (GradScaler enabled).")
    else:
        use_amp = False
        amp_dtype = torch.float32
        use_scaler = False
        print(f"⚡ [AMP] Using full float32 precision on {device.type}.")

    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    checkpointer = Checkpointer(args.out_dir, prefix=args.ckpt_prefix)
    checkpointer.register_model("student", student)
    checkpointer.register_model("order_head", order_head)
    checkpointer.register_model("next_clause_head", next_clause_head)
    checkpointer.register_model("char_anchor_head", char_anchor_head)
    checkpointer.register_dataloader("dataloader", dataloader)
    checkpointer.register_args(args)

    global_step = 0
    print(f"\n========================================================")
    print(f"🚀 Phase 1 v2 Dual-Head Masked Engine Ignited! Target Epochs: {args.epochs}")
    print(f"⚙️  Batch: {args.batch_size} | d_sub: {args.student_d_sub} | Peak LR: {args.lr}")
    print(f"========================================================\n")

    try:
        for epoch in range(dataloader.current_epoch, args.epochs):
            for token_inputs, batch_embs, attention_mask in dataloader:
                global_step += 1
                checkpointer.set_step(global_step)
                if len(batch_embs) != len(emb_dims):
                    if args.strict_route_check or len(batch_embs) < len(emb_dims):
                        raise ValueError(
                            f"Runtime embedding route mismatch: got {len(batch_embs)}, expected {len(emb_dims)}"
                        )
                    batch_embs = batch_embs[: len(emb_dims)]

                lr = custom_lr_schedule(global_step, args.lr, args.warmup_steps)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

                token_inputs = token_inputs.to(device, non_blocking=True)
                attention_mask = (
                    attention_mask.to(device, non_blocking=True)
                    if attention_mask is not None
                    else None
                )
                batch_embs = [emb.to(device, non_blocking=True) for emb in batch_embs]

                optimizer.zero_grad(set_to_none=True)

                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                    with torch.no_grad():
                        f_t = fuser(batch_embs, weights=None)
                        z_target_truth = god_encoder(f_t)
                        if z_target_truth.ndim == 3:
                            z_target_truth = z_target_truth.squeeze(1)

                    loss, metrics = compute_phase1_v2_distill_loss(
                        student=student,
                        token_inputs=token_inputs,
                        attention_mask=attention_mask,
                        z_target_truth=z_target_truth,
                        batch_embs=batch_embs,
                        order_head=order_head,
                        next_clause_head=next_clause_head,
                        char_anchor_head=char_anchor_head,
                        order_margin=args.order_margin,
                        w_distill=args.w_distill,
                        w_rel=args.w_rel,
                        w_order=args.w_order,
                        w_next=args.w_next,
                        w_char_iso=args.w_char_iso,
                        w_char_id=args.w_char_id,
                        update_ema=True,
                        ema_momentum=args.ema_momentum,
                    )

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                if global_step % 10 == 0:
                    print(
                        f"Epoch {epoch+1} | Step {global_step} | "
                        f"Total: {loss.item():.4f} | "
                        f"Distill: {metrics['loss_distill'].item():.4f} (cos={metrics['z_god_cos'].item():.3f}) | "
                        f"Rel: {metrics['loss_rel'].item():.4f} | "
                        f"Order: {metrics['loss_order'].item():.4f} (shuf_cos={metrics['shuffled_order_cos'].item():.3f}, "
                        f"acc={metrics['order_discrim_acc'].item():.3f}) | "
                        f"CharIso: {metrics['loss_char_iso'].item():.4f} (PC1={metrics['subspace_pc1_var_ratio'].item():.3f}, "
                        f"PR={metrics['subspace_participation_ratio'].item():.1f}) | "
                        f"CharID: {metrics['loss_char_id'].item():.4f} (acc={metrics['subspace_char_top1_acc'].item():.3f})"
                    )

                if args.eval_steps > 0 and global_step % args.eval_steps == 0:
                    health = evaluate_phase1_v2_health(
                        student=student,
                        token_inputs=token_inputs,
                        attention_mask=attention_mask,
                        z_target_truth=z_target_truth,
                        batch_embs=batch_embs,
                        order_head=order_head,
                        char_anchor_head=char_anchor_head,
                    )
                    print(
                        f"[Health @ Step {global_step}] "
                        f"z_god_cos={health['z_god_cos']:.4f} | "
                        f"rel_spearman={health['relational_spearman']:.4f} | "
                        f"shuf_cos={health['shuffled_order_cos']:.4f} | "
                        f"order_acc={health['order_discrim_acc']:.4f} | "
                        f"PC1={health['subspace_pc1_var_ratio']:.4f} | "
                        f"PR={health['subspace_participation_ratio']:.2f}/32 | "
                        f"char_acc={health['subspace_char_top1_acc']:.4f} | "
                        f"mask_diff={health['masked_vs_auto_mask_max_diff']:.2e}"
                    )

                if global_step % args.save_steps == 0:
                    assert_colab_drive_mounted(args.out_dir)
                    checkpointer.save(global_step)
                    save_torch_optimizer_state(optimizer, args.out_dir, checkpointer.prefix, global_step)
                    save_char_subspace_stats(student, args.out_dir, checkpointer.prefix, global_step)
                    ckpt_dir = expected_checkpoint_dir(args.out_dir, checkpointer.prefix, global_step)
                    if args.hf_ckpt_repo:
                        mirror_checkpoint_to_hf_hub(ckpt_dir, args.hf_ckpt_repo)

        if global_step > 0 and global_step % args.save_steps != 0:
            assert_colab_drive_mounted(args.out_dir)
            checkpointer.save(global_step)
            save_torch_optimizer_state(optimizer, args.out_dir, checkpointer.prefix, global_step)
            save_char_subspace_stats(student, args.out_dir, checkpointer.prefix, global_step)
            final_dir = expected_checkpoint_dir(args.out_dir, checkpointer.prefix, global_step)
            if args.hf_ckpt_repo:
                mirror_checkpoint_to_hf_hub(final_dir, args.hf_ckpt_repo)

    except KeyboardInterrupt:
        print(f"\n[Interrupt] Caught kill signal at step {global_step}! Saving emergency checkpoint...")
        assert_colab_drive_mounted(args.out_dir)
        save_torch_optimizer_state(optimizer, args.out_dir, checkpointer.prefix, global_step, is_emergency=True)
        save_char_subspace_stats(student, args.out_dir, checkpointer.prefix, global_step, is_emergency=True)
        emerg_dir = expected_checkpoint_dir(
            args.out_dir, checkpointer.prefix, global_step, is_emergency=True
        )
        try:
            checkpointer.save(global_step, is_emergency=True)
        except SystemExit:
            if args.hf_ckpt_repo:
                mirror_checkpoint_to_hf_hub(emerg_dir, args.hf_ckpt_repo)
            raise
        if args.hf_ckpt_repo:
            mirror_checkpoint_to_hf_hub(emerg_dir, args.hf_ckpt_repo)


if __name__ == "__main__":
    main()
