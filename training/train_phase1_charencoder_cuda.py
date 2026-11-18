import argparse
import glob
import json
import math
import os

# Optimization (Task 071):
# 1) Problem: PyTorch's default CUDA caching allocator fragments large variable-size buffers
#    across batches and chunked vocab cross-entropy steps, risking CUDA OOM on L4/A100.
# 2) Method: Default `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` immediately after
#    `import os` and prior to `import torch` via `os.environ.setdefault`.
# 3) Caveats: Must be set before `import torch` initializes the CUDA caching allocator.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import re
import shutil
import sys
import tempfile
import threading
import time
import zipfile
import zlib
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from safetensors.torch import load_file as load_safetensors_file
from safetensors.torch import save_file as save_safetensors_file

# Auto-resolve the parent workspace root and force it to the FRONT of the import path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir in sys.path and os.path.exists(os.path.join(script_dir, "model.py")):
    sys.path = [p for p in sys.path if p != script_dir] + [script_dir]

from training.data.alignment_dataloader import (
    Phase1_ChunkedNpzDataLoader,
    Phase1_MultiEmbDataLoader,
    is_valid_npz_archive,
    repair_gdrive_chunk_from_remote,
    resolve_gdrive_chunk_file,
    validate_npz_archive,
)

from training.core.char_tokenizer import CharTokenizer
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


def cosine_warmup_lr_schedule(
    step: int,
    max_lr: float,
    warmup_steps: int,
    total_steps: int = 100000,
    min_lr_ratio: float = 0.1,
) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return max_lr * (float(step) / float(warmup_steps))
    if total_steps <= warmup_steps:
        return max_lr
    progress = min(max(float(step - warmup_steps) / float(total_steps - warmup_steps), 0.0), 1.0)
    cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
    return max_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine_factor)


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
    return load_safetensors_file(path)


def _has_valid_p0_teacher_files(ckpt_dir: str) -> bool:
    if not ckpt_dir or not os.path.isdir(ckpt_dir):
        return False
    fuser_names = (
        "sense_fuser.safetensors",
        "sense_adapter.safetensors",
        "sense_fuser.pt",
        "sense_adapter.pt",
    )
    god_names = ("god_encoder.safetensors", "god_encoder.pt")
    has_fuser = any(
        os.path.isfile(os.path.join(ckpt_dir, fn))
        and os.path.getsize(os.path.join(ckpt_dir, fn)) > 0
        for fn in fuser_names
    )
    has_god = any(
        os.path.isfile(os.path.join(ckpt_dir, fn))
        and os.path.getsize(os.path.join(ckpt_dir, fn)) > 0
        for fn in god_names
    )
    return bool(has_fuser and has_god)


def _extract_p0_ckpt_step(ckpt_dir: str) -> int:
    ms_path = os.path.join(ckpt_dir, "manifold_stats.pt")
    if os.path.isfile(ms_path) and os.path.getsize(ms_path) > 0:
        try:
            ms_obj = torch.load(ms_path, map_location="cpu", weights_only=False)
            if isinstance(ms_obj, dict) and ms_obj.get("global_step") is not None:
                return int(ms_obj["global_step"])
        except Exception:
            pass

    dl_path = os.path.join(ckpt_dir, "dataloader.json")
    if os.path.isfile(dl_path) and os.path.getsize(dl_path) > 0:
        try:
            with open(dl_path, "r", encoding="utf-8") as f:
                dl_obj = json.load(f)
            if isinstance(dl_obj, dict) and dl_obj.get("global_step") is not None:
                return int(dl_obj["global_step"])
        except Exception:
            pass

    m = re.search(r"(?:^|_)step_(\d+)$", os.path.basename(os.path.normpath(ckpt_dir)))
    if m:
        return int(m.group(1))
    return 0


# Bug Fix (Task 070):
# 1) Problem: Users may pass `--p0_ckpt /content/drive/MyDrive/hollyshit/checkpoints/p0_v3`
#    (the parent directory) or run Phase 1 v2 before Cell 7 exported `p0_v3_5models_best` (when
#    only `p0_v3_5models_step_<N>` / `p0_v3_5models_latest_emergency` exist), causing an immediate
#    `FileNotFoundError` in `infer_teacher_arch_from_ckpt`.
# 2) Method: Auto-discover direct teacher checkpoints, `p0_v3_5models_best` / `*_best` aliases, or
#    the highest-step valid child checkpoint directory (preferring regular `step_*` over emergency
#    on ties), and also read `sphere_scale` from `manifold_stats.pt` when initializing `GodEncoderCUDA`.
# 3) Caveats: Falls back to returning `p0_ckpt` unchanged if nothing valid is found so
#    `infer_teacher_arch_from_ckpt` still raises a clear `FileNotFoundError`.
def resolve_p0_teacher_ckpt_dir(p0_ckpt: str) -> str:
    if not p0_ckpt:
        return p0_ckpt
    if _has_valid_p0_teacher_files(p0_ckpt):
        return p0_ckpt
    if not os.path.isdir(p0_ckpt):
        return p0_ckpt

    preferred_best = os.path.join(p0_ckpt, "p0_v3_5models_best")
    if _has_valid_p0_teacher_files(preferred_best):
        print(f"[Teacher] Auto-resolved Phase 0 teacher checkpoint: {preferred_best}")
        return preferred_best

    best_children: List[Tuple[int, str]] = []
    step_children: List[Tuple[int, int, str]] = []
    for entry in os.listdir(p0_ckpt):
        child = os.path.join(p0_ckpt, entry)
        if not _has_valid_p0_teacher_files(child):
            continue
        step_val = _extract_p0_ckpt_step(child)
        if entry.endswith("_best") or entry == "best":
            best_children.append((step_val, child))
        else:
            is_emerg = entry.endswith("latest_emergency") or entry.endswith("_emergency")
            non_emerg_rank = 0 if is_emerg else 1
            step_children.append((step_val, non_emerg_rank, child))

    if best_children:
        best_children.sort(key=lambda item: (item[0], item[1]))
        chosen = best_children[-1][1]
        print(f"[Teacher] Auto-resolved Phase 0 teacher checkpoint: {chosen}")
        return chosen

    if step_children:
        step_children.sort(key=lambda item: (item[0], item[1], item[2]))
        chosen = step_children[-1][2]
        print(f"[Teacher] Auto-resolved Phase 0 teacher checkpoint: {chosen}")
        return chosen

    return p0_ckpt


def _infer_teacher_sphere_scale(p0_ckpt: str) -> bool:
    ms_path = os.path.join(p0_ckpt, "manifold_stats.pt")
    if os.path.isfile(ms_path) and os.path.getsize(ms_path) > 0:
        try:
            ms_obj = torch.load(ms_path, map_location="cpu", weights_only=False)
            if isinstance(ms_obj, dict) and "sphere_scale" in ms_obj:
                return bool(ms_obj["sphere_scale"])
        except Exception:
            pass
    return False


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


# Bug Fix (Task 070):
# 1) Problem: Unlike Phase 0 v3 (`embedding_only=True`), Phase 1 v2 requires the aligned clause
#    text Parquet (`chunked_mixed_omni.parquet`), which may reside under `/content/drive/MyDrive/hollyshit/data`
#    or a remote dataset repo rather than the container working directory.
# 2) Method: Check direct `parquet_path` first, search candidate Drive/local data directories for
#    `os.path.basename(parquet_path)`, and if still missing, attempt remote fetch from ModelScope
#    (`catclever/emb_npy`) -> Hugging Face (`drunkencat/emb_npy`) with `assert_colab_drive_mounted`
#    and atomic `.tmp` + `os.replace` write-back.
# 3) Caveats: Never swallows `[Colab Mount Guard]` `RuntimeError`; on benign offline/missing-remote
#    errors, logs a concise warning and returns `parquet_path` for `Phase1_ChunkedNpzDataLoader` validation.
def resolve_or_fetch_parquet_path(parquet_path: str, args, ds_config: Optional[dict]) -> str:
    ds_config = ds_config or {}
    if parquet_path and os.path.isfile(parquet_path) and os.path.getsize(parquet_path) > 0:
        return parquet_path

    cfg_rel = ds_config.get("parquet_path", "data/Basic_ZH/chunked_mixed_omni.parquet")
    base_name = os.path.basename(parquet_path or cfg_rel or "chunked_mixed_omni.parquet")

    search_roots: List[str] = ["/content/drive/MyDrive/hollyshit/data"]
    gdrive_dir = getattr(args, "gdrive_dir", None) if args is not None else None
    if gdrive_dir:
        search_roots.append(gdrive_dir)
        gd_parent = os.path.dirname(os.path.abspath(gdrive_dir.rstrip("/")))
        if gd_parent:
            search_roots.append(os.path.join(gd_parent, "data"))
            search_roots.append(gd_parent)
    data_dir = getattr(args, "data_dir", None) if args is not None else None
    if data_dir:
        search_roots.append(data_dir)
    search_roots.append(os.path.abspath("data"))

    seen_roots = set()
    for root in search_roots:
        if not root:
            continue
        abs_r = os.path.abspath(root)
        if abs_r in seen_roots or not os.path.isdir(abs_r):
            continue
        seen_roots.add(abs_r)

        direct_cands = [
            os.path.join(abs_r, base_name),
            os.path.join(abs_r, "Basic_ZH", base_name),
            os.path.join(abs_r, "data", "Basic_ZH", base_name),
        ]
        for dc in direct_cands:
            if os.path.isfile(dc) and os.path.getsize(dc) > 0:
                print(f"[Data] Resolved existing Parquet corpus at: {dc}")
                return dc

        for hit in glob.glob(os.path.join(abs_r, "**", base_name), recursive=True):
            if os.path.isfile(hit) and os.path.getsize(hit) > 0:
                print(f"[Data] Resolved existing Parquet corpus at: {hit}")
                return os.path.abspath(hit)

    eff_ms_repo = (
        (getattr(args, "ms_repo_id", None) if args is not None else None)
        or ds_config.get("modelscope_repo_id")
        or "catclever/emb_npy"
    ).strip()
    eff_hf_repo = (
        (getattr(args, "hf_repo_id", None) if args is not None else None)
        or ds_config.get("huggingface_repo_id")
        or "drunkencat/emb_npy"
    ).strip()

    remote_candidates: List[str] = []
    for rc in (
        cfg_rel.replace("\\", "/").lstrip("/"),
        re.sub(r"^data/", "", cfg_rel.replace("\\", "/").lstrip("/")),
        f"Basic_ZH/{base_name}",
        f"data/Basic_ZH/{base_name}",
        base_name,
    ):
        if rc and rc not in remote_candidates:
            remote_candidates.append(rc)

    stage_base = (
        (getattr(args, "download_cache_dir", None) if args is not None else None)
        or (getattr(args, "data_dir", None) if args is not None else None)
        or tempfile.gettempdir()
    )
    work_dir = os.path.join(
        os.path.abspath(stage_base),
        f"_parquet_fetch_{os.getpid()}_{threading.get_ident()}",
    )

    dl_path: Optional[str] = None
    try:
        os.makedirs(work_dir, exist_ok=True)
        # 1. Probe ModelScope first
        try:
            from modelscope.hub.api import HubApi
            from modelscope.hub.file_download import dataset_file_download

            ms_cands = list(remote_candidates)
            try:
                ms_api = HubApi()
                ms_tok = (
                    os.environ.get("MODELSCOPE_API_TOKEN")
                    or os.environ.get("MS_TOKEN")
                    or ""
                ).strip()
                if ms_tok:
                    try:
                        ms_api.login(ms_tok)
                    except Exception:
                        pass
                repo_items = ms_api.get_dataset_files(eff_ms_repo, recursive=True)
                repo_paths = [
                    it.get("Path", it.get("Name", "")).replace("\\", "/").lstrip("/")
                    for it in repo_items
                ]
                matched_ms = [
                    rp for rp in repo_paths if os.path.basename(rp) == base_name or rp in ms_cands
                ]
                ms_cands = matched_ms if matched_ms else []
            except Exception:
                pass

            for rc in ms_cands:
                try:
                    cand_dl = dataset_file_download(eff_ms_repo, rc, cache_dir=work_dir)
                    if cand_dl and os.path.isfile(cand_dl) and os.path.getsize(cand_dl) > 0:
                        dl_path = cand_dl
                        break
                except Exception:
                    continue
        except Exception:
            pass

        # 2. Fallback to Hugging Face Hub
        if dl_path is None:
            try:
                from huggingface_hub import HfApi

                hf_cands = list(remote_candidates)
                try:
                    hf_files = [
                        rf.replace("\\", "/").lstrip("/")
                        for rf in HfApi(token=_get_hf_token()).list_repo_files(
                            repo_id=eff_hf_repo, repo_type="dataset"
                        )
                    ]
                    matched_hf = [
                        rf for rf in hf_files if os.path.basename(rf) == base_name or rf in hf_cands
                    ]
                    hf_cands = matched_hf if matched_hf else []
                except Exception:
                    pass

                for rc in hf_cands:
                    try:
                        cand_dl = _download_hf_dataset_chunk_file(
                            eff_hf_repo, rc, local_dir=work_dir, cache_dir=work_dir
                        )
                        if cand_dl and os.path.isfile(cand_dl) and os.path.getsize(cand_dl) > 0:
                            dl_path = cand_dl
                            break
                    except Exception:
                        continue
            except Exception:
                pass

        if dl_path is not None and parquet_path:
            real_dl = os.path.realpath(dl_path) if os.path.islink(dl_path) else os.path.abspath(dl_path)
            assert_colab_drive_mounted(parquet_path)
            dest_abs = os.path.abspath(parquet_path)
            os.makedirs(os.path.dirname(dest_abs), exist_ok=True)
            tmp_dest = f"{dest_abs}.fetch_{os.getpid()}_{threading.get_ident()}.tmp"
            try:
                shutil.copy2(real_dl, tmp_dest)
                if os.path.getsize(tmp_dest) <= 0:
                    raise ValueError(f"Fetched empty Parquet file: {tmp_dest}")
                os.replace(tmp_dest, dest_abs)
            finally:
                if os.path.exists(tmp_dest):
                    try:
                        os.remove(tmp_dest)
                    except OSError:
                        pass
            print(f"[Data] ✅ Auto-fetched Parquet corpus to: {dest_abs}")
            return dest_abs
    except RuntimeError as rt_err:
        if "[Colab Mount Guard]" in str(rt_err):
            raise
        print(f"[Data] ⚠️ Remote Parquet probe skipped ({rt_err}); keeping path: {parquet_path}")
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    print(
        f"[Data] ⚠️ Parquet corpus not found locally or in remote dataset repos ({base_name}); "
        f"keeping target path: {parquet_path}"
    )
    return parquet_path


def build_dataloader(args, tokenizer, ds_config: dict, expected_routes: int):
    ds_config = ds_config or {}
    has_chunked_conf = bool(ds_config.get("base_models") and ds_config.get("chunk_name_patterns"))
    config_base_dir = os.path.dirname(os.path.abspath(args.config_file)) if args.config_file else os.getcwd()
    parquet_path = args.parquet_path or ds_config.get("parquet_path", "data/Basic_ZH/chunked_mixed_omni.parquet")
    parquet_path = resolve_path_maybe_relative(parquet_path, config_base_dir)
    parquet_path = resolve_or_fetch_parquet_path(parquet_path, args, ds_config)

    raw_data_source = getattr(args, "data_source", None) or ds_config.get("default_source", "ms")
    ds_lower = (raw_data_source or "ms").strip().lower()
    if ds_lower in ("ms", "modelscope"):
        data_source = "ms"
    elif ds_lower in ("gdrive", "google_drive", "googledrive"):
        data_source = "gdrive"
    elif ds_lower in ("hf", "huggingface", "hugging_face"):
        data_source = "hf"
    elif ds_lower in ("local", "dir", "local_dir"):
        data_source = "local"
    else:
        data_source = "ms"

    gdrive_dir = getattr(args, "gdrive_dir", None) or ds_config.get(
        "gdrive_dir", "/content/drive/MyDrive/hollyshit/embs"
    )

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

        if data_source == "ms":
            src_desc = f"ms_repo_id={ms_repo_id or 'catclever/emb_npy'}"
        elif data_source == "gdrive":
            src_desc = f"gdrive_dir={gdrive_dir} (staged to {args.download_cache_dir or args.data_dir})"
        elif data_source == "hf":
            src_desc = f"hf_repo_id={hf_repo_id or 'drunkencat/emb_npy'}"
        else:
            src_desc = f"local_npz_dir={args.data_dir}"

        print(f"[Data] Using Phase1_ChunkedNpzDataLoader ({len(models)} routes, source={data_source}, {src_desc}).")
        return Phase1_ChunkedNpzDataLoader(
            parquet_path=parquet_path,
            models=models,
            chunk_patterns=chunk_patterns,
            tokenizer=tokenizer,
            ms_repo_id=ms_repo_id,
            chunk_size=args.chunk_size,
            local_npz_dir=args.data_dir,
            cache_dir=args.download_cache_dir,
            batch_size=args.batch_size,
            max_seq_len=args.max_seq_len,
            shuffle=True,
            lazy_start=True,
            backend="torch",
            hf_repo_id=hf_repo_id,
            data_source=data_source,
            gdrive_dir=gdrive_dir,
            pair_shuffle=True,
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

        gdrive_dir_chk = getattr(loader, "gdrive_dir", None)
        gd_abs_chk = os.path.abspath(gdrive_dir_chk) if gdrive_dir_chk else None

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
            abs_r = os.path.abspath(root)
            if gd_abs_chk and (abs_r == gd_abs_chk or abs_r.startswith(gd_abs_chk + os.sep)):
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
                        or ".repair_" in fn
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
            data_source = getattr(loader, "data_source", "ms")
            hf_repo = getattr(loader, "hf_repo_id", None)
            ms_repo = getattr(loader, "ms_repo_id", None)
            gdrive_dir = getattr(loader, "gdrive_dir", None)

            # Bug Fix (Task 069):
            # 1) Problem: Phase 1 router previously skipped `.npz` header/EOF validation on cached
            #    `local_path` and `gdrive_src`, failing when Google Drive had truncated/missing chunks.
            # 2) Method: Validate cached NVMe hits via `is_valid_npz_archive`, validate `gdrive_src`
            #    via `validate_npz_archive`, and invoke `repair_gdrive_chunk_from_remote` to heal
            #    corrupt/missing Google Drive chunks online from ModelScope / Hugging Face.
            # 3) Caveats: Healthy persistent files in `gdrive_dir` and `local` mode are never modified.
            if os.path.exists(local_path):
                if data_source == "local" or is_valid_npz_archive(local_path):
                    return local_path
                is_inside_gdrive = False
                if gdrive_dir:
                    gd_abs = os.path.abspath(gdrive_dir)
                    lp_abs = os.path.abspath(local_path)
                    lp_real = os.path.realpath(local_path)
                    if (
                        lp_abs == gd_abs
                        or lp_abs.startswith(gd_abs + os.sep)
                        or lp_real == gd_abs
                        or lp_real.startswith(gd_abs + os.sep)
                    ):
                        is_inside_gdrive = True
                if not is_inside_gdrive:
                    try:
                        os.remove(local_path)
                    except OSError:
                        pass

            if data_source == "gdrive":
                if not gdrive_dir:
                    raise ValueError("[attach_chunk_download_router] loader.data_source 为 'gdrive' 但未配置 gdrive_dir")
                file_map = getattr(loader, "_gdrive_file_map", None)
                if file_map is None:
                    file_map = {}
                    loader._gdrive_file_map = file_map
                broken_src: Optional[str] = None
                try:
                    gdrive_src = resolve_gdrive_chunk_file(
                        gdrive_dir,
                        cand_name,
                        file_map=file_map,
                        source_label="Google Drive",
                    )
                    broken_src = gdrive_src
                    validate_npz_archive(gdrive_src)
                except (
                    FileNotFoundError,
                    zipfile.BadZipFile,
                    zlib.error,
                    OSError,
                    ValueError,
                    KeyError,
                    EOFError,
                ):
                    if getattr(loader, "allow_gdrive_remote_repair", True):
                        eff_ms = (
                            ms_repo
                            or getattr(loader, "_fallback_ms_repo_id", None)
                            or "catclever/emb_npy"
                        )
                        eff_hf = (
                            hf_repo
                            or getattr(loader, "_fallback_hf_repo_id", None)
                            or "drunkencat/emb_npy"
                        )
                        return repair_gdrive_chunk_from_remote(
                            gdrive_dir,
                            cand_name,
                            cache_dir=local_data_dir,
                            ms_repo_id=eff_ms,
                            hf_repo_id=eff_hf,
                            file_map=file_map,
                            broken_gdrive_path=broken_src,
                        )
                    raise

                if os.path.abspath(gdrive_src) == os.path.abspath(local_path):
                    return local_path
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                shutil.copy2(gdrive_src, local_path)
                validate_npz_archive(local_path)
                downloaded_path = local_path
            elif data_source == "hf" or (data_source is None and hf_repo):
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                last_err = None
                downloaded_path = None
                for _ in range(3):
                    try:
                        downloaded_path = _download_hf_dataset_chunk_file(
                            hf_repo or "drunkencat/emb_npy",
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
            elif data_source == "ms" or (data_source is None and ms_repo):
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                from modelscope.hub.file_download import dataset_file_download
                last_err = None
                downloaded_path = None
                for _ in range(3):
                    try:
                        downloaded_path = dataset_file_download(
                            ms_repo or "catclever/emb_npy", cand_name, cache_dir=download_cache_dir
                        )
                        break
                    except FileNotFoundError as e:
                        last_err = e
                        time.sleep(0.5)
                if downloaded_path is None:
                    raise last_err
                downloaded_path = os.path.abspath(downloaded_path)
            else:
                # Bug Fix (Task 068):
                # 1) Problem: Local mode fell through to `os.replace(downloaded_path, local_path)`,
                #    moving persistent flat/nested files and failing on nested ModelScope directories.
                # 2) Method: Maintain `loader._local_file_map` and return `resolve_gdrive_chunk_file(...)` directly.
                # 3) Caveats: Early return prevents `os.replace` or cleanup sweeps from altering persistent local chunks.
                local_map = getattr(loader, "_local_file_map", None)
                if local_map is None:
                    local_map = {}
                    loader._local_file_map = local_map
                return resolve_gdrive_chunk_file(
                    local_data_dir,
                    cand_name,
                    file_map=local_map,
                    source_label="local",
                )

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


# Bug Fix (Task 070):
# 1) Problem: Phase 1 v2 previously lacked `load_npz_features_with_repair` and `disable_chunk_prefetch`,
#    so `--no-prefetch` in Colab still spawned the background prefetch queue (doubling `.npz` RAM on
#    top of the in-memory Parquet text pool and risking OOM) and could not recover from mid-stream
#    CRC corruption during `np.load(...)["features"]`.
# 2) Method: Port `load_npz_features_with_repair`, `disable_chunk_prefetch`, and
#    `enable_download_only_prefetch` from Phase 0 v3, preserving `loader.emb_dims` auto-detection
#    and `pair_shuffle` micro-batch ordering.
# 3) Caveats: Healthy persistent files in `gdrive_dir` and `local` mode are never deleted or modified.
def load_npz_features_with_repair(
    loader,
    path: str,
    cand_name: Optional[str] = None,
    max_retries: int = 2,
) -> np.ndarray:
    attempt = 0
    last_err = None
    while attempt <= max_retries:
        try:
            return np.load(path)["features"]
        except (zipfile.BadZipFile, zlib.error, OSError, ValueError, KeyError, EOFError) as e:
            last_err = e
            attempt += 1
            if attempt > max_retries:
                break
            data_source = getattr(loader, "data_source", "ms")
            gdrive_dir = getattr(loader, "gdrive_dir", None)
            local_npz_dir = getattr(loader, "npz_dir", None)
            abs_p = os.path.abspath(path) if path else ""
            real_p = os.path.realpath(path) if path else ""
            is_protected_persistent = False
            if gdrive_dir:
                gd_abs = os.path.abspath(gdrive_dir)
                if (
                    abs_p == gd_abs
                    or abs_p.startswith(gd_abs + os.sep)
                    or real_p == gd_abs
                    or real_p.startswith(gd_abs + os.sep)
                ):
                    is_protected_persistent = True
            if data_source == "local" and local_npz_dir:
                loc_abs = os.path.abspath(local_npz_dir)
                if (
                    abs_p == loc_abs
                    or abs_p.startswith(loc_abs + os.sep)
                    or real_p == loc_abs
                    or real_p.startswith(loc_abs + os.sep)
                ):
                    is_protected_persistent = True

            if not is_protected_persistent:
                try:
                    if os.path.islink(path):
                        real_link = os.path.realpath(path)
                        os.remove(path)
                        if real_link and os.path.exists(real_link):
                            os.remove(real_link)
                    elif os.path.exists(path):
                        os.remove(path)
                except OSError:
                    pass

            if (
                cand_name
                and data_source == "gdrive"
                and gdrive_dir
                and getattr(loader, "allow_gdrive_remote_repair", True)
                and attempt >= 1
            ):
                file_map = getattr(loader, "_gdrive_file_map", None)
                broken_src: Optional[str] = None
                try:
                    broken_src = resolve_gdrive_chunk_file(
                        gdrive_dir,
                        cand_name,
                        file_map=file_map,
                        source_label="Google Drive",
                    )
                except FileNotFoundError:
                    broken_src = None

                stage_dir = getattr(loader, "npz_dir", None) or getattr(loader, "cache_dir", None)
                eff_ms = (
                    getattr(loader, "ms_repo_id", None)
                    or getattr(loader, "_fallback_ms_repo_id", None)
                    or "catclever/emb_npy"
                )
                eff_hf = (
                    getattr(loader, "hf_repo_id", None)
                    or getattr(loader, "_fallback_hf_repo_id", None)
                    or "drunkencat/emb_npy"
                )
                path = repair_gdrive_chunk_from_remote(
                    gdrive_dir,
                    cand_name,
                    cache_dir=stage_dir,
                    ms_repo_id=eff_ms,
                    hf_repo_id=eff_hf,
                    file_map=file_map,
                    broken_gdrive_path=broken_src,
                )
                if (
                    broken_src
                    and os.path.abspath(path) != os.path.abspath(broken_src)
                    and os.path.isfile(path)
                ):
                    assert_colab_drive_mounted(broken_src)
                    tmp_gd = f"{broken_src}.repair_{os.getpid()}_{threading.get_ident()}.tmp"
                    try:
                        shutil.copy2(path, tmp_gd)
                        validate_npz_archive(tmp_gd)
                        os.replace(tmp_gd, broken_src)
                    finally:
                        if os.path.exists(tmp_gd):
                            try:
                                os.remove(tmp_gd)
                            except OSError:
                                pass
            elif cand_name and hasattr(loader, "_resolve_npz_path"):
                path = loader._resolve_npz_path(cand_name)
            elif cand_name and data_source == "gdrive":
                if gdrive_dir:
                    src = resolve_gdrive_chunk_file(
                        gdrive_dir,
                        cand_name,
                        file_map=getattr(loader, "_gdrive_file_map", None),
                        source_label="Google Drive",
                    )
                    cache_dir = getattr(loader, "cache_dir", None) or "./datas"
                    staged = os.path.join(cache_dir, cand_name)
                    if os.path.abspath(src) != os.path.abspath(staged):
                        os.makedirs(os.path.dirname(staged), exist_ok=True)
                        shutil.copy2(src, staged)
                        path = staged
                    else:
                        path = src
                else:
                    raise
            elif cand_name and (data_source == "hf" or getattr(loader, "hf_repo_id", None)):
                cache_dir = getattr(loader, "cache_dir", None) or getattr(loader, "npz_dir", "./datas")
                path = _download_hf_dataset_chunk_file(
                    loader.hf_repo_id or "drunkencat/emb_npy",
                    cand_name,
                    local_dir=cache_dir,
                    cache_dir=cache_dir,
                )
                _cleanup_hf_download_metadata(path, [cache_dir, getattr(loader, "npz_dir", None)])
            elif cand_name and (data_source == "ms" or getattr(loader, "ms_repo_id", None)):
                from modelscope.hub.file_download import dataset_file_download

                path = dataset_file_download(
                    loader.ms_repo_id or "catclever/emb_npy", cand_name, cache_dir=loader.cache_dir
                )
            else:
                raise
    raise RuntimeError(f"Failed to load npz after retries: {path} | last_error={last_err}")


def disable_chunk_prefetch(loader):
    if not hasattr(loader, "chunk_bounds") or not hasattr(loader, "macro_chunk_indices"):
        return loader

    try:
        if hasattr(loader, "stop_event"):
            loader.stop_event.set()
        if hasattr(loader, "_drain_and_cleanup_prefetch_queue"):
            loader._drain_and_cleanup_prefetch_queue()
        if hasattr(loader, "prefetch_thread") and loader.prefetch_thread.is_alive():
            loader.prefetch_thread.join(timeout=2.0)
        if hasattr(loader, "_drain_and_cleanup_prefetch_queue"):
            loader._drain_and_cleanup_prefetch_queue()
    except Exception:
        pass

    def _load_payload_sync(macro_ptr):
        if macro_ptr >= len(loader.macro_chunk_indices):
            return None
        chunk_idx = loader.macro_chunk_indices[macro_ptr]
        start_idx, end_idx = loader.chunk_bounds[chunk_idx]
        cur_chunk_size = end_idx - start_idx
        active_chunk_embs = []
        cached_paths = []
        for model_name in loader.models:
            pattern = loader.chunk_patterns.get(
                model_name, f"{model_name}_chunk_{{start:07d}}_{{end:07d}}.npz"
            )
            model_end = loader._per_model_bounds.get(model_name, {}).get(start_idx, end_idx)
            cand_name = pattern.format(start=start_idx, end=model_end)
            if hasattr(loader, "_resolve_npz_path"):
                path = loader._resolve_npz_path(cand_name)
            else:
                path = os.path.join(loader.npz_dir, cand_name)
            arr = load_npz_features_with_repair(loader, path, cand_name=cand_name)
            if arr.shape[0] > cur_chunk_size:
                arr = arr[:cur_chunk_size]
            active_chunk_embs.append(arr)
            cached_paths.append(path)

        eff_chunk_rows = (
            min(cur_chunk_size, *(int(a.shape[0]) for a in active_chunk_embs))
            if active_chunk_embs
            else cur_chunk_size
        )
        if eff_chunk_rows < cur_chunk_size:
            active_chunk_embs = [a[:eff_chunk_rows] for a in active_chunk_embs]

        if loader.shuffle and eff_chunk_rows > 1:
            rng = np.random.default_rng(loader.seed + macro_ptr + loader.current_epoch)
            if getattr(loader, "pair_shuffle", False) and eff_chunk_rows >= 2:
                n_pairs = eff_chunk_rows // 2
                pairs = np.arange(n_pairs * 2).reshape(n_pairs, 2)
                perm = rng.permutation(n_pairs)
                shuffled_pairs = pairs[perm].reshape(-1)
                if eff_chunk_rows % 2 == 1:
                    active_micro_indices = np.concatenate(
                        [shuffled_pairs, np.array([eff_chunk_rows - 1], dtype=shuffled_pairs.dtype)]
                    )
                else:
                    active_micro_indices = shuffled_pairs
            else:
                active_micro_indices = np.arange(eff_chunk_rows)
                rng.shuffle(active_micro_indices)
        else:
            active_micro_indices = np.arange(eff_chunk_rows)

        return {
            "embs": active_chunk_embs,
            "micro_indices": active_micro_indices,
            "global_start": start_idx,
            "macro_idx_ptr": macro_ptr,
            "cached_paths": cached_paths,
        }

    def _start_prefetching_sync(start_macro_ptr):
        loader._sync_next_macro_ptr = start_macro_ptr

    def _pop_next_chunk_sync():
        if loader.auto_cleanup and getattr(loader, "_prev_chunk_paths", []):
            loader._cleanup_cached_files(loader._prev_chunk_paths)
            loader._prev_chunk_paths = []
        if not hasattr(loader, "_sync_next_macro_ptr"):
            loader._sync_next_macro_ptr = getattr(loader, "active_macro_idx_ptr", 0)
        payload = _load_payload_sync(loader._sync_next_macro_ptr)
        if payload is None:
            raise StopIteration
        loader.active_chunk_embs = payload["embs"]
        loader.active_micro_indices = payload["micro_indices"]
        loader.active_global_start = payload["global_start"]
        loader.active_macro_idx_ptr = payload["macro_idx_ptr"]
        loader.active_micro_ptr = 0
        loader._prev_chunk_paths = payload.get("cached_paths", [])
        loader._sync_next_macro_ptr += 1
        if not hasattr(loader, "emb_dims"):
            loader.emb_dims = [arr.shape[-1] for arr in loader.active_chunk_embs]
            print(f"[DataLoader] 自动侦测到多路特征维度: {loader.emb_dims}")

    loader._start_prefetching = _start_prefetching_sync
    loader._pop_next_chunk = _pop_next_chunk_sync
    return loader


def enable_download_only_prefetch(loader):
    return disable_chunk_prefetch(loader)


# Bug Fix (Task 070):
# 1) Problem: Relying on `training.core.checkpoint.Checkpointer` installed global SIGINT/SIGTERM
#    handlers that called `sys.exit(0)` on emergency save, lacked rolling `keep_last_ckpts` pruning,
#    only saved `.pt` weights (omitting `phase1_charencoder.safetensors` required by downstream
#    Translator & Phase 4 Left Brain), and lacked `resolve_latest_phase1_v2_checkpoint` auto-resume.
# 2) Method: Implement native Phase 1 v2 checkpoint lifecycle (`expected_checkpoint_dir`,
#    `_is_valid_phase1_v2_ckpt_dir`, `resolve_latest_phase1_v2_checkpoint`,
#    `prune_old_phase1_v2_checkpoints`, and `save_phase1_v2_checkpoint`) matching Phase 0 v3 1:1
#    and exporting `phase1_charencoder.safetensors`, `student.safetensors`, `student.pt`, and
#    `char_subspace_stats.pt` under `assert_colab_drive_mounted`.
# 3) Caveats: `training/core/` submodule remains strictly untouched; downstream stages can load
#    either `phase1_charencoder.safetensors` or `student.pt` interchangeably.
def expected_checkpoint_dir(
    out_dir: str,
    prefix: str,
    step: int,
    is_emergency: bool = False,
) -> str:
    base_name = "latest_emergency" if is_emergency else f"step_{step}"
    folder_name = f"{prefix}_{base_name}" if prefix else base_name
    return os.path.join(out_dir, folder_name)


def _is_valid_phase1_v2_ckpt_dir(ckpt_dir: str) -> Tuple[bool, Optional[int]]:
    if not ckpt_dir or not os.path.isdir(ckpt_dir):
        return False, None

    student_files = (
        "phase1_charencoder.safetensors",
        "student.safetensors",
        "student.pt",
    )
    has_student = any(
        os.path.isfile(os.path.join(ckpt_dir, fn))
        and os.path.getsize(os.path.join(ckpt_dir, fn)) > 0
        for fn in student_files
    )
    if not has_student:
        return False, None

    stats_path = os.path.join(ckpt_dir, "char_subspace_stats.pt")
    if not os.path.isfile(stats_path) or os.path.getsize(stats_path) <= 0:
        return False, None

    dl_path = os.path.join(ckpt_dir, "dataloader.json")
    if os.path.isfile(dl_path) and os.path.getsize(dl_path) > 0:
        try:
            with open(dl_path, "r", encoding="utf-8") as f:
                dl_meta = json.load(f)
            if isinstance(dl_meta, dict) and dl_meta.get("global_step") is not None:
                return True, int(dl_meta["global_step"])
        except (OSError, ValueError, TypeError):
            pass

    try:
        st_meta = torch.load(stats_path, map_location="cpu", weights_only=False)
    except Exception:
        return False, None
    if isinstance(st_meta, dict) and st_meta.get("global_step") is not None:
        try:
            return True, int(st_meta["global_step"])
        except (ValueError, TypeError):
            pass

    m = re.search(r"(?:^|_)step_(\d+)$", os.path.basename(os.path.normpath(ckpt_dir)))
    return True, (int(m.group(1)) if m else 0)


def resolve_latest_phase1_v2_checkpoint(out_dir: str, prefix: str) -> Optional[str]:
    if not out_dir or not os.path.isdir(out_dir):
        return None

    step_re = re.compile(
        rf"^{re.escape(prefix)}_step_(\d+)$" if prefix else r"^step_(\d+)$"
    )
    emerg_name = f"{prefix}_latest_emergency" if prefix else "latest_emergency"

    candidates: List[Tuple[int, int, str]] = []
    for entry in os.listdir(out_dir):
        is_emerg = entry == emerg_name
        is_step = bool(step_re.match(entry))
        if not (is_emerg or is_step):
            continue
        cand_path = os.path.join(out_dir, entry)
        ok, step_val = _is_valid_phase1_v2_ckpt_dir(cand_path)
        if not ok or step_val is None:
            continue
        non_emerg_rank = 0 if is_emerg else 1
        candidates.append((step_val, non_emerg_rank, cand_path))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[-1][2]


def prune_old_phase1_v2_checkpoints(out_dir: str, prefix: str, keep_last: int) -> List[str]:
    removed: List[str] = []
    if keep_last <= 0 or not out_dir or not os.path.isdir(out_dir):
        return removed

    step_re = re.compile(
        rf"^{re.escape(prefix)}_step_(\d+)$" if prefix else r"^step_(\d+)$"
    )
    emerg_name = f"{prefix}_latest_emergency" if prefix else "latest_emergency"

    valid_steps: List[Tuple[int, str]] = []
    for entry in os.listdir(out_dir):
        m = step_re.match(entry)
        if not m:
            continue
        cand_path = os.path.join(out_dir, entry)
        ok, step_val = _is_valid_phase1_v2_ckpt_dir(cand_path)
        if not ok or step_val is None:
            continue
        valid_steps.append((step_val, cand_path))

    valid_steps.sort(key=lambda item: (item[0], item[1]))
    if len(valid_steps) > keep_last:
        for _old_step, old_path in valid_steps[:-keep_last]:
            shutil.rmtree(old_path, ignore_errors=True)
            removed.append(old_path)

    if valid_steps:
        newest_regular_step = valid_steps[-1][0]
        emerg_path = os.path.join(out_dir, emerg_name)
        if os.path.isdir(emerg_path):
            ok_emerg, emerg_step = _is_valid_phase1_v2_ckpt_dir(emerg_path)
            if (not ok_emerg) or (emerg_step is not None and emerg_step <= newest_regular_step):
                shutil.rmtree(emerg_path, ignore_errors=True)
                removed.append(emerg_path)

    return removed


def save_phase1_v2_checkpoint(
    out_dir: str,
    prefix: str,
    step: int,
    student: TinyCharEncoderCUDA,
    order_head: Optional[OrderDirectionHeadCUDA] = None,
    next_clause_head: Optional[AsymmetricNextClauseHeadCUDA] = None,
    char_anchor_head: Optional[CharSubspaceAnchorHeadCUDA] = None,
    optimizer: Optional[optim.Optimizer] = None,
    dataloader=None,
    args: Optional[argparse.Namespace] = None,
    health_metrics: Optional[Dict[str, float]] = None,
    is_emergency: bool = False,
) -> str:
    assert_colab_drive_mounted(out_dir)
    ckpt_dir = expected_checkpoint_dir(out_dir, prefix, step, is_emergency=is_emergency)
    assert_colab_drive_mounted(ckpt_dir)
    os.makedirs(ckpt_dir, exist_ok=True)

    student_cpu_state = {
        k: v.detach().cpu().contiguous() for k, v in student.state_dict().items()
    }
    save_safetensors_file(
        student_cpu_state, os.path.join(ckpt_dir, "phase1_charencoder.safetensors")
    )
    save_safetensors_file(
        student_cpu_state, os.path.join(ckpt_dir, "student.safetensors")
    )
    torch.save(student_cpu_state, os.path.join(ckpt_dir, "student.pt"))

    if order_head is not None:
        torch.save(order_head.state_dict(), os.path.join(ckpt_dir, "order_head.pt"))
    if next_clause_head is not None:
        torch.save(next_clause_head.state_dict(), os.path.join(ckpt_dir, "next_clause_head.pt"))
    if char_anchor_head is not None:
        torch.save(char_anchor_head.state_dict(), os.path.join(ckpt_dir, "char_anchor_head.pt"))
    if optimizer is not None:
        torch.save(optimizer.state_dict(), os.path.join(ckpt_dir, "optimizer.pt"))

    if dataloader is not None and hasattr(dataloader, "state_dict"):
        dl_state = dict(dataloader.state_dict())
        dl_state["global_step"] = int(step)
        with open(os.path.join(ckpt_dir, "dataloader.json"), "w", encoding="utf-8") as f:
            json.dump(dl_state, f)

    if args is not None:
        with open(os.path.join(ckpt_dir, "training_args.json"), "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=4, ensure_ascii=False)

    w_sub = student.char_subspace_proj.weight.detach().cpu().contiguous()
    stats_payload = {
        "char_subspace_proj": w_sub,
        "subspace_proj": w_sub.t().contiguous(),
        "char_token_mean": student.char_token_mean.detach().cpu().contiguous(),
        "d_sub": int(student.d_sub),
        "d_model": int(student.d_model),
        "pad_token_id": int(student.pad_token_id),
        "global_step": int(step),
        "health_metrics": dict(health_metrics) if health_metrics else {},
    }
    torch.save(stats_payload, os.path.join(ckpt_dir, "char_subspace_stats.pt"))
    return ckpt_dir


def save_char_subspace_stats(
    student: TinyCharEncoderCUDA,
    out_dir: str,
    prefix: str,
    step: int,
    is_emergency: bool = False,
    health_metrics: Optional[Dict[str, float]] = None,
):
    """
    Backward-compatible helper saving `char_subspace_stats.pt` containing the native orthogonal
    32-D basis and valid-token mean offset for downstream Phase 4/5.
    """
    assert_colab_drive_mounted(out_dir)
    ckpt_dir = expected_checkpoint_dir(out_dir, prefix, step, is_emergency=is_emergency)
    assert_colab_drive_mounted(ckpt_dir)
    os.makedirs(ckpt_dir, exist_ok=True)
    stats_path = os.path.join(ckpt_dir, "char_subspace_stats.pt")
    w_sub = student.char_subspace_proj.weight.detach().cpu().contiguous()
    payload = {
        "char_subspace_proj": w_sub,
        "subspace_proj": w_sub.t().contiguous(),
        "char_token_mean": student.char_token_mean.detach().cpu().contiguous(),
        "d_sub": int(student.d_sub),
        "d_model": int(student.d_model),
        "pad_token_id": int(student.pad_token_id),
        "global_step": int(step),
        "health_metrics": dict(health_metrics) if health_metrics else {},
    }
    torch.save(payload, stats_path)
    print(f"[Checkpoint] Saved char_subspace_stats.pt to {stats_path}")
    return stats_path


def save_torch_optimizer_state(
    optimizer, out_dir: str, prefix: str, step: int, is_emergency: bool = False
):
    assert_colab_drive_mounted(out_dir)
    ckpt_dir = expected_checkpoint_dir(out_dir, prefix, step, is_emergency=is_emergency)
    assert_colab_drive_mounted(ckpt_dir)
    os.makedirs(ckpt_dir, exist_ok=True)
    opt_path = os.path.join(ckpt_dir, "optimizer.pt")
    torch.save(optimizer.state_dict(), opt_path)
    print(f"[Checkpoint] Saved optimizer state to {opt_path}")


def _build_batch_contiguous_pairs(
    dataloader, batch_size: int, device: torch.device
) -> Optional[torch.Tensor]:
    """
    Extracts true adjacent clause index pairs `(i, j)` within the current batch where
    `global_idx[j] == global_idx[i] + 1` so `AsymmetricNextClauseHeadCUDA` never treats
    random shuffled sentences as adjacent discourse transitions.
    """
    g_idx = getattr(dataloader, "last_batch_global_indices", None)
    if g_idx is None:
        return None
    if len(g_idx) != batch_size or batch_size < 2:
        return torch.empty((0, 2), dtype=torch.long, device=device)

    pos_map = {int(g): idx for idx, g in enumerate(g_idx)}
    pairs: List[List[int]] = []
    for i, g in enumerate(g_idx):
        nxt_pos = pos_map.get(int(g) + 1)
        if nxt_pos is not None and nxt_pos != i:
            pairs.append([i, nxt_pos])
    if not pairs:
        return torch.empty((0, 2), dtype=torch.long, device=device)
    return torch.tensor(pairs, dtype=torch.long, device=device)


def get_distill_parser():
    parser = argparse.ArgumentParser(
        description="Phase 1 v2 Dual-Head Masked Character Encoder Distillation (Zero WeakDecoder CE)"
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=5000)
    parser.add_argument(
        "--total_steps",
        type=int,
        default=0,
        help="Total steps for cosine LR schedule (0 = auto-compute from dataset size * epochs)",
    )
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--save_steps", type=int, default=2000)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--log_steps", type=int, default=20)
    parser.add_argument("--out_dir", type=str, default="checkpoints/p1_v2")
    parser.add_argument("--ckpt_prefix", type=str, default="charencoder_pt_v2")
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument(
        "--auto_resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatically resume from latest valid checkpoint in --out_dir (pass --no-auto_resume to disable)",
    )
    parser.add_argument(
        "--keep_last_ckpts",
        type=int,
        default=3,
        help="Keep only the N most recent step_* checkpoints in --out_dir (0 = keep all)",
    )
    parser.add_argument("--p0_ckpt", type=str, required=True, help="Frozen Phase 0 checkpoint to distill from")
    parser.add_argument("--config_file", type=str, default=None, help="Optional dataset config JSON")
    parser.add_argument(
        "--data_source",
        type=str,
        default="ms",
        choices=["ms", "gdrive", "hf", "local", "modelscope", "huggingface"],
        help="Data source: 'ms' (ModelScope, default), 'gdrive' (Google Drive staged NVMe), 'hf' (Hugging Face CDN), or 'local'",
    )
    parser.add_argument(
        "--gdrive_dir",
        type=str,
        default="/content/drive/MyDrive/hollyshit/embs",
        help="Google Drive directory holding persistent .npz chunks (used when --data_source gdrive)",
    )
    parser.add_argument(
        "--hf_repo_id",
        type=str,
        default=None,
        help="Optional Hugging Face dataset repo ID overriding dataset_config.json (e.g. drunkencat/emb_npy)",
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
    parser.add_argument(
        "--grad_checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable layer-wise gradient checkpointing on TinyCharEncoderCUDA (pass --no-grad_checkpoint to disable)",
    )
    return parser


def main():
    parser = get_distill_parser()
    args = parser.parse_args()

    # Hard Drive mount guard at second 0 (before dataloaders or models!)
    assert_colab_drive_mounted(args.out_dir)
    if args.data_source in ("gdrive",):
        assert_colab_drive_mounted(args.gdrive_dir)

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

    p0_ckpt_resolved = resolve_p0_teacher_ckpt_dir(args.p0_ckpt)
    fuser_path, god_path, emb_dims, d_model, z_dim = infer_teacher_arch_from_ckpt(p0_ckpt_resolved)
    teacher_sphere_scale = _infer_teacher_sphere_scale(p0_ckpt_resolved)
    print(
        f"\n[Teacher Arch] ckpt={p0_ckpt_resolved} | routes={len(emb_dims)} emb_dims={emb_dims} "
        f"d_model={d_model} z_dim={z_dim} sphere_scale={teacher_sphere_scale}"
    )

    dataloader = build_dataloader(args, tokenizer, ds_config, expected_routes=len(emb_dims))
    dataloader = attach_chunk_download_router(dataloader, args.data_dir, args.download_cache_dir)
    if args.download_only_prefetch:
        dataloader = enable_download_only_prefetch(dataloader)
    elif not args.prefetch:
        dataloader = disable_chunk_prefetch(dataloader)

    if args.total_steps <= 0:
        eff_total_steps = max(
            1,
            int(math.ceil(max(getattr(dataloader, "total_samples", 0), 1) / float(args.batch_size)))
            * args.epochs,
        )
    else:
        eff_total_steps = args.total_steps

    # 1. Instantiate the Frozen Teacher Replicas (SensoryFuser + GodEncoder; NO WeakDecoder!)
    fuser = SensoryFuserCUDA(emb_dims, d_model).to(device)
    god_encoder = GodEncoderCUDA(d_model, z_dim, sphere_scale=teacher_sphere_scale).to(device)

    print(f"\n[Teacher] Loading Frozen Phase 0 Teacher from {p0_ckpt_resolved} (Zero WeakDecoder CE)...")
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
    # Optimization (Task 071):
    # 1) Problem: Uncheckpointed 6-layer dual-pass activations exceed L4/A100 VRAM at Batch=512/1024.
    # 2) Method: Apply `args.grad_checkpoint` (default True) via `student.set_gradient_checkpointing(...)`.
    # 3) Caveats: Trades ~20-30% recomputation for major activation VRAM savings with identical grads.
    student.set_gradient_checkpointing(bool(args.grad_checkpoint))
    print(f"[Student] Layer gradient checkpointing: {'ENABLED' if args.grad_checkpoint else 'DISABLED'}")

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

    global_step = 0
    last_health: Dict[str, float] = {}

    resume_dir = args.resume_from
    if not resume_dir and args.auto_resume:
        resume_dir = resolve_latest_phase1_v2_checkpoint(args.out_dir, args.ckpt_prefix)
        if resume_dir:
            print(f"[Auto-Resume] Found latest valid Phase 1 v2 checkpoint: {resume_dir}")

    if resume_dir and os.path.isdir(resume_dir):
        args.resume_from = resume_dir
        print(f"[Resume] Loading Phase 1 v2 state from {resume_dir}...")
        stu_pt = os.path.join(resume_dir, "student.pt")
        stu_st1 = os.path.join(resume_dir, "phase1_charencoder.safetensors")
        stu_st2 = os.path.join(resume_dir, "student.safetensors")
        if os.path.isfile(stu_pt):
            student.load_state_dict(
                torch.load(stu_pt, map_location=device, weights_only=True), strict=False
            )
        elif os.path.isfile(stu_st1):
            load_mlx_safetensors_into_torch(student, stu_st1)
        elif os.path.isfile(stu_st2):
            load_mlx_safetensors_into_torch(student, stu_st2)

        oh_path = os.path.join(resume_dir, "order_head.pt")
        if os.path.isfile(oh_path):
            order_head.load_state_dict(torch.load(oh_path, map_location=device, weights_only=True))
        nh_path = os.path.join(resume_dir, "next_clause_head.pt")
        if os.path.isfile(nh_path):
            next_clause_head.load_state_dict(
                torch.load(nh_path, map_location=device, weights_only=True)
            )
        ca_path = os.path.join(resume_dir, "char_anchor_head.pt")
        if os.path.isfile(ca_path):
            char_anchor_head.load_state_dict(
                torch.load(ca_path, map_location=device, weights_only=True)
            )
        opt_path = os.path.join(resume_dir, "optimizer.pt")
        if os.path.isfile(opt_path):
            optimizer.load_state_dict(torch.load(opt_path, map_location=device, weights_only=False))

        dl_path = os.path.join(resume_dir, "dataloader.json")
        if os.path.isfile(dl_path):
            with open(dl_path, "r", encoding="utf-8") as f:
                dl_state = json.load(f)
            dataloader.load_state_dict(dl_state)
            global_step = int(dl_state.get("global_step", 0))

        stats_path = os.path.join(resume_dir, "char_subspace_stats.pt")
        if os.path.isfile(stats_path):
            st_meta = torch.load(stats_path, map_location="cpu", weights_only=False)
            if isinstance(st_meta, dict):
                if isinstance(st_meta.get("health_metrics"), dict):
                    last_health = {k: float(v) for k, v in st_meta["health_metrics"].items()}
                if global_step == 0 and st_meta.get("global_step") is not None:
                    global_step = int(st_meta["global_step"])

    print(f"\n========================================================")
    print(f"🚀 Phase 1 v2 Dual-Head Masked Engine Ignited! Target Epochs: {args.epochs}")
    print(
        f"⚙️  Batch: {args.batch_size} | d_sub: {args.student_d_sub} | Peak LR: {args.lr} | "
        f"Total Steps: {eff_total_steps}"
    )
    print(f"========================================================\n")

    autocast_dev = "cuda" if device.type == "cuda" else "cpu"

    # Bug Fix (Task 070):
    # 1) Problem: The training loop previously called `compute_phase1_v2_distill_loss(z_god=...)`
    #    unpacking 3 values, invoked non-existent `student.update_subspace_basis()` /
    #    `student.embed_tokens()`, and read stale metric keys (`metrics['loss']`, etc.).
    # 2) Method: Align all arguments (`z_target_truth=z_god`, `batch_embs=batch_embs`,
    #    `contiguous_pairs=...`, `update_ema=True`), unpack `(loss, metrics)`, and record both
    #    canonical `char_encoder_v2_loss` health keys and summary aliases in `last_health`.
    # 3) Caveats: `char_subspace_stats.pt['health_metrics']` exposes both canonical and alias keys
    #    so Notebook summaries and diagnostic scripts remain 100% compatible.
    try:
        for epoch in range(dataloader.current_epoch, args.epochs):
            for token_inputs, batch_embs, attention_mask in dataloader:
                global_step += 1
                if len(batch_embs) != len(emb_dims):
                    if args.strict_route_check or len(batch_embs) < len(emb_dims):
                        raise ValueError(
                            f"Runtime embedding route mismatch: got {len(batch_embs)}, expected {len(emb_dims)}"
                        )
                    batch_embs = batch_embs[: len(emb_dims)]

                lr = cosine_warmup_lr_schedule(
                    global_step, args.lr, args.warmup_steps, eff_total_steps
                )
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

                token_inputs = token_inputs.to(device, non_blocking=True)
                attention_mask = (
                    attention_mask.to(device, non_blocking=True)
                    if attention_mask is not None
                    else None
                )
                batch_embs = [emb.to(device, non_blocking=True).float() for emb in batch_embs]
                contig_pairs = _build_batch_contiguous_pairs(
                    dataloader, int(token_inputs.size(0)), device
                )

                optimizer.zero_grad(set_to_none=True)

                with torch.no_grad():
                    e_fused = fuser(batch_embs)
                    z_god = god_encoder(e_fused)

                with torch.autocast(device_type=autocast_dev, dtype=amp_dtype, enabled=use_amp):
                    loss, metrics = compute_phase1_v2_distill_loss(
                        student=student,
                        token_inputs=token_inputs,
                        attention_mask=attention_mask,
                        z_target_truth=z_god,
                        batch_embs=batch_embs,
                        order_head=order_head,
                        next_clause_head=next_clause_head,
                        char_anchor_head=char_anchor_head,
                        contiguous_pairs=contig_pairs,
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

                if global_step % args.log_steps == 0:
                    print(
                        f"Epoch {epoch + 1} | Step {global_step} | LR {lr:.2e} | "
                        f"Loss {float(metrics['loss_total']):.4f} "
                        f"(Distill {float(metrics['loss_distill']):.3f} [cos={float(metrics['z_god_cos']):.3f}] | "
                        f"Rel {float(metrics['loss_rel']):.3f} | "
                        f"Order {float(metrics['loss_order']):.3f} [acc={float(metrics['order_discrim_acc']):.2f}] | "
                        f"Next {float(metrics['loss_next']):.3f} | "
                        f"CharIso {float(metrics['loss_char_iso']):.3f} [PC1={float(metrics['subspace_pc1_var_ratio']):.3f}] | "
                        f"CharID {float(metrics['loss_char_id']):.3f} [top1={float(metrics['subspace_char_top1_acc']):.2f}])"
                    )

                if global_step % args.eval_steps == 0:
                    eval_metrics = evaluate_phase1_v2_health(
                        student=student,
                        token_inputs=token_inputs,
                        attention_mask=attention_mask,
                        z_target_truth=z_god,
                        batch_embs=batch_embs,
                        order_head=order_head,
                        char_anchor_head=char_anchor_head,
                    )
                    with torch.no_grad():
                        ortho_dev = float(student.orthogonal_subspace_loss().sqrt().item())
                    last_health = dict(eval_metrics)
                    last_health["mean_distill_cos"] = float(eval_metrics["z_god_cos"])
                    last_health["order_pair_acc"] = float(eval_metrics["order_discrim_acc"])
                    last_health["char_subspace_ortho_dev"] = ortho_dev
                    last_health["char_anchor_top1_acc"] = float(
                        eval_metrics["subspace_char_top1_acc"]
                    )
                    print(
                        f"  🩺 [Health @ Step {global_step}] "
                        f"DistillCos={last_health['mean_distill_cos']:.3f} | "
                        f"Spearman={last_health['relational_spearman']:.3f} | "
                        f"OrderAcc={last_health['order_pair_acc']:.3f} | "
                        f"PC1={last_health['subspace_pc1_var_ratio']:.4f} | "
                        f"PR={last_health['subspace_participation_ratio']:.1f} | "
                        f"CharIsoDev={last_health['char_subspace_ortho_dev']:.4f} | "
                        f"AnchorTop1={last_health['char_anchor_top1_acc']:.3f}"
                    )

                if global_step % args.save_steps == 0:
                    saved_dir = save_phase1_v2_checkpoint(
                        out_dir=args.out_dir,
                        prefix=args.ckpt_prefix,
                        step=global_step,
                        student=student,
                        order_head=order_head,
                        next_clause_head=next_clause_head,
                        char_anchor_head=char_anchor_head,
                        optimizer=optimizer,
                        dataloader=dataloader,
                        args=args,
                        health_metrics=last_health,
                    )
                    print(f"[Checkpoint] Saved Phase 1 v2 artifacts to {saved_dir}")
                    if args.hf_ckpt_repo:
                        mirror_checkpoint_to_hf_hub(saved_dir, args.hf_ckpt_repo)
                    if args.keep_last_ckpts > 0:
                        pruned = prune_old_phase1_v2_checkpoints(
                            args.out_dir, args.ckpt_prefix, args.keep_last_ckpts
                        )
                        if pruned:
                            print(
                                f"[Checkpoint] Pruned {len(pruned)} old checkpoint(s): "
                                f"{[os.path.basename(p) for p in pruned]}"
                            )

        final_dir = save_phase1_v2_checkpoint(
            out_dir=args.out_dir,
            prefix=args.ckpt_prefix,
            step=global_step,
            student=student,
            order_head=order_head,
            next_clause_head=next_clause_head,
            char_anchor_head=char_anchor_head,
            optimizer=optimizer,
            dataloader=dataloader,
            args=args,
            health_metrics=last_health,
        )
        if args.hf_ckpt_repo:
            mirror_checkpoint_to_hf_hub(final_dir, args.hf_ckpt_repo)
        if args.keep_last_ckpts > 0:
            prune_old_phase1_v2_checkpoints(args.out_dir, args.ckpt_prefix, args.keep_last_ckpts)
        print(f"🎉 Phase 1 v2 Distillation Training Complete! Final checkpoint: {final_dir}")

    except KeyboardInterrupt:
        print(f"\n[Interrupt] Saving emergency Phase 1 v2 checkpoint at step {global_step}...")
        emerg_dir = save_phase1_v2_checkpoint(
            out_dir=args.out_dir,
            prefix=args.ckpt_prefix,
            step=global_step,
            student=student,
            order_head=order_head,
            next_clause_head=next_clause_head,
            char_anchor_head=char_anchor_head,
            optimizer=optimizer,
            dataloader=dataloader,
            args=args,
            health_metrics=last_health,
            is_emergency=True,
        )
        if args.hf_ckpt_repo:
            mirror_checkpoint_to_hf_hub(emerg_dir, args.hf_ckpt_repo)
    # Bug Fix (Task 069 & Task 070):
    # 1) Problem: Runtime exceptions after hours of distillation exited without saving an emergency
    #    checkpoint (or previously relied on Checkpointer which called sys.exit(0)).
    # 2) Method: Catch `Exception as crash_err` when `global_step > 0`, save `is_emergency=True`
    #    via `save_phase1_v2_checkpoint` (plus optional HF mirror), and re-raise `crash_err`.
    # 3) Caveats: Always re-raises the original exception so failures remain visible.
    except Exception as crash_err:
        if global_step > 0:
            print(
                f"\n⚠️ [Crash-Guard] Unexpected error at step {global_step}: {crash_err}\n"
                f"💾 Saving emergency Phase 1 v2 checkpoint before exiting..."
            )
            try:
                emerg_dir = save_phase1_v2_checkpoint(
                    out_dir=args.out_dir,
                    prefix=args.ckpt_prefix,
                    step=global_step,
                    student=student,
                    order_head=order_head,
                    next_clause_head=next_clause_head,
                    char_anchor_head=char_anchor_head,
                    optimizer=optimizer,
                    dataloader=dataloader,
                    args=args,
                    health_metrics=last_health,
                    is_emergency=True,
                )
                print(f"[Crash-Guard] ✅ Saved emergency checkpoint to {emerg_dir}")
                if args.hf_ckpt_repo:
                    mirror_checkpoint_to_hf_hub(emerg_dir, args.hf_ckpt_repo)
            except Exception as save_err:
                print(f"[Crash-Guard] ❌ Failed to save emergency checkpoint: {save_err}")
        raise
    finally:
        if hasattr(dataloader, "close"):
            try:
                dataloader.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
