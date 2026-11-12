#!/usr/bin/env python3
"""
Cloud-to-Cloud Streaming Dataset Migration:
  ModelScope (`catclever/emb_npy`) -> Hugging Face Hub (`<hf_user>/emb_npy`, Public Dataset)

Key Properties:
  1. Zero-RAM `.npz` integrity validation (`validate_npz_archive`): reads Zip central directory
     at EOF + `.npy` magic/header ONLY (< 5 MB RAM; never decompresses 2.6 GB arrays).
  2. Resumable Macro-Chunk Streaming: queries `HfApi.list_repo_files` upfront, skips any
     completed chunks (`⏭️ [Skip]`), stages missing teacher `.npz` files per chunk, commits
     each chunk in 1 single atomic HF commit (`api.create_commit`), and immediately wipes
     staging + ModelScope download cache so peak disk stays <= 8.5 GB across all 18 chunks.
"""

import argparse
import json
import os
import re
import shutil
import sys
import time
import zipfile
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np


DEFAULT_DATASET_CONFIG: Dict[str, Any] = {
    "huggingface_repo_id": "drunkencat/emb_npy",
    "modelscope_repo_id": "catclever/emb_npy",
    "base_models": ["bge", "qwen", "xiaobu", "youtu", "conan_v1"],
    "chunk_name_patterns": {
        "bge": "bge_m3/bge_embeddings_chunk_{start:07d}_{end:07d}.npz",
        "qwen": "gte_qwen/gte_qwen2_embeddings_chunk_{start:07d}_{end:07d}.npz",
        "xiaobu": "xiaobu/xiaobu_embeddings_chunk_{start:07d}_{end:07d}.npz",
        "youtu": "youtu/youtu_embeddings_chunk_{start:07d}_{end:07d}.npz",
        "conan_v1": "conan/conan_v1_embeddings_chunk_{start:07d}_{end:07d}.npz",
    },
    "emb_dims": [1024, 3584, 1792, 2048, 1792],
    "parquet_path": "data/Basic_ZH/chunked_mixed_omni.parquet",
}

_INTERNAL_SKIP_FILES: Set[str] = {
    ".mdl",
    ".msc",
    ".gitattributes",
    ".gitignore",
    "README.md",
}


def validate_npz_archive(
    npz_path: str,
    expected_key: str = "features",
) -> Tuple[Tuple[int, ...], str]:
    """
    Zero-RAM `.npz` integrity & header validator (< 5 MB RAM even on 2.8 GB `.npz` files):
      1. Verifies file exists and `os.path.getsize(npz_path) > 0`.
      2. Opens `zipfile.ZipFile(npz_path, "r")` to validate EOF central directory (catches
         truncated downloads immediately via `zipfile.BadZipFile`).
      3. Verifies `f"{expected_key}.npy"` exists in `zf.namelist()`.
      4. Streams ONLY the `.npy` magic & header (`read_magic` + `read_array_header_*`)
         without decompressing array payload into RAM.
      5. Asserts 2D non-empty shape `(N, D)` and verifies uncompressed member size exceeds
         `N * D * itemsize`.
    """
    if not npz_path or not os.path.isfile(npz_path):
        raise FileNotFoundError(f"NPZ file does not exist: {npz_path}")
    file_size = os.path.getsize(npz_path)
    if file_size <= 0:
        raise ValueError(f"NPZ file is empty (0 bytes): {npz_path}")

    member_name = f"{expected_key}.npy"
    with zipfile.ZipFile(npz_path, "r") as zf:
        names = zf.namelist()
        if member_name not in names:
            raise KeyError(
                f"Expected array member {member_name!r} missing in {npz_path}; found={names}"
            )
        zinfo = zf.getinfo(member_name)
        with zf.open(member_name, "r") as fp:
            version = np.lib.format.read_magic(fp)
            if version == (1, 0):
                shape, _fortran_order, dtype = np.lib.format.read_array_header_1_0(fp)
            elif version == (2, 0):
                shape, _fortran_order, dtype = np.lib.format.read_array_header_2_0(fp)
            elif hasattr(np.lib.format, "_read_array_header"):
                shape, _fortran_order, dtype = np.lib.format._read_array_header(fp, version)
            else:
                raise ValueError(f"Unsupported .npy version {version} in {npz_path}")

    if not isinstance(shape, tuple) or len(shape) != 2 or shape[0] <= 0 or shape[1] <= 0:
        raise ValueError(
            f"Invalid embedding matrix shape {shape!r} in {npz_path} (expected 2D with N>0, D>0)"
        )

    np_dtype = np.dtype(dtype)
    expected_payload_bytes = int(shape[0]) * int(shape[1]) * int(np_dtype.itemsize)
    if int(zinfo.file_size) <= expected_payload_bytes:
        raise ValueError(
            f"Corrupted/truncated .npy entry {member_name!r} in {npz_path}: "
            f"uncompressed_size={zinfo.file_size} <= raw_payload_bytes={expected_payload_bytes}"
        )

    return (int(shape[0]), int(shape[1])), str(np_dtype)


def parse_chunk_selector(chunks_spec: Optional[str], total_chunks: int) -> List[int]:
    """
    Parses optional `--chunks` specification (e.g. `"0-17"`, `"0,3,5-8"`, `"4"`) into
    sorted 0-based chunk indices in `[0, total_chunks)`.
    """
    if total_chunks <= 0:
        return []
    if chunks_spec is None or not chunks_spec.strip():
        return list(range(total_chunks))

    selected: Set[int] = set()
    for part in chunks_spec.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            lo_str, hi_str = token.split("-", 1)
            lo, hi = int(lo_str.strip()), int(hi_str.strip())
            if lo > hi:
                raise ValueError(f"Invalid chunk range {token!r}: start > end")
            for idx in range(lo, hi + 1):
                if idx < 0 or idx >= total_chunks:
                    raise ValueError(
                        f"Chunk index {idx} out of range [0, {total_chunks - 1}]"
                    )
                selected.add(idx)
        else:
            idx = int(token)
            if idx < 0 or idx >= total_chunks:
                raise ValueError(
                    f"Chunk index {idx} out of range [0, {total_chunks - 1}]"
                )
            selected.add(idx)

    return sorted(selected)


def _get_hf_token() -> Optional[str]:
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or ""
    tok = tok.strip()
    return tok if tok else None


def _load_base_dataset_config(config_file: Optional[str] = None) -> Dict[str, Any]:
    candidates: List[str] = []
    if config_file:
        candidates.append(config_file)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates.append(os.path.join(repo_root, "dataset_config.json"))
    for cand in candidates:
        if cand and os.path.isfile(cand):
            with open(cand, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                merged = dict(DEFAULT_DATASET_CONFIG)
                merged.update(loaded)
                return merged
    return dict(DEFAULT_DATASET_CONFIG)


def _create_commit_with_retries(
    api: Any,
    repo_id: str,
    operations: List[Any],
    commit_message: str,
    max_retries: int = 3,
    backoff_base: float = 2.0,
) -> Any:
    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            return api.create_commit(
                repo_id=repo_id,
                repo_type="dataset",
                operations=operations,
                commit_message=commit_message,
            )
        except Exception as e:
            last_err = e
            print(
                f"  ⚠️ [HF-Commit] Attempt {attempt}/{max_retries} failed ({commit_message}): {e}"
            )
            if attempt < max_retries:
                sleep_s = backoff_base ** (attempt - 1)
                time.sleep(sleep_s)
    raise RuntimeError(
        f"Failed to commit to HF dataset {repo_id} after {max_retries} attempts: {last_err}"
    )


def _download_ms_file_with_retries(
    ms_downloader: Callable[..., str],
    ms_repo_id: str,
    rel_path: str,
    cache_dir: str,
    validate_npz: bool = True,
    max_retries: int = 3,
    retry_sleep: float = 1.0,
) -> Tuple[str, Optional[Tuple[Tuple[int, ...], str]]]:
    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        dl_path: Optional[str] = None
        try:
            os.makedirs(cache_dir, exist_ok=True)
            dl_path = ms_downloader(ms_repo_id, rel_path, cache_dir=cache_dir)
            if not dl_path or not os.path.isfile(dl_path):
                raise FileNotFoundError(
                    f"ModelScope downloader returned missing file path for {rel_path}: {dl_path}"
                )
            npz_meta = validate_npz_archive(dl_path) if validate_npz else None
            return dl_path, npz_meta
        except Exception as e:
            last_err = e
            print(
                f"  ⚠️ [MS-Download] Attempt {attempt}/{max_retries} failed for {rel_path}: {e}"
            )
            if dl_path and os.path.exists(dl_path):
                try:
                    os.remove(dl_path)
                except OSError:
                    pass
            # Purge any partial files inside cache_dir before retrying
            if os.path.isdir(cache_dir):
                for dp, _, fs in os.walk(cache_dir):
                    for fn in fs:
                        if fn == os.path.basename(rel_path) or fn.endswith(
                            (".tmp", ".incomplete", ".partial")
                        ):
                            try:
                                os.remove(os.path.join(dp, fn))
                            except OSError:
                                pass
            if attempt < max_retries:
                time.sleep(retry_sleep * attempt)

    raise RuntimeError(
        f"Failed to download & validate {rel_path} from ModelScope {ms_repo_id} "
        f"after {max_retries} attempts: {last_err}"
    )


def get_migration_parser() -> argparse.ArgumentParser:
    default_work_dir = (
        "/content/ms_to_hf_staging" if os.path.exists("/content") else "./ms_to_hf_staging"
    )
    parser = argparse.ArgumentParser(
        description="Cloud-to-cloud streaming migrator: ModelScope dataset -> Hugging Face Hub dataset"
    )
    parser.add_argument(
        "--ms_repo_id",
        type=str,
        default="catclever/emb_npy",
        help="Source ModelScope dataset repo ID (default: catclever/emb_npy)",
    )
    parser.add_argument(
        "--hf_repo_id",
        type=str,
        default=None,
        help="Target Hugging Face dataset repo ID (default: auto-resolves <whoami>/emb_npy)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        default=False,
        help="Create private HF dataset repo (default: False = Public dataset)",
    )
    parser.add_argument(
        "--work_dir",
        type=str,
        default=default_work_dir,
        help=f"Local ephemeral staging directory wiped after every chunk (default: {default_work_dir})",
    )
    parser.add_argument(
        "--parquet_path",
        type=str,
        default=None,
        help="Optional local/Drive chunked_mixed_omni.parquet path to upload if present",
    )
    parser.add_argument(
        "--vocab_path",
        type=str,
        default=None,
        help="Optional local char_vocab.json path to upload if present",
    )
    parser.add_argument(
        "--chunks",
        type=str,
        default=None,
        help="Optional 0-based chunk slice/list e.g. '0-17' or '0,3,5-8'",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="Optional local dataset_config.json path",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        default=False,
        help="Discover remote files and print migration plan without downloading or committing",
    )
    return parser


def run_migration(
    args: argparse.Namespace,
    hf_api: Any = None,
    ms_api: Any = None,
    ms_downloader: Optional[Callable[..., str]] = None,
    commit_op_cls: Any = None,
    retry_sleep: float = 1.0,
) -> Dict[str, Any]:
    if hf_api is None or commit_op_cls is None:
        try:
            from huggingface_hub import CommitOperationAdd, HfApi
        except ImportError:
            raise RuntimeError(
                "请安装 huggingface_hub: pip install 'huggingface_hub[hf_transfer]'"
            )
        if hf_api is None:
            hf_token = _get_hf_token()
            if not hf_token and not args.dry_run:
                raise RuntimeError(
                    "❌ 未检测到 HF_TOKEN！请在环境变量或 Colab Secrets 中配置拥有 Write 权限的 HF_TOKEN。"
                )
            hf_api = HfApi(token=hf_token)
        if commit_op_cls is None:
            commit_op_cls = CommitOperationAdd

    if ms_api is None or ms_downloader is None:
        try:
            from modelscope.hub.api import HubApi
            from modelscope.hub.file_download import dataset_file_download
        except ImportError:
            raise RuntimeError("请安装 modelscope: pip install modelscope")
        if ms_api is None:
            ms_api = HubApi()
            ms_tok = (
                os.environ.get("MODELSCOPE_API_TOKEN") or os.environ.get("MS_TOKEN") or ""
            ).strip()
            if ms_tok:
                try:
                    ms_api.login(ms_tok)
                except Exception as e:
                    print(f"⚠️ ModelScope Token 登录忽略 (使用匿名公开访问): {e}")
        if ms_downloader is None:
            ms_downloader = dataset_file_download

    hf_repo_id = args.hf_repo_id.strip() if args.hf_repo_id and args.hf_repo_id.strip() else None
    if hf_repo_id is None:
        who = hf_api.whoami()
        username = who.get("name") if isinstance(who, dict) else getattr(who, "name", None)
        if not username:
            raise RuntimeError(
                "无法从 HF_TOKEN 自动解析 Hugging Face 用户名，请显式传入 --hf_repo_id <user>/emb_npy"
            )
        hf_repo_id = f"{username}/emb_npy"

    visibility_str = "Private" if args.private else "Public"
    print("=" * 72)
    print(f"🚀 ModelScope -> Hugging Face Hub 流式云对云迁移引擎")
    print(f"   • 源仓库 (ModelScope)  : {args.ms_repo_id}")
    print(f"   • 目标仓库 (HF Dataset): {hf_repo_id} ({visibility_str})")
    print(f"   • 本地滚动暂存目录     : {args.work_dir}")
    print(f"   • 模式                 : {'DRY-RUN (仅规划)' if args.dry_run else 'LIVE STREAMING'}")
    print("=" * 72)

    # 1. Create / verify HF dataset repo
    if not args.dry_run:
        hf_api.create_repo(
            repo_id=hf_repo_id,
            repo_type="dataset",
            private=bool(args.private),
            exist_ok=True,
        )

    # 2. List already-uploaded files on HF for zero-cost breakpoint resume
    try:
        existing_hf_files: Set[str] = set(
            hf_api.list_repo_files(repo_id=hf_repo_id, repo_type="dataset")
        )
    except Exception:
        if args.dry_run:
            existing_hf_files = set()
        else:
            raise
    print(f"📡 目标 HF 仓库当前已有文件数: {len(existing_hf_files)}")

    # 3. Discover all remote files in ModelScope dataset
    raw_ms_entries = ms_api.get_dataset_files(args.ms_repo_id, recursive=True)
    ms_rel_files: List[str] = []
    for entry in raw_ms_entries:
        rel = (entry.get("Path") or entry.get("Name") or "").replace("\\", "/").lstrip("/")
        if not rel or os.path.basename(rel) in _INTERNAL_SKIP_FILES:
            continue
        if entry.get("Type") == "tree":
            continue
        ms_rel_files.append(rel)

    npz_by_start: Dict[int, List[str]] = {}
    ms_aux_files: List[str] = []
    for rel in sorted(ms_rel_files):
        if rel.endswith(".npz"):
            m = re.search(r"chunk_(\d+)_(\d+)\.npz$", os.path.basename(rel))
            if m:
                start_idx = int(m.group(1))
                npz_by_start.setdefault(start_idx, []).append(rel)
        elif rel.endswith((".parquet", ".json")):
            ms_aux_files.append(rel)

    chunk_starts = sorted(npz_by_start.keys())
    total_chunks = len(chunk_starts)
    print(
        f"📦 ModelScope 远端探测完成: 共 {total_chunks} 个宏分块 "
        f"({sum(len(v) for v in npz_by_start.values())} 个 .npz 切片文件)"
    )
    if total_chunks == 0:
        raise RuntimeError(f"未在 ModelScope {args.ms_repo_id} 中探测到任何 chunk_*.npz 文件！")

    ds_cfg = _load_base_dataset_config(args.config_file)
    ds_cfg["huggingface_repo_id"] = hf_repo_id
    ds_cfg["modelscope_repo_id"] = args.ms_repo_id

    # 4. Upload dataset_config.json + auxiliary non-npz files (.parquet / .json)
    os.makedirs(args.work_dir, exist_ok=True)
    aux_stage_dir = os.path.join(args.work_dir, "_aux_metadata")
    os.makedirs(aux_stage_dir, exist_ok=True)
    try:
        aux_ops: List[Any] = []
        if "dataset_config.json" not in existing_hf_files:
            cfg_local = os.path.join(aux_stage_dir, "dataset_config.json")
            with open(cfg_local, "w", encoding="utf-8") as f:
                json.dump(ds_cfg, f, indent=4, ensure_ascii=False)
            aux_ops.append(
                commit_op_cls(path_in_repo="dataset_config.json", path_or_fileobj=cfg_local)
            )

        for ms_aux in ms_aux_files:
            if ms_aux == "dataset_config.json" or ms_aux in existing_hf_files:
                continue
            if not args.dry_run:
                dl_aux, _ = _download_ms_file_with_retries(
                    ms_downloader=ms_downloader,
                    ms_repo_id=args.ms_repo_id,
                    rel_path=ms_aux,
                    cache_dir=os.path.join(aux_stage_dir, "ms_aux_cache"),
                    validate_npz=False,
                    retry_sleep=retry_sleep,
                )
                aux_ops.append(commit_op_cls(path_in_repo=ms_aux, path_or_fileobj=dl_aux))

        parquet_repo_rel = ds_cfg.get("parquet_path", "data/Basic_ZH/chunked_mixed_omni.parquet")
        if (
            args.parquet_path
            and os.path.isfile(args.parquet_path)
            and parquet_repo_rel not in existing_hf_files
        ):
            print(f"📄 发现本地/云盘 Parquet 语料，将同步上传至 {parquet_repo_rel}")
            aux_ops.append(
                commit_op_cls(path_in_repo=parquet_repo_rel, path_or_fileobj=args.parquet_path)
            )

        if (
            args.vocab_path
            and os.path.isfile(args.vocab_path)
            and "char_vocab.json" not in existing_hf_files
        ):
            print(f"🔤 发现本地词表 char_vocab.json，将同步上传至 char_vocab.json")
            aux_ops.append(
                commit_op_cls(path_in_repo="char_vocab.json", path_or_fileobj=args.vocab_path)
            )

        if aux_ops and not args.dry_run:
            _create_commit_with_retries(
                api=hf_api,
                repo_id=hf_repo_id,
                operations=aux_ops,
                commit_message="Upload dataset_config.json and auxiliary corpus metadata",
                backoff_base=max(1.0, retry_sleep * 2.0),
            )
            for op in aux_ops:
                existing_hf_files.add(op.path_in_repo)
            print(f"✅ 已提交 {len(aux_ops)} 个配置/辅助元数据文件至 {hf_repo_id}")
    finally:
        shutil.rmtree(aux_stage_dir, ignore_errors=True)

    # 5. Stream & commit macro-chunks one by one (peak disk <= 8.5 GB)
    selected_indices = parse_chunk_selector(args.chunks, total_chunks)
    skipped_chunks = 0
    migrated_chunks = 0
    migrated_files = 0

    for chunk_idx in selected_indices:
        start_val = chunk_starts[chunk_idx]
        chunk_files = npz_by_start[start_val]
        missing_files = [rf for rf in chunk_files if rf not in existing_hf_files]

        if not missing_files:
            skipped_chunks += 1
            print(
                f"⏭️ [Skip] Chunk {chunk_idx + 1}/{total_chunks} ({start_val:07d}) already complete on HF"
            )
            continue

        if args.dry_run:
            print(
                f"📝 [Dry-Run] Chunk {chunk_idx + 1}/{total_chunks} ({start_val:07d}): "
                f"would migrate {len(missing_files)}/{len(chunk_files)} files -> {missing_files}"
            )
            continue

        print(
            f"\n🔄 [Migrate] Chunk {chunk_idx + 1}/{total_chunks} (start={start_val:07d}): "
            f"待迁移 {len(missing_files)}/{len(chunk_files)} 个教师 .npz 分块..."
        )
        chunk_stage_dir = os.path.join(args.work_dir, f"chunk_{start_val:07d}")
        ms_cache_dir = os.path.join(chunk_stage_dir, "ms_cache")
        os.makedirs(ms_cache_dir, exist_ok=True)
        t0 = time.time()
        try:
            ops: List[Any] = []
            total_bytes = 0
            for rel_path in missing_files:
                dl_path, npz_meta = _download_ms_file_with_retries(
                    ms_downloader=ms_downloader,
                    ms_repo_id=args.ms_repo_id,
                    rel_path=rel_path,
                    cache_dir=ms_cache_dir,
                    validate_npz=True,
                    retry_sleep=retry_sleep,
                )
                fsize = os.path.getsize(dl_path)
                total_bytes += fsize
                shape_info, dtype_info = npz_meta if npz_meta else ((), "unknown")
                print(
                    f"   ✓ 已拉取并零内存校验 {rel_path} | shape={shape_info} dtype={dtype_info} "
                    f"| size={fsize / (1024 ** 2):.1f} MB"
                )
                ops.append(commit_op_cls(path_in_repo=rel_path, path_or_fileobj=dl_path))

            commit_msg = (
                f"Migrate macro-chunk {chunk_idx + 1}/{total_chunks} "
                f"(start={start_val:07d}, {len(ops)} files, {total_bytes / (1024 ** 3):.2f} GB)"
            )
            print(
                f"   ☁️ 正在单次原子提交 Chunk {chunk_idx + 1}/{total_chunks} "
                f"({total_bytes / (1024 ** 3):.2f} GB) 至 Hugging Face Hub..."
            )
            _create_commit_with_retries(
                api=hf_api,
                repo_id=hf_repo_id,
                operations=ops,
                commit_message=commit_msg,
                backoff_base=max(1.0, retry_sleep * 2.0),
            )
            existing_hf_files.update(missing_files)
            migrated_chunks += 1
            migrated_files += len(ops)
            elapsed = max(time.time() - t0, 1e-3)
            print(
                f"   ✅ Chunk {chunk_idx + 1}/{total_chunks} ({start_val:07d}) 迁移完成！耗时 {elapsed:.1f}s，"
                f"正在立即清空本地暂存盘..."
            )
        finally:
            shutil.rmtree(chunk_stage_dir, ignore_errors=True)

    # Wipe top-level work_dir if empty
    try:
        if os.path.isdir(args.work_dir) and not os.listdir(args.work_dir):
            os.rmdir(args.work_dir)
    except OSError:
        pass

    # 6. Re-query HF dataset repo and print final verification table
    if not args.dry_run:
        final_hf_files = set(hf_api.list_repo_files(repo_id=hf_repo_id, repo_type="dataset"))
    else:
        final_hf_files = existing_hf_files

    patterns = ds_cfg.get("chunk_name_patterns", DEFAULT_DATASET_CONFIG["chunk_name_patterns"])
    models = ds_cfg.get("base_models", DEFAULT_DATASET_CONFIG["base_models"])
    per_teacher_counts: Dict[str, int] = {}
    for m_name in models:
        pat = patterns.get(m_name, "")
        prefix = pat.split("/")[0] if "/" in pat else m_name
        cnt = sum(
            1
            for rf in final_hf_files
            if rf.endswith(".npz") and (rf.startswith(prefix + "/") or f"/{prefix}/" in rf)
        )
        per_teacher_counts[m_name] = cnt

    dataset_url = f"https://huggingface.co/datasets/{hf_repo_id}"
    print("\n" + "=" * 72)
    print("📊 Hugging Face 数据集迁移终态校验表")
    print("-" * 72)
    print(f"{'Teacher Route':<16} | {'Prefix':<14} | {'HF .npz Chunks':<16} | {'Status'}")
    print("-" * 72)
    for m_name in models:
        pat = patterns.get(m_name, "")
        prefix = pat.split("/")[0] if "/" in pat else m_name
        cnt = per_teacher_counts[m_name]
        status = "✅ COMPLETE" if cnt >= total_chunks else f"⚠️ {cnt}/{total_chunks}"
        print(f"{m_name:<16} | {prefix:<14} | {f'{cnt}/{total_chunks}':<16} | {status}")
    print("-" * 72)
    print(f"• dataset_config.json : {'✅ Present' if 'dataset_config.json' in final_hf_files else '❌ Missing'}")
    print(f"• 本次跳过已完成分块   : {skipped_chunks}")
    print(f"• 本次新迁移分块       : {migrated_chunks} ({migrated_files} files)")
    print(f"• 数据集访问地址       : {dataset_url}")
    print("=" * 72)

    return {
        "hf_repo_id": hf_repo_id,
        "dataset_url": dataset_url,
        "total_chunks": total_chunks,
        "skipped_chunks": skipped_chunks,
        "migrated_chunks": migrated_chunks,
        "migrated_files": migrated_files,
        "per_teacher_counts": per_teacher_counts,
        "final_hf_files": sorted(final_hf_files),
    }


def main():
    parser = get_migration_parser()
    args = parser.parse_args()
    run_migration(args)


if __name__ == "__main__":
    main()
