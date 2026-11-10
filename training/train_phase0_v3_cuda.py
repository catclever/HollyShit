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
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from safetensors.torch import load_file as load_safetensors_file
from safetensors.torch import save_file as save_safetensors_file

# Ensure workspace root is at the front of sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir in sys.path and os.path.exists(os.path.join(script_dir, "model.py")):
    sys.path = [p for p in sys.path if p != script_dir] + [script_dir]

from model.adapter_cuda import SensoryFuserCUDA
from model.god_encoder_cuda import (
    AsymmetricNextClauseHeadCUDA,
    GodEncoderCUDA,
    MultiViewReconstructionHeadsCUDA,
)
from training.core.char_tokenizer import CharTokenizer
from training.data.alignment_dataloader import (
    Phase1_ChunkedNpzDataLoader,
    Phase1_MultiEmbDataLoader,
)
from training.losses.god_manifold_loss import (
    GodManifoldLossWeights,
    ManifoldFIFOQueue,
    compute_god_manifold_loss,
    evaluate_god_manifold_health,
)


def load_optional_config(config_file: str) -> dict:
    if not config_file:
        return {}
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found: {config_file}")
    with open(config_file, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_path_maybe_relative(path_value: str, base_dir: str) -> str:
    if not path_value:
        return path_value
    if os.path.isabs(path_value):
        return path_value
    candidate = os.path.join(base_dir, path_value)
    if os.path.exists(candidate):
        return candidate
    return path_value


def expected_checkpoint_dir(
    out_dir: str,
    prefix: str,
    step: int,
    is_emergency: bool = False,
) -> str:
    base_name = "latest_emergency" if is_emergency else f"step_{step}"
    folder_name = f"{prefix}_{base_name}" if prefix else base_name
    return os.path.join(out_dir, folder_name)


def load_mlx_safetensors_into_torch(torch_module: nn.Module, safetensor_path: str):
    """
    Loads an MLX/PyTorch safetensors file into a PyTorch nn.Module without requiring mlx installed.
    """
    if not os.path.exists(safetensor_path):
        raise FileNotFoundError(f"Safetensors file not found at {safetensor_path}")

    state_dict = load_safetensors_file(safetensor_path)
    target_keys = set(torch_module.state_dict().keys())

    remapped = {}
    for k, v in state_dict.items():
        new_k = (
            k.replace(".wq.", ".query_proj.")
            .replace(".wk.", ".key_proj.")
            .replace(".wv.", ".value_proj.")
            .replace(".wo.", ".out_proj.")
        )
        if new_k not in target_keys:
            test_k1 = (
                new_k.replace(".net.layers.0.", ".fc1.")
                .replace(".net.layers.2.", ".fc2.")
                .replace("net.layers.0.", "fc1.")
                .replace("net.layers.2.", "fc2.")
            )
            test_k2 = new_k.replace(".net.layers.0.", ".linear1.").replace(
                ".net.layers.2.", ".linear2."
            )
            if test_k1 in target_keys:
                new_k = test_k1
            elif test_k2 in target_keys:
                new_k = test_k2
        remapped[new_k] = v

    torch_module.load_state_dict(remapped, strict=False)


def infer_teacher_arch_from_ckpt(p0_ckpt: str) -> Tuple[str, str, List[int], int, int]:
    """
    Infers teacher route count, embedding dimensions, d_model, and z_dim directly from a
    Phase 0 checkpoint directory containing sense_fuser.safetensors and god_encoder.safetensors.
    """
    fuser_path = os.path.join(p0_ckpt, "sense_fuser.safetensors")
    if not os.path.exists(fuser_path):
        fuser_path = os.path.join(p0_ckpt, "sense_adapter.safetensors")
    if not os.path.exists(fuser_path):
        raise FileNotFoundError(f"Missing sense_fuser/sense_adapter checkpoint under: {p0_ckpt}")

    god_path = os.path.join(p0_ckpt, "god_encoder.safetensors")
    if not os.path.exists(god_path):
        raise FileNotFoundError(f"Missing god_encoder.safetensors under: {p0_ckpt}")

    fuser_state = load_safetensors_file(fuser_path)
    god_state = load_safetensors_file(god_path)

    route_dims: Dict[int, int] = {}
    d_model: Optional[int] = None
    for key, tensor in fuser_state.items():
        m = re.match(r"adapters\.(\d+)\.(?:net\.layers\.0|fc1)\.weight$", key)
        if m:
            idx = int(m.group(1))
            route_dims[idx] = int(tensor.shape[1])
            d_model = int(tensor.shape[0])

    if not route_dims:
        raise RuntimeError(f"Could not infer adapter routes from {fuser_path}")

    emb_dims = [route_dims[i] for i in sorted(route_dims.keys())]

    z_dim: Optional[int] = None
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
        except (zipfile.BadZipFile, OSError, ValueError) as e:
            last_err = e
            attempt += 1
            if attempt > max_retries:
                break
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError:
                pass
            if cand_name and hasattr(loader, "_resolve_npz_path"):
                path = loader._resolve_npz_path(cand_name)
            elif cand_name and getattr(loader, "ms_repo_id", None):
                from modelscope.hub.file_download import dataset_file_download

                path = dataset_file_download(
                    loader.ms_repo_id, cand_name, cache_dir=loader.cache_dir
                )
            else:
                raise
    raise RuntimeError(f"Failed to load npz after retries: {path} | last_error={last_err}")


def attach_chunk_download_router(loader, local_data_dir: str, download_cache_dir: str):
    if not hasattr(loader, "models") or not hasattr(loader, "chunk_patterns"):
        return loader

    local_data_dir = os.path.abspath(local_data_dir)
    download_cache_dir = os.path.abspath(download_cache_dir)
    loader.npz_dir = local_data_dir
    loader.cache_dir = download_cache_dir
    _cand_locks: Dict[str, threading.Lock] = {}
    _cand_locks_guard = threading.Lock()

    def _get_cand_lock(cand_name: str) -> threading.Lock:
        with _cand_locks_guard:
            if cand_name not in _cand_locks:
                _cand_locks[cand_name] = threading.Lock()
            return _cand_locks[cand_name]

    def _resolve_npz_path(cand_name: str) -> str:
        local_path = os.path.join(local_data_dir, cand_name)
        cand_lock = _get_cand_lock(cand_name)
        with cand_lock:
            if os.path.exists(local_path):
                return local_path
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            if loader.ms_repo_id:
                from modelscope.hub.file_download import dataset_file_download

                last_err = None
                downloaded_path = None
                for _retry in range(3):
                    try:
                        downloaded_path = dataset_file_download(
                            loader.ms_repo_id, cand_name, cache_dir=download_cache_dir
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
                    raise FileNotFoundError(f"Missing local npz chunk: {downloaded_path}")

            if downloaded_path != os.path.abspath(local_path):
                try:
                    os.replace(downloaded_path, local_path)
                except OSError:
                    shutil.copy2(downloaded_path, local_path)
                    try:
                        os.remove(downloaded_path)
                    except OSError:
                        pass
            return local_path

    loader._resolve_npz_path = _resolve_npz_path
    return loader


def disable_chunk_prefetch(loader):
    if not hasattr(loader, "chunk_bounds") or not hasattr(loader, "macro_chunk_indices"):
        return loader

    try:
        if hasattr(loader, "stop_event"):
            loader.stop_event.set()
        if hasattr(loader, "prefetch_thread") and loader.prefetch_thread.is_alive():
            loader.prefetch_thread.join(timeout=1.0)
    except Exception:
        pass

    def _load_payload_sync(macro_ptr):
        if macro_ptr >= len(loader.macro_chunk_indices):
            return None
        chunk_idx = loader.macro_chunk_indices[macro_ptr]
        start_idx, end_idx = loader.chunk_bounds[chunk_idx]
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
            active_chunk_embs.append(arr)
            cached_paths.append(path)

        cur_chunk_size = end_idx - start_idx
        active_micro_indices = np.arange(cur_chunk_size)
        if loader.shuffle:
            rng = np.random.default_rng(loader.seed + macro_ptr + loader.current_epoch)
            rng.shuffle(active_micro_indices)
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

    loader._start_prefetching = _start_prefetching_sync
    loader._pop_next_chunk = _pop_next_chunk_sync
    return loader


def enable_download_only_prefetch(loader):
    return disable_chunk_prefetch(loader)


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


def sample_contiguous_triplets_from_loader(
    loader,
    num_triplets: int,
    device: torch.device,
    rng: Optional[np.random.Generator] = None,
) -> Optional[Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]]:
    """
    Samples `num_triplets` contiguous clause triplets (i, i+1, i+2) directly from the active
    chunk buffer (`loader.active_chunk_embs`) or memory-mapped embeddings (`loader.embs`).
    """
    if num_triplets <= 0:
        return None

    source_arrays = None
    if getattr(loader, "active_chunk_embs", None) is not None:
        source_arrays = loader.active_chunk_embs
    elif getattr(loader, "embs", None) is not None:
        source_arrays = loader.embs

    if not source_arrays or len(source_arrays) == 0:
        return None

    n_avail = int(source_arrays[0].shape[0])
    if n_avail < 3:
        return None

    gen = rng if rng is not None else np.random.default_rng()
    k = min(int(num_triplets), n_avail - 2)
    start_indices = gen.integers(0, n_avail - 2, size=k)

    embs_prev: List[torch.Tensor] = []
    embs_curr: List[torch.Tensor] = []
    embs_next: List[torch.Tensor] = []

    for arr in source_arrays:
        e0_np = np.asarray(arr[start_indices], dtype=np.float32).copy()
        e1_np = np.asarray(arr[start_indices + 1], dtype=np.float32).copy()
        e2_np = np.asarray(arr[start_indices + 2], dtype=np.float32).copy()
        embs_prev.append(torch.from_numpy(e0_np).to(device))
        embs_curr.append(torch.from_numpy(e1_np).to(device))
        embs_next.append(torch.from_numpy(e2_np).to(device))

    return embs_prev, embs_curr, embs_next


def save_phase0_v3_checkpoint(
    out_dir: str,
    prefix: str,
    step: int,
    fuser: SensoryFuserCUDA,
    god_encoder: GodEncoderCUDA,
    recon_heads: Optional[MultiViewReconstructionHeadsCUDA] = None,
    next_clause_head: Optional[AsymmetricNextClauseHeadCUDA] = None,
    optimizer: Optional[optim.Optimizer] = None,
    dataloader=None,
    args: Optional[argparse.Namespace] = None,
    z_mean_stat: Optional[torch.Tensor] = None,
    z_std_stat: Optional[torch.Tensor] = None,
    health_metrics: Optional[Dict[str, float]] = None,
    is_emergency: bool = False,
) -> str:
    """
    Saves Phase 0 v3 checkpoint in both PyTorch (.pt) and MLX-compatible (.safetensors) formats:
      - sense_fuser.safetensors & sense_fuser.pt
      - god_encoder.safetensors & god_encoder.pt
      - recon_heads.pt & next_clause_head.pt
      - manifold_stats.pt (z_mean, z_std, health_metrics, architecture metadata)
      - optimizer.pt, dataloader.json, training_args.json
    """
    ckpt_dir = expected_checkpoint_dir(out_dir, prefix, step, is_emergency=is_emergency)
    os.makedirs(ckpt_dir, exist_ok=True)

    fuser_cpu_state = {k: v.detach().cpu().contiguous() for k, v in fuser.state_dict().items()}
    god_cpu_state = {k: v.detach().cpu().contiguous() for k, v in god_encoder.state_dict().items()}

    # 1. Save MLX/PyTorch dual-compatible safetensors (used by Phase 1 infer_teacher_arch_from_ckpt)
    save_safetensors_file(fuser_cpu_state, os.path.join(ckpt_dir, "sense_fuser.safetensors"))
    save_safetensors_file(god_cpu_state, os.path.join(ckpt_dir, "god_encoder.safetensors"))

    # 2. Save PyTorch .pt state_dicts
    torch.save(fuser_cpu_state, os.path.join(ckpt_dir, "sense_fuser.pt"))
    torch.save(god_cpu_state, os.path.join(ckpt_dir, "god_encoder.pt"))

    if recon_heads is not None:
        torch.save(recon_heads.state_dict(), os.path.join(ckpt_dir, "recon_heads.pt"))
    if next_clause_head is not None:
        torch.save(next_clause_head.state_dict(), os.path.join(ckpt_dir, "next_clause_head.pt"))
    if optimizer is not None:
        torch.save(optimizer.state_dict(), os.path.join(ckpt_dir, "optimizer.pt"))

    if dataloader is not None and hasattr(dataloader, "state_dict"):
        dl_state = dataloader.state_dict()
        dl_state["global_step"] = int(step)
        with open(os.path.join(ckpt_dir, "dataloader.json"), "w", encoding="utf-8") as f:
            json.dump(dl_state, f)

    if args is not None:
        with open(os.path.join(ckpt_dir, "training_args.json"), "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=4, ensure_ascii=False)

    manifold_payload = {
        "global_step": int(step),
        "emb_dims": list(fuser.emb_dims),
        "d_model": int(fuser.d_model),
        "z_dim": int(god_encoder.z_dim),
        "sphere_scale": bool(god_encoder.sphere_scale),
        "z_mean": (
            z_mean_stat.detach().cpu()
            if z_mean_stat is not None
            else torch.zeros(god_encoder.z_dim, dtype=torch.float32)
        ),
        "z_std": (
            z_std_stat.detach().cpu()
            if z_std_stat is not None
            else torch.ones(god_encoder.z_dim, dtype=torch.float32)
        ),
        "health_metrics": dict(health_metrics) if health_metrics is not None else {},
    }
    torch.save(manifold_payload, os.path.join(ckpt_dir, "manifold_stats.pt"))
    return ckpt_dir


def build_phase0_v3_dataloader(args, tokenizer, ds_config: dict):
    has_chunked_conf = bool(ds_config.get("base_models") and ds_config.get("chunk_name_patterns"))
    config_base_dir = (
        os.path.dirname(os.path.abspath(args.config_file)) if args.config_file else os.getcwd()
    )
    parquet_path = args.parquet_path or ds_config.get(
        "parquet_path", "data/Basic_ZH/chunked_mixed_omni.parquet"
    )
    parquet_path = resolve_path_maybe_relative(parquet_path, config_base_dir)

    if has_chunked_conf:
        models = ds_config["base_models"]
        chunk_patterns = ds_config["chunk_name_patterns"]
        ms_repo_id = ds_config.get("modelscope_repo_id")
        print(
            f"[Data] Using Phase1_ChunkedNpzDataLoader ({len(models)} teacher routes: {models})."
        )
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
        )

    emb_paths = args.emb_paths or ds_config.get("emb_paths")
    if not emb_paths:
        emb_paths = sorted(glob.glob(os.path.join(args.emb_dir, "*embeddings*.npy")))
    if not emb_paths:
        raise FileNotFoundError(
            "No teacher embedding files found; provide --config_file or --emb_paths."
        )
    print(f"[Data] Using Phase1_MultiEmbDataLoader ({len(emb_paths)} routes): {emb_paths}")
    return Phase1_MultiEmbDataLoader(
        parquet_path=parquet_path,
        emb_paths=emb_paths,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        backend="torch",
    )


def get_phase0_v3_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phase 0 v3: Cavity-Free Isotropic 5-Teacher GodEncoder Manifold Distillation (PyTorch CUDA/MPS/CPU)"
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--triplet_batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--total_steps", type=int, default=100000)
    parser.add_argument("--d_model", type=int, default=1024)
    parser.add_argument("--z_dim", type=int, default=1024)
    parser.add_argument(
        "--sphere_scale",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Project z_god onto the hypersphere of radius sqrt(z_dim)",
    )
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--save_steps", type=int, default=5000)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--log_steps", type=int, default=20)
    parser.add_argument("--out_dir", type=str, default="checkpoints/p0_v3")
    parser.add_argument("--ckpt_prefix", type=str, default="p0_v3_5models")
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--config_file", type=str, default="dataset_config.json")
    parser.add_argument("--emb_paths", nargs="*", default=None)
    parser.add_argument("--emb_dir", type=str, default="data/Basic_ZH/embs/hy-tmp")
    parser.add_argument("--parquet_path", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default="./embs")
    parser.add_argument("--download_cache_dir", type=str, default=None)
    parser.add_argument("--chunk_size", type=int, default=500000)
    parser.add_argument("--prefetch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--download_only_prefetch", action="store_true")

    # Loss weights
    parser.add_argument("--w_recon", type=float, default=2.0)
    parser.add_argument("--w_rel", type=float, default=1.5)
    parser.add_argument("--w_cloud", type=float, default=1.0)
    parser.add_argument("--w_iso", type=float, default=1.0)
    parser.add_argument("--w_unif", type=float, default=0.5)
    parser.add_argument("--w_momentum", type=float, default=0.3)
    parser.add_argument("--w_next", type=float, default=0.5)
    parser.add_argument("--dirichlet_alpha", type=float, default=1.0)
    parser.add_argument("--modality_drop_prob", type=float, default=0.15)
    return parser


def main():
    parser = get_phase0_v3_parser()
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("\n🔥 [Hardware] CUDA detected. Booting Phase 0 v3 Isotropic God Manifold Engine...")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("\n🍏 [Hardware] Apple Silicon MPS detected. Booting Phase 0 v3 Engine...")
    else:
        device = torch.device("cpu")
        print("\n🐢 [Hardware] CPU detected. Booting Phase 0 v3 Engine...")

    ds_config = load_optional_config(args.config_file) if args.config_file else {}
    if args.download_cache_dir is None:
        args.download_cache_dir = args.data_dir

    emb_dims = ds_config.get("emb_dims", [1024, 3584, 1792, 2048, 1792])
    tokenizer = CharTokenizer()

    dataloader = build_phase0_v3_dataloader(args, tokenizer, ds_config)
    dataloader = attach_chunk_download_router(dataloader, args.data_dir, args.download_cache_dir)
    if args.download_only_prefetch:
        dataloader = enable_download_only_prefetch(dataloader)
    elif not args.prefetch:
        dataloader = disable_chunk_prefetch(dataloader)

    fuser = SensoryFuserCUDA(emb_dims=emb_dims, d_model=args.d_model).to(device)
    god_encoder = GodEncoderCUDA(
        d_model=args.d_model, z_dim=args.z_dim, sphere_scale=args.sphere_scale
    ).to(device)
    recon_heads = MultiViewReconstructionHeadsCUDA(z_dim=args.z_dim, emb_dims=emb_dims).to(device)
    next_clause_head = AsymmetricNextClauseHeadCUDA(z_dim=args.z_dim).to(device)

    all_params = (
        list(fuser.parameters())
        + list(god_encoder.parameters())
        + list(recon_heads.parameters())
        + list(next_clause_head.parameters())
    )
    optimizer = optim.AdamW(all_params, lr=args.lr, weight_decay=args.weight_decay)
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    loss_weights = GodManifoldLossWeights(
        w_recon=args.w_recon,
        w_rel=args.w_rel,
        w_cloud=args.w_cloud,
        w_iso=args.w_iso,
        w_unif=args.w_unif,
        w_momentum=args.w_momentum,
        w_next=args.w_next,
        dirichlet_alpha=args.dirichlet_alpha,
        modality_drop_prob=args.modality_drop_prob,
    )

    global_step = 0
    running_z_mean = torch.zeros(args.z_dim, dtype=torch.float32, device=device)
    running_z_std = torch.ones(args.z_dim, dtype=torch.float32, device=device)
    manifold_queue = ManifoldFIFOQueue(max_size=max(2048, args.z_dim * 2))
    last_health: Dict[str, float] = {}

    if args.resume_from and os.path.isdir(args.resume_from):
        print(f"[Resume] Loading Phase 0 v3 state from {args.resume_from}...")
        fuser.load_state_dict(
            torch.load(os.path.join(args.resume_from, "sense_fuser.pt"), map_location=device)
        )
        god_encoder.load_state_dict(
            torch.load(os.path.join(args.resume_from, "god_encoder.pt"), map_location=device)
        )
        rh_path = os.path.join(args.resume_from, "recon_heads.pt")
        if os.path.exists(rh_path):
            recon_heads.load_state_dict(torch.load(rh_path, map_location=device))
        nh_path = os.path.join(args.resume_from, "next_clause_head.pt")
        if os.path.exists(nh_path):
            next_clause_head.load_state_dict(torch.load(nh_path, map_location=device))
        opt_path = os.path.join(args.resume_from, "optimizer.pt")
        if os.path.exists(opt_path):
            optimizer.load_state_dict(torch.load(opt_path, map_location=device))
        dl_path = os.path.join(args.resume_from, "dataloader.json")
        if os.path.exists(dl_path):
            with open(dl_path, "r", encoding="utf-8") as f:
                dl_state = json.load(f)
            dataloader.load_state_dict(dl_state)
            global_step = int(dl_state.get("global_step", 0))

    rng = np.random.default_rng(42)

    try:
        for epoch in range(dataloader.current_epoch, args.epochs):
            for _token_inputs, batch_embs, _attention_mask in dataloader:
                global_step += 1
                lr = cosine_warmup_lr_schedule(
                    global_step, args.lr, args.warmup_steps, args.total_steps
                )
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

                batch_embs = [emb.to(device, non_blocking=True).float() for emb in batch_embs]
                triplet_embs = sample_contiguous_triplets_from_loader(
                    dataloader, args.triplet_batch_size, device=device, rng=rng
                )

                optimizer.zero_grad(set_to_none=True)

                with torch.autocast(device_type="cuda", enabled=use_amp):
                    loss, metrics, aux = compute_god_manifold_loss(
                        fuser=fuser,
                        god_encoder=god_encoder,
                        recon_heads=recon_heads,
                        teacher_embs=batch_embs,
                        next_clause_head=next_clause_head,
                        triplet_teacher_embs=triplet_embs,
                        weights=loss_weights,
                        manifold_queue=manifold_queue,
                        enqueue_to_queue=(global_step >= max(15, args.warmup_steps // 10)),
                    )

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                with torch.no_grad():
                    z_cur = aux["z_mean"].float()
                    batch_mean = z_cur.mean(dim=0)
                    batch_std = z_cur.std(dim=0, unbiased=True).clamp(min=1e-6)
                    ema = 0.05 if global_step > 1 else 1.0
                    running_z_mean = (1.0 - ema) * running_z_mean + ema * batch_mean
                    running_z_std = (1.0 - ema) * running_z_std + ema * batch_std

                if global_step % args.log_steps == 0:
                    print(
                        f"Epoch {epoch + 1} | Step {global_step} | LR {lr:.2e} | "
                        f"Loss {metrics['total_loss']:.4f} "
                        f"(Recon {metrics['loss_recon']:.3f} [cos={metrics['mean_teacher_recon_cos']:.3f}] | "
                        f"Rel {metrics['loss_rel']:.3f} | Cloud {metrics['loss_cloud']:.3f} | "
                        f"Iso {metrics['loss_iso']:.3f} | Unif {metrics['loss_unif']:.3f} | "
                        f"Mom {metrics['loss_momentum']:.3f} | Next {metrics['loss_next']:.3f})"
                    )

                if global_step % args.eval_steps == 0:
                    z_eval_pool = manifold_queue.get()
                    last_health = evaluate_god_manifold_health(
                        fuser=fuser,
                        god_encoder=god_encoder,
                        recon_heads=recon_heads,
                        teacher_embs=batch_embs,
                        next_clause_head=next_clause_head,
                        triplet_teacher_embs=triplet_embs,
                        z_eval_pool=z_eval_pool,
                        dirichlet_alpha=args.dirichlet_alpha,
                        modality_drop_prob=args.modality_drop_prob,
                    )
                    print(
                        f"  🩺 [Manifold Health @ Step {global_step}] "
                        f"PR={last_health['participation_ratio']:.1f} | "
                        f"ReconCos={last_health['mean_teacher_recon_cos']:.3f} | "
                        f"RelSpearman={last_health['relational_spearman']:.3f} | "
                        f"CloudTop1={last_health['cloud_view_top1_acc']:.3f} | "
                        f"MidpointCos={last_health['midpoint_cavity_recon_cos']:.3f} | "
                        f"NextTop1={last_health['next_clause_top1_acc']:.3f} | "
                        f"AdjAUC={last_health['adjacent_auc']:.3f}"
                    )

                if global_step % args.save_steps == 0:
                    saved_dir = save_phase0_v3_checkpoint(
                        out_dir=args.out_dir,
                        prefix=args.ckpt_prefix,
                        step=global_step,
                        fuser=fuser,
                        god_encoder=god_encoder,
                        recon_heads=recon_heads,
                        next_clause_head=next_clause_head,
                        optimizer=optimizer,
                        dataloader=dataloader,
                        args=args,
                        z_mean_stat=running_z_mean,
                        z_std_stat=running_z_std,
                        health_metrics=last_health,
                    )
                    print(f"[Checkpoint] Saved Phase 0 v3 artifacts to {saved_dir}")

        final_dir = save_phase0_v3_checkpoint(
            out_dir=args.out_dir,
            prefix=args.ckpt_prefix,
            step=global_step,
            fuser=fuser,
            god_encoder=god_encoder,
            recon_heads=recon_heads,
            next_clause_head=next_clause_head,
            optimizer=optimizer,
            dataloader=dataloader,
            args=args,
            z_mean_stat=running_z_mean,
            z_std_stat=running_z_std,
            health_metrics=last_health,
        )
        print(f"🎉 Phase 0 v3 Training Complete! Final checkpoint: {final_dir}")

    except KeyboardInterrupt:
        print(f"\n[Interrupt] Saving emergency Phase 0 v3 checkpoint at step {global_step}...")
        save_phase0_v3_checkpoint(
            out_dir=args.out_dir,
            prefix=args.ckpt_prefix,
            step=global_step,
            fuser=fuser,
            god_encoder=god_encoder,
            recon_heads=recon_heads,
            next_clause_head=next_clause_head,
            optimizer=optimizer,
            dataloader=dataloader,
            args=args,
            z_mean_stat=running_z_mean,
            z_std_stat=running_z_std,
            health_metrics=last_health,
            is_emergency=True,
        )


if __name__ == "__main__":
    main()
