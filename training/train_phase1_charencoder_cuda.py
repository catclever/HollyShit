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
    has_chunked_conf = bool(ds_config.get("base_models") and ds_config.get("chunk_name_patterns"))
    config_base_dir = os.path.dirname(os.path.abspath(args.config_file)) if args.config_file else os.getcwd()
    parquet_path = args.parquet_path or ds_config.get("parquet_path", "data/Basic_ZH/chunked_mixed_wiki.parquet")
    parquet_path = resolve_path_maybe_relative(parquet_path, config_base_dir)

    if has_chunked_conf:
        models = ds_config["base_models"]
        chunk_patterns = ds_config["chunk_name_patterns"]
        ms_repo_id = ds_config.get("modelscope_repo_id")
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
        print(f"[Data] Using Phase1_ChunkedNpzDataLoader ({len(models)} routes, dynamic ModelScope/local chunks).")
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
    _cand_locks = {}
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
            if loader.ms_repo_id:
                from modelscope.hub.file_download import dataset_file_download
                last_err = None
                downloaded_path = None
                for _ in range(3):
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
                    raise FileNotFoundError(f"本地目标块缺失：{downloaded_path}")

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
    ckpt_dir = expected_checkpoint_dir(out_dir, prefix, step, is_emergency=is_emergency)
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
    ckpt_dir = expected_checkpoint_dir(out_dir, prefix, step, is_emergency=is_emergency)
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
    return parser


def main():
    parser = get_distill_parser()
    args = parser.parse_args()

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
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

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

                with torch.autocast(device_type="cuda", enabled=use_amp):
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
                    checkpointer.save(global_step)
                    save_torch_optimizer_state(optimizer, args.out_dir, checkpointer.prefix, global_step)
                    save_char_subspace_stats(student, args.out_dir, checkpointer.prefix, global_step)

    except KeyboardInterrupt:
        print(f"\n[Interrupt] Caught kill signal at step {global_step}! Saving emergency checkpoint...")
        checkpointer.save(global_step, is_emergency=True)
        save_torch_optimizer_state(optimizer, args.out_dir, checkpointer.prefix, global_step, is_emergency=True)
        save_char_subspace_stats(student, args.out_dir, checkpointer.prefix, global_step, is_emergency=True)


if __name__ == "__main__":
    main()
