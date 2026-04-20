import os
from typing import Dict, Tuple

import torch
import torch.nn as nn
from safetensors.torch import load_file


def _key_candidates(mlx_key: str):
    # Primary: same name
    yield mlx_key

    # Compatibility fallback for earlier PyTorch modules that used nn.Sequential indices.
    # e.g. net.layers.0.weight -> net.0.weight
    if ".layers." in mlx_key:
        yield mlx_key.replace(".layers.", ".")


def map_mlx_state_dict_to_cuda(module: nn.Module, mlx_state: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], int]:
    target_state = module.state_dict()
    mapped = {}
    dropped = 0

    for mlx_key, tensor in mlx_state.items():
        matched = False
        for cand in _key_candidates(mlx_key):
            if cand in target_state and target_state[cand].shape == tensor.shape:
                mapped[cand] = tensor.to(dtype=target_state[cand].dtype)
                matched = True
                break
        if not matched:
            dropped += 1

    return mapped, dropped


def load_mlx_safetensors_into_cuda_module(module: nn.Module, ckpt_file: str) -> Dict[str, int]:
    mlx_state = load_file(ckpt_file, device="cpu")
    mapped_state, dropped = map_mlx_state_dict_to_cuda(module, mlx_state)

    missing, unexpected = module.load_state_dict(mapped_state, strict=False)

    stats = {
        "total_ckpt_keys": len(mlx_state),
        "loaded_keys": len(mapped_state),
        "dropped_ckpt_keys": dropped,
        "missing_model_keys": len(missing),
        "unexpected_model_keys": len(unexpected),
    }

    base = os.path.basename(ckpt_file)
    if stats["missing_model_keys"] > 0:
        print(f"[WARN] Missing keys while loading {base}: {stats['missing_model_keys']}")
    if stats["dropped_ckpt_keys"] > 0:
        print(f"[WARN] Unmapped ckpt keys while loading {base}: {stats['dropped_ckpt_keys']}")
    if stats["unexpected_model_keys"] > 0:
        print(f"[WARN] Unexpected keys while loading {base}: {stats['unexpected_model_keys']}")

    return stats
