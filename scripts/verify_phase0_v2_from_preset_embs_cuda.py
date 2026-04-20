import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.core.char_tokenizer import CharTokenizer
from model.config import ModelConfig
from model.adapter_cuda import SensoryFuserCUDA
from model.god_encoder_cuda import GodEncoderCUDA
from model.decoder_cuda import WeakDecoderCUDA
from model.mlx_to_cuda_mapper import load_mlx_safetensors_into_cuda_module


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def reconstruct_one(
    text: str,
    embs_list: List[torch.Tensor],
    fuser: SensoryFuserCUDA,
    god_encoder: GodEncoderCUDA,
    decoder: WeakDecoderCUDA,
    tokenizer: CharTokenizer,
    temperature: float,
) -> str:
    f_t = fuser(embs_list, weights=None)
    z_target = god_encoder(f_t)
    generated_ids = decoder.generate(
        z_target,
        start_token=tokenizer.bos_token_id,
        eos_token=tokenizer.eos_token_id,
        max_tokens=100,
        temperature=temperature,
    )
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(
        description="Phase0 v2 verification from precomputed per-model embeddings (preset texts only)."
    )
    parser.add_argument("--texts_file", required=True, help="Preset text file path (one sentence per line)")
    parser.add_argument("--emb_dir", required=True, help="Directory containing per-model .npy embeddings")
    parser.add_argument("--ckpt", required=True, help="Checkpoint directory containing safetensors")
    parser.add_argument("--config_file", default="/home/HollyShit/dataset_config.json")
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--temperature", type=float, default=0.0, help="Decode temperature (0.0 recommended for stability)")
    parser.add_argument("--max_rows", type=int, default=0, help="Limit number of rows, 0 means all")
    args = parser.parse_args()

    device = pick_device()
    print(f"[Init] device={device}")

    texts = [x.strip() for x in Path(args.texts_file).read_text(encoding="utf-8").splitlines() if x.strip()]
    if args.max_rows > 0:
        texts = texts[: args.max_rows]
    if not texts:
        raise ValueError("No non-empty lines found in texts_file.")

    ds_config = json.loads(Path(args.config_file).read_text(encoding="utf-8"))
    models: List[str] = ds_config["base_models"]
    emb_dims: List[int] = ds_config["emb_dims"]

    emb_arrays: Dict[str, np.ndarray] = {}
    emb_dir = Path(args.emb_dir)
    for model_key in models:
        emb_path = emb_dir / f"{model_key}.npy"
        if not emb_path.exists():
            raise FileNotFoundError(f"Missing embedding file for model '{model_key}': {emb_path}")
        arr = np.load(str(emb_path))
        if arr.shape[0] < len(texts):
            raise ValueError(f"{emb_path} rows({arr.shape[0]}) < texts({len(texts)})")
        emb_arrays[model_key] = arr[: len(texts)]
        print(f"[Emb] {model_key} -> {arr.shape}")

    tokenizer = CharTokenizer("training/core/char_vocab.json")
    cfg = ModelConfig()
    fuser = SensoryFuserCUDA(emb_dims, cfg.d_model).to(device)
    god_encoder = GodEncoderCUDA(cfg.d_model, cfg.z_dim).to(device)
    decoder = WeakDecoderCUDA(cfg.z_dim, tokenizer.vocab_size, d_model=256, n_layers=4).to(device)

    ckpt = Path(args.ckpt)
    fuser_path = ckpt / "sense_fuser.safetensors"
    if not fuser_path.exists():
        fuser_path = ckpt / "sense_adapter.safetensors"
    decoder_path = ckpt / "weak_decoder.safetensors"
    if not decoder_path.exists():
        decoder_path = ckpt / "decoder.safetensors"

    load_mlx_safetensors_into_cuda_module(fuser, str(fuser_path))
    load_mlx_safetensors_into_cuda_module(god_encoder, str(ckpt / "god_encoder.safetensors"))
    load_mlx_safetensors_into_cuda_module(decoder, str(decoder_path))
    fuser.eval()
    god_encoder.eval()
    decoder.eval()

    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for i, text in enumerate(texts):
        embs_list = []
        for model_key in models:
            vec = emb_arrays[model_key][i : i + 1]
            embs_list.append(torch.as_tensor(vec, dtype=torch.float32, device=device))
        recon = reconstruct_one(
            text=text,
            embs_list=embs_list,
            fuser=fuser,
            god_encoder=god_encoder,
            decoder=decoder,
            tokenizer=tokenizer,
            temperature=args.temperature,
        )
        rec = {"idx": i + 1, "text": text, "reconstruction": recon}
        results.append(rec)
        print(f"[{i+1}/{len(texts)}] done")

    with out_path.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[Done] saved: {out_path} rows={len(results)}")


if __name__ == "__main__":
    main()
