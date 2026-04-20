import argparse
import json
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch


MODEL_ID_BY_KEY = {
    "bge": "BAAI/bge-m3",
    "qwen": "iic/gte_Qwen2-7B-instruct",
    "xiaobu": "lier007/xiaobu-embedding-v2",
    "youtu": "tencent/Youtu-Embedding",
    "conan_v1": "TencentBAC/Conan-embedding-v1",
    "conan": "TencentBAC/Conan-embedding-v1",
}


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_local_model_dir(model_key: str, cache_root: str) -> Optional[str]:
    model_id = MODEL_ID_BY_KEY.get(model_key)
    if not model_id:
        return None

    direct = os.path.join(cache_root, model_id)
    if os.path.isdir(direct) and os.path.isfile(os.path.join(direct, "config.json")):
        return direct

    if "/" in model_id:
        org, name = model_id.split("/", 1)
        snapshot_root = os.path.join(cache_root, f"models--{org}--{name}", "snapshots")
        if os.path.isdir(snapshot_root):
            snapshot_dirs = [
                os.path.join(snapshot_root, d)
                for d in os.listdir(snapshot_root)
                if os.path.isdir(os.path.join(snapshot_root, d))
            ]
            snapshot_dirs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            for path in snapshot_dirs:
                if os.path.isfile(os.path.join(path, "config.json")):
                    return path
    return None


def ensure_losskwargs_compat() -> None:
    import transformers.utils as tf_utils
    if hasattr(tf_utils, "LossKwargs"):
        return
    from typing import TypedDict

    class LossKwargs(TypedDict, total=False):
        pass

    tf_utils.LossKwargs = LossKwargs


def ensure_rope_default_compat() -> None:
    import transformers.modeling_rope_utils as rope_utils
    rope_init = rope_utils.ROPE_INIT_FUNCTIONS
    if "default" not in rope_init:
        def _compute_default_rope_parameters(config, device=None, seq_len=None):
            dim = int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads))
            base = float(getattr(config, "rope_theta", 10000.0))
            inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
            return inv_freq, 1.0

        rope_init["default"] = _compute_default_rope_parameters

    if not hasattr(rope_utils, "dynamic_rope_update"):
        def _dynamic_rope_update_noop(*args, **kwargs):
            return None

        rope_utils.dynamic_rope_update = _dynamic_rope_update_noop


def ensure_pruned_heads_compat() -> None:
    from transformers.configuration_utils import PretrainedConfig
    if not hasattr(PretrainedConfig, "pruned_heads"):
        PretrainedConfig.pruned_heads = {}


def ensure_flash_attention_kwargs_compat() -> None:
    import transformers.modeling_flash_attention_utils as fa_utils
    if hasattr(fa_utils, "FlashAttentionKwargs"):
        return
    from typing import TypedDict

    class FlashAttentionKwargs(TypedDict, total=False):
        pass

    fa_utils.FlashAttentionKwargs = FlashAttentionKwargs


def ensure_attention_registry_compat() -> None:
    import transformers.modeling_utils as modeling_utils
    if not hasattr(modeling_utils, "ALL_ATTENTION_FUNCTIONS"):
        modeling_utils.ALL_ATTENTION_FUNCTIONS = {}


class EmbeddingExtractor:
    def __init__(self, model_key: str, cache_dir: str, explicit_model_dir: Optional[str] = None):
        from sentence_transformers import SentenceTransformer
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        self.device = pick_device()
        self.model_key = model_key

        if model_key in {"bge", "qwen"}:
            self.engine = "auto"
        elif model_key in {"xiaobu", "youtu", "conan_v1", "conan"}:
            self.engine = "st"
        else:
            raise RuntimeError(f"Unsupported model type: {model_key}")

        model_id = MODEL_ID_BY_KEY[model_key]
        model_source = explicit_model_dir if (explicit_model_dir and os.path.isdir(explicit_model_dir)) else model_id
        print(f"[Extractor] model={model_key} source={model_source} device={self.device}")

        if self.engine == "st":
            ensure_losskwargs_compat()
            ensure_rope_default_compat()
            ensure_pruned_heads_compat()
            ensure_flash_attention_kwargs_compat()
            ensure_attention_registry_compat()
            self.model = SentenceTransformer(
                model_source,
                device=str(self.device),
                cache_folder=cache_dir,
                trust_remote_code=True,
            )
        else:
            ensure_losskwargs_compat()
            ensure_rope_default_compat()
            ensure_pruned_heads_compat()

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_source,
                cache_dir=cache_dir,
                trust_remote_code=True,
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            model_kwargs = {
                "cache_dir": cache_dir,
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if self.device.type in {"cuda", "mps"} else torch.float32,
            }
            if self.model_key == "qwen":
                cfg = AutoConfig.from_pretrained(model_source, cache_dir=cache_dir, trust_remote_code=True)
                if not hasattr(cfg, "rope_theta"):
                    setattr(cfg, "rope_theta", 10000.0)
                model_kwargs["config"] = cfg

            self.model = AutoModel.from_pretrained(model_source, **model_kwargs).to(self.device)
            self.model.eval()

    @staticmethod
    def _get_last_token_pooling(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        seq_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.size(0)
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), seq_lengths, :]

    @torch.no_grad()
    def encode(self, texts: List[str], batch_size: int) -> np.ndarray:
        if self.engine == "auto":
            all_embs = []
            for i in range(0, len(texts), batch_size):
                chunk = texts[i:i + batch_size]
                inputs = self.tokenizer(
                    chunk,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(self.device)
                outputs = self.model(**inputs)
                last_hidden = outputs.last_hidden_state
                if self.model_key == "bge":
                    embs = last_hidden[:, 0, :]
                else:
                    embs = self._get_last_token_pooling(last_hidden, inputs.attention_mask)
                embs = torch.nn.functional.normalize(embs, p=2, dim=1)
                all_embs.append(embs.float().cpu().numpy())
            return np.concatenate(all_embs, axis=0)

        arr = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=True,
        ).astype(np.float32)
        return arr


def main():
    parser = argparse.ArgumentParser(description="Embed preset text file with one teacher model.")
    parser.add_argument("--model", required=True, choices=["bge", "qwen", "xiaobu", "youtu", "conan_v1", "conan"])
    parser.add_argument("--texts_file", required=True, help="Preset text file path, one sentence per line")
    parser.add_argument("--output", required=True, help="Output .npy path")
    parser.add_argument("--hf_cache_dir", default="/hy-tmp")
    parser.add_argument("--model_dir", default=None, help="Explicit local model directory")
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    texts = [x.strip() for x in Path(args.texts_file).read_text(encoding="utf-8").splitlines() if x.strip()]
    if not texts:
        raise ValueError("No non-empty lines found in texts_file.")

    model_dir = args.model_dir
    if not model_dir:
        model_dir = resolve_local_model_dir(args.model, args.hf_cache_dir)

    extractor = EmbeddingExtractor(args.model, cache_dir=args.hf_cache_dir, explicit_model_dir=model_dir)
    embs = extractor.encode(texts, batch_size=max(1, args.batch_size))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out), embs)
    meta = out.with_suffix(".meta.json")
    meta.write_text(
        json.dumps(
            {
                "model": args.model,
                "num_texts": len(texts),
                "emb_dim": int(embs.shape[1]),
                "output": str(out),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[Done] saved embeddings: {out} shape={embs.shape}")


if __name__ == "__main__":
    main()
