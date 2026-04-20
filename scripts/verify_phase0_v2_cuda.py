import argparse
import glob
import json
import os
import random
import sys
from typing import List, Optional, Sequence

import numpy as np
import torch

# Ensure workspace root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.core.dataloader import ChunkedNpzDataLoader
from training.core.char_tokenizer import CharTokenizer
from model.config import ModelConfig
from model.adapter_cuda import SensoryFuserCUDA
from model.god_encoder_cuda import GodEncoderCUDA
from model.decoder_cuda import WeakDecoderCUDA
from model.mlx_to_cuda_mapper import load_mlx_safetensors_into_cuda_module


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
    ROPE_INIT_FUNCTIONS = rope_utils.ROPE_INIT_FUNCTIONS

    if "default" in ROPE_INIT_FUNCTIONS:
        pass

    else:
        def _compute_default_rope_parameters(config, device=None, seq_len=None):
            dim = int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads))
            base = float(getattr(config, "rope_theta", 10000.0))
            inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
            attention_factor = 1.0
            return inv_freq, attention_factor

        ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters

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


def to_tensor(value, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype)
    return torch.as_tensor(value, device=device, dtype=dtype)


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


class EmbeddingExtractor:
    def __init__(self, model_key: str, cache_dir: str, explicit_model_dir: Optional[str] = None):
        from sentence_transformers import SentenceTransformer
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        self.device = pick_device()
        self.model_key = model_key

        if model_key == "bge":
            model_id = "BAAI/bge-m3"
            self.engine = "auto"
        elif model_key == "qwen":
            model_id = "iic/gte_Qwen2-7B-instruct"
            self.engine = "auto"
        elif model_key == "xiaobu":
            model_id = "lier007/xiaobu-embedding-v2"
            self.engine = "st"
        elif model_key == "youtu":
            model_id = "tencent/Youtu-Embedding"
            self.engine = "st"
        elif model_key in {"conan_v1", "conan"}:
            model_id = "TencentBAC/Conan-embedding-v1"
            self.engine = "st"
        else:
            raise RuntimeError(f"Unsupported model type: {model_key}")

        if explicit_model_dir and os.path.isdir(explicit_model_dir):
            model_source = explicit_model_dir
            print(f"Loading local model explicitly from {model_source} onto {self.device}...")
        else:
            model_source = model_id
            print(f"Loading {model_id} from HF cache ({cache_dir}) onto {self.device}...")

        if self.engine == "st":
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

            self.model = AutoModel.from_pretrained(
                model_source,
                **model_kwargs,
            ).to(self.device)
            self.model.eval()

    @staticmethod
    def get_last_token_pooling(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        seq_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.size(0)
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), seq_lengths, :]

    @torch.no_grad()
    def encode(self, texts: List[str]) -> torch.Tensor:
        if self.engine == "auto":
            inputs = self.tokenizer(
                texts,
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
                embs = self.get_last_token_pooling(last_hidden, inputs.attention_mask)
            embs = torch.nn.functional.normalize(embs, p=2, dim=1)
            return embs.to(dtype=torch.float32)

        numpy_embs = self.model.encode(
            texts,
            batch_size=4,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        ).astype(np.float32)
        return torch.from_numpy(numpy_embs).to(self.device)


@torch.no_grad()
def perform_reconstruction(
    text: str,
    teacher_embs: List[torch.Tensor],
    fuser: SensoryFuserCUDA,
    god_encoder: GodEncoderCUDA,
    decoder: WeakDecoderCUDA,
    tokenizer: CharTokenizer,
    weights: Optional[Sequence[float]] = None,
):
    print("\n======================================")
    print("5. Forwarding through SensoryFuser and GodEncoder -> z_target...")
    f_t = fuser(teacher_embs, weights=weights)
    z_target = god_encoder(f_t)

    print("6. Autoregressive Decoding from z_target (The Topological Dream)...")
    start_token = tokenizer.bos_token_id
    eos_token = tokenizer.eos_token_id

    generated_ids = decoder.generate(
        z_target,
        start_token=start_token,
        eos_token=eos_token,
        max_tokens=100,
        temperature=0.2,
    )
    decoded_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    print("\n=================【终极拓扑解码对比】=================")
    print("[真实世界原句 (Original)]:")
    print(f" > {text}\n")
    print("[降维压缩后梦境重建 (Dream Reconstruction)]:")
    print(f" > {decoded_text}")
    print("======================================================\n")


def verify():
    parser = argparse.ArgumentParser(description="Phase 0 Ultimate Verification Script (CUDA/PyTorch)")
    parser.add_argument("--num_samples", type=int, default=3, help="Number of random sentences to pull")
    parser.add_argument("--ckpt", type=str, default="checkpoints/run/p0_v2_step_120000", help="Path to checkpoint directory")
    parser.add_argument("--data_dir", type=str, default="./embs/catclever/emb_npy", help="Local cache directory containing npz files")
    parser.add_argument("--config_file", type=str, default="dataset_config.json", help="Path to 5-model config")
    parser.add_argument(
        "--hf_cache_dir",
        type=str,
        default=os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface/hub")),
        help="Default cache directory for HuggingFace models",
    )
    parser.add_argument("--model_dir", type=str, default=None, help="Explicit local path to the embedding model weights")
    parser.add_argument("--model", type=str, default=None, help="Force inference via this single model (e.g., xiaobu, bge, qwen)")
    parser.add_argument("--text", type=str, default=None, help="Custom text to reconstruct")
    parser.add_argument(
        "--exclude_models",
        nargs="*",
        default=[],
        help="Models to exclude when --model all (e.g., --exclude_models youtu)",
    )
    parser.add_argument("--interactive", action="store_true", help="Launch interactive shell mode")
    args = parser.parse_args()

    device = pick_device()
    print(f"0. Runtime device: {device}")
    print("1. Loading physical architecture...")
    tokenizer = CharTokenizer("training/core/char_vocab.json")

    with open(args.config_file, "r", encoding="utf-8") as f:
        ds_config = json.load(f)

    models = ds_config["base_models"]
    emb_dims = ds_config["emb_dims"]
    chunk_patterns = ds_config["chunk_name_patterns"]
    parquet_path = ds_config["parquet_path"]
    model_to_dim = {m: emb_dims[i] for i, m in enumerate(models)}

    excluded_models = set(args.exclude_models or [])
    unknown_excluded = sorted([m for m in excluded_models if m not in models])
    if unknown_excluded:
        raise ValueError(f"FATAL: Unknown --exclude_models: {unknown_excluded}. Supported: {models}")

    if (args.text or args.interactive) and not args.model:
        raise ValueError("FATAL: --text or --interactive requires a specific --model to be provided (e.g. --model xiaobu)")
    if args.model and args.model != "all" and args.model not in models:
        raise ValueError(f"FATAL: Model '{args.model}' is not supported by the Fuser configuration. Supported: {models}")
    if args.model and args.model != "all" and args.model in excluded_models:
        raise ValueError(f"FATAL: --model {args.model} conflicts with --exclude_models.")

    target_model = args.model or random.choice(models)
    if not args.model:
        print(f"[Auto-Select] No specific --model requested. Randomly selected: {target_model}")

    if target_model == "all":
        fuser_weights = None
        target_models = [m for m in models if m not in excluded_models]
        if len(target_models) == 0:
            raise ValueError("FATAL: all models were excluded.")
        target_model_idx = -1
        if excluded_models:
            print(f"[Info] Excluding models in ALL mode: {sorted(excluded_models)}")
    else:
        target_model_idx = models.index(target_model)
        fuser_weights = [0.0] * len(models)
        fuser_weights[target_model_idx] = 1.0
        target_models = [target_model]

    config = ModelConfig()
    d_model = config.d_model
    vocab_size = tokenizer.vocab_size

    fuser = SensoryFuserCUDA(emb_dims, d_model).to(device)
    god_encoder = GodEncoderCUDA(d_model, config.z_dim).to(device)
    decoder = WeakDecoderCUDA(config.z_dim, vocab_size, d_model=256, n_layers=4).to(device)

    ckpt_path = args.ckpt
    print(f"2. Loading weights from {ckpt_path}...")

    fuser_path = f"{ckpt_path}/sense_fuser.safetensors"
    if not os.path.exists(fuser_path):
        fuser_path = f"{ckpt_path}/sense_adapter.safetensors"
    load_mlx_safetensors_into_cuda_module(fuser, fuser_path)

    load_mlx_safetensors_into_cuda_module(god_encoder, f"{ckpt_path}/god_encoder.safetensors")

    decoder_path = f"{ckpt_path}/weak_decoder.safetensors"
    if not os.path.exists(decoder_path):
        decoder_path = f"{ckpt_path}/decoder.safetensors"
    load_mlx_safetensors_into_cuda_module(decoder, decoder_path)

    fuser.eval()
    god_encoder.eval()
    decoder.eval()

    if args.text or args.interactive:
        print("\n=== [ON-THE-FLY EMBEDDING MODE] ===")
        if target_model == "all":
            if args.interactive:
                raise ValueError("FATAL: --interactive with --model all is disabled (too slow/heavy). Use --text mode.")

            def run_single_text(text: str):
                print(f"\nEncoding with all models: {text}")
                embs_list: List[torch.Tensor] = []
                for model_key in models:
                    if model_key in excluded_models:
                        dim = model_to_dim[model_key]
                        embs_list.append(torch.zeros((1, dim), device=device, dtype=torch.float32))
                        continue
                    local_dir = resolve_local_model_dir(model_key, args.hf_cache_dir)
                    extractor = EmbeddingExtractor(model_key, cache_dir=args.hf_cache_dir, explicit_model_dir=local_dir)
                    emb_tensor = extractor.encode([text]).to(device=device, dtype=torch.float32)
                    embs_list.append(emb_tensor)
                    del extractor
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                perform_reconstruction(text, embs_list, fuser, god_encoder, decoder, tokenizer, weights=None)
        else:
            extractor = EmbeddingExtractor(target_model, cache_dir=args.hf_cache_dir, explicit_model_dir=args.model_dir)

            def run_single_text(text: str):
                print(f"\nEncoding: {text}")
                emb_tensor = extractor.encode([text]).to(device=device, dtype=torch.float32)
                embs_list: List[torch.Tensor] = []
                for i, dim in enumerate(emb_dims):
                    if i == target_model_idx:
                        embs_list.append(emb_tensor)
                    else:
                        embs_list.append(torch.zeros((1, dim), device=device, dtype=torch.float32))
                perform_reconstruction(text, embs_list, fuser, god_encoder, decoder, tokenizer, weights=fuser_weights)

        if args.text:
            run_single_text(args.text)

        if args.interactive:
            print("\nEntering Interactive Mode. Type 'quit' or 'exit' to terminate.")
            while True:
                try:
                    user_text = input("\n[Enter Text] > ")
                    if user_text.strip().lower() in ["quit", "exit"]:
                        break
                    if not user_text.strip():
                        continue
                    run_single_text(user_text)
                except KeyboardInterrupt:
                    break
    else:
        print("\n=== [NPZ CHUNK MODE] ===")
        print(f"Targeting '{target_model}' for Validation.")

        pattern = chunk_patterns.get(target_models[0], "")
        prefix = pattern.split("/")[0] if "/" in pattern else target_models[0]

        local_files = []
        if os.path.exists(args.data_dir):
            all_npz = glob.glob(os.path.join(args.data_dir, "**", "*.npz"), recursive=True)
            for file_path in all_npz:
                normalized = file_path.replace("\\", "/")
                if normalized.startswith(prefix + "/") or f"/{prefix}/" in normalized:
                    local_files.append(file_path)

        ms_repo_id = None
        if len(local_files) == 0:
            print(f"[Warning] No local `.npz` chunks found for '{target_models[0]}' in '{args.data_dir}'.")
            print(f"          Falling back to Cloud Download from ModelScope: {ds_config['modelscope_repo_id']}")
            ms_repo_id = ds_config["modelscope_repo_id"]
        else:
            print("Found local `.npz` chunks. Enforcing Local Mode.")

        print("3. Initializing ChunkedNpzDataLoader...")
        dataloader = ChunkedNpzDataLoader(
            parquet_path=parquet_path,
            models=target_models,
            chunk_patterns=chunk_patterns,
            tokenizer=tokenizer,
            ms_repo_id=ms_repo_id,
            local_npz_dir=args.data_dir,
            cache_dir=args.data_dir,
            batch_size=args.num_samples,
            max_seq_len=512,
            shuffle=True,
            lazy_start=False,
            backend="numpy",
        )

        print(f"\n4. Pulling {args.num_samples} samples from dataloader...")
        token_inputs, active_embs, _ = next(iter(dataloader))
        actual_batch_size = int(token_inputs.shape[0])

        for i in range(args.num_samples):
            if i >= actual_batch_size:
                break

            target_tokens = token_inputs[i].tolist()
            target_tokens = [tok for tok in target_tokens if tok != tokenizer.pad_token_id]
            target_text = tokenizer.decode(target_tokens, skip_special_tokens=True)

            embs_list: List[torch.Tensor] = []
            for j, dim in enumerate(emb_dims):
                if target_model == "all":
                    embs_list.append(to_tensor(active_embs[j][i : i + 1], device))
                else:
                    if j == target_model_idx:
                        embs_list.append(to_tensor(active_embs[0][i : i + 1], device))
                    else:
                        embs_list.append(torch.zeros((1, dim), device=device))

            perform_reconstruction(target_text, embs_list, fuser, god_encoder, decoder, tokenizer, weights=fuser_weights)


if __name__ == "__main__":
    verify()
