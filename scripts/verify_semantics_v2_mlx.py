import mlx.core as mx
import numpy as np
import os
import sys
import json
import argparse
import random
import glob
import torch
import torch.nn.functional as F

# Ensure workspace root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from training.core.dataloader import ChunkedNpzDataLoader
from training.core.char_tokenizer import CharTokenizer
from model.config import ModelConfig
from model.adapter import SensoryFuser
from model.god_encoder import GodEncoder
from model.decoder import WeakDecoder

class EmbeddingExtractor:
    def __init__(self, model_key, cache_dir, explicit_model_dir=None):
        from transformers import AutoTokenizer, AutoModel
        from sentence_transformers import SentenceTransformer
        
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
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
        elif model_key == "conan_v1" or model_key == "conan":
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
            st_kwargs = {"trust_remote_code": True}
            self.model = SentenceTransformer(model_source, device=str(self.device), cache_folder=cache_dir, **st_kwargs)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_source, cache_dir=cache_dir, trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            model_kwargs = {
                "cache_dir": cache_dir,
                "torch_dtype": torch.float16,
                "trust_remote_code": True,
            }
            self.model = AutoModel.from_pretrained(model_source, **model_kwargs).to(self.device)
            self.model.eval()

    def get_last_token_pooling(self, last_hidden_states, attention_mask):
        seq_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.size(0)
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), seq_lengths, :]

    def encode(self, texts):
        if self.engine == "auto":
            inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                last_hidden = outputs.last_hidden_state
                if self.model_key == "bge":
                    embs = last_hidden[:, 0, :]
                    embs = torch.nn.functional.normalize(embs, p=2, dim=1)
                elif self.model_key in ["qwen", "gte_qwen"]:
                    embs = self.get_last_token_pooling(last_hidden, inputs.attention_mask)
                    embs = torch.nn.functional.normalize(embs, p=2, dim=1)
                numpy_embs = embs.cpu().to(torch.float16).numpy()
        else:
            numpy_embs = self.model.encode(texts, batch_size=4, convert_to_numpy=True, normalize_embeddings=False, show_progress_bar=False).astype(np.float16)
        
        return numpy_embs

def compute_cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    e1 = torch.tensor(emb1, dtype=torch.float32).unsqueeze(0)
    e2 = torch.tensor(emb2, dtype=torch.float32).unsqueeze(0)
    similarity = F.cosine_similarity(e1, e2).item()
    return similarity

def process_sentence(target_text, embs_list, fuser, god_encoder, decoder, tokenizer, weights, judge_extractor, sample_idx):
    print(f"\n======================================")
    print(f"       >>> LIVE SAMPLE {sample_idx} <<<")
    print(f"======================================")
    
    # embs_list is a list of numpy arrays, convert to mlx
    mlx_embs = [mx.array(arr) for arr in embs_list]
    
    f_t = fuser(mlx_embs, weights=weights)
    z_target = god_encoder(f_t)
    
    start_token = tokenizer.bos_token_id
    eos_token = tokenizer.eos_token_id
    
    generated_ids = decoder.generate(z_target, start_token=start_token, eos_token=eos_token, max_tokens=100, temperature=0.2)
    decoded_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    print(f"[1] Original Text : {target_text}")
    print(f"[2] Dreamed Text  : {decoded_text}\n")
    
    if not decoded_text.strip():
        print("   [!] Model hallucinated complete silence. Similarity: 0.000")
        return
        
    # Judge
    judge_orig_emb = judge_extractor.encode([target_text])[0]
    judge_dream_emb = judge_extractor.encode([decoded_text])[0]
    
    similarity = compute_cosine_similarity(judge_orig_emb, judge_dream_emb)
    print(f"   >>> [Semantic Cosine Similarity]: {similarity:.4f} / 1.000 <<<")
    
    if similarity > 0.8:
        print("   (Verdict: Outstanding Concept Preservation!)")
    elif similarity > 0.6:
        print("   (Verdict: Core Concepts Recognized but Syntax Tangled)")
    else:
        print("   (Verdict: Semantic Collapse)")

def verify():
    parser = argparse.ArgumentParser(description="Phase 0 V2 Semantic Verification Script")
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--ckpt", type=str, default="checkpoints/run/p0_v2_step_120000")
    parser.add_argument("--data_dir", type=str, default="./embs/catclever/emb_npy")
    parser.add_argument("--config_file", type=str, default="dataset_config.json")
    parser.add_argument("--hf_cache_dir", type=str, default=os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface/hub")))
    parser.add_argument("--model_dir", type=str, default=None)
    
    parser.add_argument("--mode", type=str, choices=["random", "interactive", "file"], default="random")
    parser.add_argument("--file_path", type=str, default="", help="Path to text file containing test sentences (required for 'file' mode)")
    parser.add_argument("--model", type=str, required=True, help="all, or specific model (e.g., xiaobu)")
    parser.add_argument("--judge_model", type=str, default=None, help="Explicit judge model. If --model=all, this is REQUIRED.")
    
    args = parser.parse_args()

    print("1. Loading physical architecture...")
    tokenizer = CharTokenizer("training/core/char_vocab.json")
    
    with open(args.config_file, "r") as f:
        ds_config = json.load(f)
        
    models = ds_config["base_models"]
    emb_dims = ds_config["emb_dims"]
    chunk_patterns = ds_config["chunk_name_patterns"]
    parquet_path = ds_config["parquet_path"]
    
    if args.model == "all":
        if not args.judge_model:
            raise ValueError("FATAL: If --model is 'all', you must explicitly specify a --judge_model (e.g., --judge_model bge).")
        target_model = "all"
        target_models = models
        target_model_idx = -1
        fuser_weights = None
        judge_model_key = args.judge_model
    else:
        if args.model not in models:
            raise ValueError(f"FATAL: Model '{args.model}' is not supported. Supported: {models}")
        target_model = args.model
        target_models = [target_model]
        target_model_idx = models.index(target_model)
        fuser_weights = [0.0] * len(models)
        fuser_weights[target_model_idx] = 1.0
        # Single model scores itself, unless explicitly overridden
        judge_model_key = args.judge_model if args.judge_model else target_model

    config = ModelConfig()
    d_model = config.d_model
    vocab_size = tokenizer.vocab_size
    
    fuser = SensoryFuser(emb_dims, d_model)
    god_encoder = GodEncoder(d_model, config.z_dim)
    decoder = WeakDecoder(config.z_dim, vocab_size, d_model=256, n_layers=4)
    
    ckpt_path = args.ckpt
    print(f"2. Loading weights from {ckpt_path}...")
    
    fuser_path = f"{ckpt_path}/sense_fuser.safetensors"
    if not os.path.exists(fuser_path):
        fuser_path = f"{ckpt_path}/sense_adapter.safetensors"
    fuser.load_weights(fuser_path)
    god_encoder.load_weights(f"{ckpt_path}/god_encoder.safetensors")
    
    decoder_path = f"{ckpt_path}/weak_decoder.safetensors"
    if not os.path.exists(decoder_path):
        decoder_path = f"{ckpt_path}/decoder.safetensors"
    decoder.load_weights(decoder_path)
    
    print("\n======================================")
    print(f"Loading Semantic Judge ({judge_model_key})...")
    judge_extractor = EmbeddingExtractor(judge_model_key, cache_dir=args.hf_cache_dir, explicit_model_dir=args.model_dir)
    print("Judge Online. Ready to evaluate semantic preservation.")
    print("======================================\n")

    if args.mode == "interactive":
        # We need an extractor for the input text if it's single model
        # Wait, if mode is interactive, we need ALL extractors if model == "all".
        # Loading 5 large models into Mac memory is impossible.
        if target_model == "all":
            raise SystemExit("Interactive mode with --model 'all' requires loading 5 large models into memory simultaneously, which will OOM. Please use --mode random for 'all'.")
        
        print("\n=== [ON-THE-FLY INTERACTIVE MODE] ===")
        # Usually judge and input model are the same here, but we can reuse judge_extractor if they match to save RAM.
        if judge_model_key == target_model:
            input_extractor = judge_extractor
        else:
            print(f"Loading Input Model ({target_model})...")
            input_extractor = EmbeddingExtractor(target_model, cache_dir=args.hf_cache_dir, explicit_model_dir=args.model_dir)

        print("\nEntering Interactive Mode. Type 'quit' or 'exit' to terminate.")
        i = 1
        while True:
            try:
                user_text = input(f"\n[You] Please enter test text (Sample {i}): ")
                if user_text.strip().lower() in ['quit', 'exit', 'q']:
                    break
                if not user_text.strip():
                    continue
                
                emb_tensor = input_extractor.encode([user_text])
                embs_list = []
                for j, dim in enumerate(emb_dims):
                    if j == target_model_idx:
                        embs_list.append(emb_tensor)
                    else:
                        embs_list.append(np.zeros((1, dim), dtype=np.float16))
                        
                process_sentence(user_text, embs_list, fuser, god_encoder, decoder, tokenizer, fuser_weights, judge_extractor, i)
                i += 1
            except KeyboardInterrupt:
                break
                
    elif args.mode == "random":
        print("\n=== [NPZ CHUNK MODE] ===")
        pattern = chunk_patterns.get(target_models[0], "")
        prefix = pattern.split("/")[0] if "/" in pattern else target_models[0]
        
        local_files = []
        if os.path.exists(args.data_dir):
            all_npz = glob.glob(os.path.join(args.data_dir, "**", "*.npz"), recursive=True)
            for f in all_npz:
                f_normalized = f.replace("\\", "/")
                if f_normalized.startswith(prefix + "/") or f"/{prefix}/" in f_normalized:
                    local_files.append(f)
                    
        ms_repo_id = None
        if len(local_files) == 0:
            print(f"[Warning] No local `.npz` chunks found for '{target_models[0]}' in '{args.data_dir}'. Falling back to Cloud Download.")
            ms_repo_id = ds_config["modelscope_repo_id"]
        else:
            print(f"Found local `.npz` chunks. Enforcing Local Mode.")

        print("Initializing ChunkedNpzDataLoader...")
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
            backend='numpy'
        )
        
        print(f"\nPulling {args.num_samples} samples from dataloader...")
        iterator = iter(dataloader)
        token_inputs, active_embs, masks = next(iterator)
        
        actual_batch_size = token_inputs.shape[0]
        
        for i in range(args.num_samples):
            if i >= actual_batch_size:
                break
                
            target_tokens = token_inputs[i].tolist()
            target_tokens = [t for t in target_tokens if t != tokenizer.pad_token_id]
            target_text = tokenizer.decode(target_tokens, skip_special_tokens=True)
            
            embs_list = []
            for j, dim in enumerate(emb_dims):
                if target_model == "all":
                    embs_list.append(active_embs[j][i:i+1])
                else:
                    if j == target_model_idx:
                        embs_list.append(active_embs[0][i:i+1])
                    else:
                        embs_list.append(np.zeros((1, dim), dtype=np.float16))
                        
            process_sentence(target_text, embs_list, fuser, god_encoder, decoder, tokenizer, fuser_weights, judge_extractor, i+1)

    elif args.mode == "file":
        if not args.file_path:
            raise ValueError("[ERROR] --file_path must be provided when using --mode file")
            
        with open(args.file_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
            
        if target_model == "all":
            raise SystemExit("File mode with --model 'all' requires loading 5 large models into memory simultaneously, which will OOM. Please use a single model for file evaluation on Mac.")
            
        print(f"Loading Input Model ({target_model})...")
        input_extractor = EmbeddingExtractor(target_model, cache_dir=args.hf_cache_dir, explicit_model_dir=args.model_dir)

        print(f"\nProcessing up to {args.num_samples} lines from {args.file_path}...")
        for i, target_text in enumerate(lines[:args.num_samples]):
            emb_tensor = input_extractor.encode([target_text])
            embs_list = []
            for j, dim in enumerate(emb_dims):
                if j == target_model_idx:
                    embs_list.append(emb_tensor)
                else:
                    embs_list.append(np.zeros((1, dim), dtype=np.float16))
            process_sentence(target_text, embs_list, fuser, god_encoder, decoder, tokenizer, fuser_weights, judge_extractor, i+1)

if __name__ == "__main__":
    verify()
