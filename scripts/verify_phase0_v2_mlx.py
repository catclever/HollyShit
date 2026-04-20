import mlx.core as mx
import numpy as np
import os
import sys
import json
import argparse
import random
import glob

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
        import torch
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
        import torch
        seq_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.size(0)
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), seq_lengths, :]

    def encode(self, texts):
        import torch
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
        
        return mx.array(numpy_embs)

def perform_reconstruction(text, teacher_embs, fuser, god_encoder, decoder, tokenizer, weights=None):
    print(f"\n======================================")
    print(f"5. Forwarding through SensoryFuser and GodEncoder -> z_target...")
    # Wrap each embedding inside a sequence dimension if needed? No, fuser takes list of (B, dim)
    f_t = fuser(teacher_embs, weights=weights)
    z_target = god_encoder(f_t)
    
    print("6. Autoregressive Decoding from z_target (The Topological Dream)...")
    start_token = tokenizer.bos_token_id
    eos_token = tokenizer.eos_token_id
    
    generated_ids = decoder.generate(z_target, start_token=start_token, eos_token=eos_token, max_tokens=100, temperature=0.2)
    decoded_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    print(f"\n=================【终极拓扑解码对比】=================")
    print(f"[真实世界原句 (Original)]:")
    print(f" > {text}\n")
    print(f"[降维压缩后梦境重建 (Dream Reconstruction)]:")
    print(f" > {decoded_text}")
    print(f"======================================================\n")

def verify():
    parser = argparse.ArgumentParser(description="Phase 0 Ultimate Verification Script (Local/Cloud/Interactive)")
    parser.add_argument("--num_samples", type=int, default=3, help="Number of random sentences to pull")
    parser.add_argument("--ckpt", type=str, default="checkpoints/run/p0_v2_step_120000", help="Path to checkpoint directory")
    parser.add_argument("--data_dir", type=str, default="./embs/catclever/emb_npy", help="Local cache directory containing npz files")
    parser.add_argument("--config_file", type=str, default="dataset_config.json", help="Path to 5-model config")
    
    parser.add_argument("--hf_cache_dir", type=str, default=os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface/hub")), help="Default cache directory for HuggingFace models")
    parser.add_argument("--model_dir", type=str, default=None, help="Explicit local path to the embedding model weights")
    
    parser.add_argument("--model", type=str, default=None, help="Force inference via this single model (e.g., xiaobu, bge, qwen)")
    parser.add_argument("--text", type=str, default=None, help="Custom text to reconstruct")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive shell mode")
    
    args = parser.parse_args()

    print("1. Loading physical architecture...")
    tokenizer = CharTokenizer("training/core/char_vocab.json")
    
    with open(args.config_file, "r") as f:
        ds_config = json.load(f)
        
    models = ds_config["base_models"]
    emb_dims = ds_config["emb_dims"]
    chunk_patterns = ds_config["chunk_name_patterns"]
    parquet_path = ds_config["parquet_path"]
    
    # Requirements Check
    if (args.text or args.interactive) and not args.model:
        raise ValueError("FATAL: --text or --interactive requires a specific --model to be provided (e.g. --model xiaobu)")
        
    if args.model and args.model != "all" and args.model not in models:
        raise ValueError(f"FATAL: Model '{args.model}' is not supported by the Fuser configuration. Supported: {models}")

    # Determine Target Model
    target_model = args.model
    if not target_model:
        target_model = random.choice(models)
        print(f"[Auto-Select] No specific --model requested. Randomly selected: {target_model}")
        
    if target_model == "all":
        fuser_weights = None
        target_models = models
        target_model_idx = -1
    else:
        target_model_idx = models.index(target_model)
        fuser_weights = [0.0] * len(models)
        fuser_weights[target_model_idx] = 1.0
        target_models = [target_model]

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
    
    # Branching Logic based on input mode
    if args.text or args.interactive:
        print("\n=== [ON-THE-FLY EMBEDDING MODE] ===")
        extractor = EmbeddingExtractor(target_model, cache_dir=args.hf_cache_dir, explicit_model_dir=args.model_dir)
        
        def run_single_text(text):
            print(f"\nEncoding: {text}")
            emb_tensor = extractor.encode([text])
            # Construct dummy list for the fuser
            embs_list = []
            for i, dim in enumerate(emb_dims):
                if i == target_model_idx:
                    embs_list.append(emb_tensor)
                else:
                    embs_list.append(mx.zeros((1, dim)))
                    
            perform_reconstruction(text, embs_list, fuser, god_encoder, decoder, tokenizer, weights=fuser_weights)

        if args.text:
            run_single_text(args.text)
            
        if args.interactive:
            print("\nEntering Interactive Mode. Type 'quit' or 'exit' to terminate.")
            while True:
                try:
                    user_text = input("\n[Enter Text] > ")
                    if user_text.strip().lower() in ['quit', 'exit']:
                        break
                    if not user_text.strip():
                        continue
                    run_single_text(user_text)
                except KeyboardInterrupt:
                    break
                    
    else:
        print("\n=== [NPZ CHUNK MODE] ===")
        print(f"Targeting '{target_model}' for Validation.")
        
        # Glob check locally for the first target model as heuristic
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
            print(f"[Warning] No local `.npz` chunks found for '{target_models[0]}' in '{args.data_dir}'.")
            print(f"          Falling back to Cloud Download from ModelScope: {ds_config['modelscope_repo_id']}")
            ms_repo_id = ds_config["modelscope_repo_id"]
        else:
            print(f"Found local `.npz` chunks. Enforcing Local Mode.")

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
            backend='mlx'
        )
        
        print(f"\n4. Pulling {args.num_samples} samples from dataloader...")
        iterator = iter(dataloader)
        token_inputs, active_embs, masks = next(iterator)
        
        actual_batch_size = token_inputs.shape[0]
        
        for i in range(args.num_samples):
            if i >= actual_batch_size:
                break
                
            # Recover original text
            target_tokens = token_inputs[i].tolist()
            target_tokens = [t for t in target_tokens if t != tokenizer.pad_token_id]
            target_text = tokenizer.decode(target_tokens, skip_special_tokens=True)
            
            # Construct Fuser List for Sample I
            embs_list = []
            for j, dim in enumerate(emb_dims):
                if target_model == "all":
                    embs_list.append(active_embs[j][i:i+1])
                else:
                    if j == target_model_idx:
                        embs_list.append(active_embs[0][i:i+1])
                    else:
                        embs_list.append(mx.zeros((1, dim)))
                    
            perform_reconstruction(target_text, embs_list, fuser, god_encoder, decoder, tokenizer, weights=fuser_weights)

if __name__ == "__main__":
    verify()
