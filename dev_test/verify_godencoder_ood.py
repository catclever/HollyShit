import torch
import torch.nn.functional as F
import numpy as np
import mlx.core as mx
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
import os
import sys

# Add parent dir to path to import HollyShit modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.adapter import SensoryFuser
from model.god_encoder import GodEncoder

def compute_similarity(z1, z2):
    """Computes Cosine Similarity and L2 Distance between two MLX arrays."""
    z1_norm = z1 / mx.linalg.norm(z1, axis=-1, keepdims=True)
    z2_norm = z2 / mx.linalg.norm(z2, axis=-1, keepdims=True)
    cos_sim = mx.sum(z1_norm * z2_norm, axis=-1).item()
    l2_dist = mx.linalg.norm(z1 - z2, axis=-1).item()
    return cos_sim, l2_dist

def get_qwen_embedding(text, model, tokenizer, device):
    """Extracts last token pooling from Qwen and normalizes"""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1]
        seq_lengths = inputs['attention_mask'].sum(dim=1) - 1
        batch_size = last_hidden.size(0)
        embs = last_hidden[torch.arange(batch_size, device=device), seq_lengths, :]
        embs = torch.nn.functional.normalize(embs, p=2, dim=1)
    return embs

def main():
    print("Loading Phase 0 GodEncoder checkpoints...")
    # Initialize architecture based on dataset_config.json
    emb_dims = [1024, 3584, 1792, 2048, 1792] # BGE, Qwen, Xiaobu, Youtu, Conan
    d_model = 1024 # Actually d_model is 1024 according to safetensors
    z_dim = 1024
    
    fuser = SensoryFuser(emb_dims, d_model=d_model)
    god_encoder = GodEncoder(d_model=d_model, z_dim=z_dim)
    
    ckpt_dir = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "p0_v2_step_202913")
    
    try:
        fuser.load_weights(os.path.join(ckpt_dir, "sense_fuser.safetensors"))
        god_encoder.load_weights(os.path.join(ckpt_dir, "god_encoder.safetensors"))
        print("Successfully loaded GodEncoder weights!")
    except Exception as e:
        print(f"Warning: Could not load exact weights from {ckpt_dir}. {e}")
        
    print("\nLoading Embedding Models from local HF cache...")
    # We found these in your ~/.cache/huggingface equivalent directory!
    
    # 1. BGE Large ZH (Dim 1024)
    print("Loading BAAI/bge-large-zh-v1.5...")
    bge_model = SentenceTransformer('BAAI/bge-large-zh-v1.5')
    
    # 2. Qwen (Dim 3584) 
    # Using Qwen2.5-7B-Instruct found in your cache
    print("Loading Qwen/Qwen2.5-7B-Instruct... (This might take a minute and requires RAM/VRAM)")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
    # Using float16 to save memory
    qwen_model = AutoModel.from_pretrained("Qwen/Qwen2.5-7B-Instruct", torch_dtype=torch.float16, trust_remote_code=True).to(device)
    qwen_model.eval()
    
    print("\nModels loaded. Preparing OOD test cases...")
    
    test_cases = {
        "Python Code (Basic)": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
        "Python Code (Class)": "class DatabaseConnection:\n    def __init__(self, uri):\n        self.uri = uri\n    def connect(self):\n        pass",
        "English Medical text": "The patient presents with acute myocardial infarction and requires immediate angioplasty.",
        "Math Equation": "\\int_{0}^{\\infty} e^{-x^2} dx = \\frac{\\sqrt{\\pi}}{2}",
        "Chinese Text (In-Distribution Control)": "今天天气非常晴朗，我们全家人一起去公园里野餐，感受大自然的美好。"
    }
    
    print("\n--- Starting Verification ---")
    
    for name, text in test_cases.items():
        print(f"\n[Test Case: {name}]")
        print(f"Text snippet: {text[:50]}...")
        
        # 1. Extract Embeddings
        # BGE (1024D)
        emb_bge_pt = bge_model.encode([text], convert_to_tensor=True, normalize_embeddings=True)
        emb_bge_mx = mx.array(emb_bge_pt.cpu().numpy())
        
        # Qwen (3584D)
        emb_qwen_pt = get_qwen_embedding(text, qwen_model, qwen_tokenizer, device)
        emb_qwen_mx = mx.array(emb_qwen_pt.cpu().numpy().astype(np.float32))
        
        # 2. Pass through specific SenseAdapters
        # fuser.adapters[0] is BGE
        # fuser.adapters[1] is Qwen
        f_bge = fuser.adapters[0](emb_bge_mx) # [1, d_model]
        f_qwen = fuser.adapters[1](emb_qwen_mx) # [1, d_model]
        
        # 3. Pass through GodEncoder
        z_bge = god_encoder(f_bge) # [1, z_dim]
        z_qwen = god_encoder(f_qwen) # [1, z_dim]
        
        # 4. Compute Consistency
        cos_sim, l2_dist = compute_similarity(z_bge, z_qwen)
        
        print(f"  -> Cosine Similarity: {cos_sim:.4f}")
        print(f"  -> L2 Distance:       {l2_dist:.4f}")
        
        if cos_sim > 0.8:
            print("  => HIGH CONSISTENCY (Zero-shot topological alignment succeeded!)")
        else:
            print("  => LOW CONSISTENCY (OOD distortion detected. GodEncoder requires re-training on this domain.)")

if __name__ == "__main__":
    main()
