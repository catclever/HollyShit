import torch
import numpy as np
import mlx.core as mx
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
import os
import sys
from scipy.stats import pearsonr, spearmanr

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.adapter import SensoryFuser
from model.god_encoder import GodEncoder

def get_qwen_embedding(texts, model, tokenizer, device):
    """Extracts last token pooling from Qwen and normalizes for a batch of texts."""
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1]
        seq_lengths = inputs['attention_mask'].sum(dim=1) - 1
        batch_size = last_hidden.size(0)
        embs = last_hidden[torch.arange(batch_size, device=device), seq_lengths, :]
        embs = torch.nn.functional.normalize(embs, p=2, dim=1)
    return embs

def compute_sim_matrix(embs):
    """Computes NxN cosine similarity matrix."""
    embs_norm = embs / np.linalg.norm(embs, axis=-1, keepdims=True)
    return np.dot(embs_norm, embs_norm.T)

def get_upper_tri(matrix):
    """Extracts the upper triangle of a matrix (excluding diagonal)."""
    indices = np.triu_indices_from(matrix, k=1)
    return matrix[indices]

def main():
    print("Loading Phase 0 Checkpoints...")
    emb_dims = [1024, 3584, 1792, 2048, 1792]
    d_model = 1024
    z_dim = 1024
    
    fuser = SensoryFuser(emb_dims, d_model=d_model)
    god_encoder = GodEncoder(d_model=d_model, z_dim=z_dim)
    
    ckpt_dir = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "p0_v2_step_202913")
    try:
        fuser.load_weights(os.path.join(ckpt_dir, "sense_fuser.safetensors"))
        god_encoder.load_weights(os.path.join(ckpt_dir, "god_encoder.safetensors"))
    except Exception as e:
        print(f"Failed to load weights: {e}")
        return

    print("Loading BAAI/bge-large-zh-v1.5...")
    bge_model = SentenceTransformer('BAAI/bge-large-zh-v1.5')

    # 15 distinct code snippets covering different concepts (loops, classes, recursion, SQL, HTML, data sci)
    code_snippets = [
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
        "def factorial(n):\n    if n == 0:\n        return 1\n    return n * factorial(n-1)",
        "class User:\n    def __init__(self, name, age):\n        self.name = name\n        self.age = age",
        "class DatabaseConnection:\n    def connect(self):\n        pass\n    def execute(self, query):\n        pass",
        "SELECT * FROM users WHERE age > 18 ORDER BY created_at DESC;",
        "UPDATE employees SET salary = salary * 1.1 WHERE department = 'Engineering';",
        "<div><h1>Hello World</h1><p>Welcome to my website!</p></div>",
        "<button onClick={() => alert('Clicked!')}>Click Me</button>",
        "import pandas as pd\ndf = pd.read_csv('data.csv')\nprint(df.describe())",
        "import numpy as np\nmatrix = np.zeros((5, 5))\nmatrix += 1",
        "for i in range(10):\n    if i % 2 == 0:\n        print(i)",
        "items = [x*2 for x in range(100) if x % 5 == 0]",
        "try:\n    result = 10 / 0\nexcept ZeroDivisionError:\n    print('Cannot divide by zero')",
        "with open('file.txt', 'r') as f:\n    content = f.read()",
        "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]"
    ]

    print(f"Extracting BGE features for {len(code_snippets)} code snippets...")
    bge_embs_np = bge_model.encode(code_snippets, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    
    print("Passing through BGE Adapter and GodEncoder...")
    bge_embs_mx = mx.array(bge_embs_np)
    
    f_bge = fuser.adapters[0](bge_embs_mx)
    z_god = god_encoder(f_bge)
    
    z_god_np = np.array(z_god)
    
    print("Computing Distance Matrices...")
    sim_matrix_original = compute_sim_matrix(bge_embs_np)
    sim_matrix_god = compute_sim_matrix(z_god_np)
    
    upper_original = get_upper_tri(sim_matrix_original)
    upper_god = get_upper_tri(sim_matrix_god)
    
    pearson_corr, p_value_p = pearsonr(upper_original, upper_god)
    spearman_corr, p_value_s = spearmanr(upper_original, upper_god)
    
    print("\n" + "="*50)
    print("🌟 Topology Consistency Verification Results 🌟")
    print("="*50)
    print(f"Snippets evaluated: {len(code_snippets)}")
    print(f"Pearson Correlation:  {pearson_corr:.4f} (p-value: {p_value_p:.4e})")
    print(f"Spearman Correlation: {spearman_corr:.4f} (p-value: {p_value_s:.4e})")
    print("="*50)
    
    if pearson_corr > 0.8:
        print("Verdict: HIGH TOPOLOGY PRESERVATION.")
        print("The GodEncoder effectively preserves the semantic structure of Code!")
    elif pearson_corr > 0.4:
        print("Verdict: MODERATE TOPOLOGY PRESERVATION.")
        print("Some semantic structure is preserved, but heavy distortion occurs.")
    else:
        print("Verdict: TOPOLOGICAL COLLAPSE.")
        print("The GodEncoder completely scrambles the Code space into noise.")

if __name__ == "__main__":
    main()
