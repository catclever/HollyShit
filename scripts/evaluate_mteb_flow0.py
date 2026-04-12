import argparse
import numpy as np
import mlx.core as mx
import os
import json
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import mteb
    import torch
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("请安装依赖： pip install mteb[zh] numpy sentence-transformers torch")
    sys.exit(1)

from model.config import ModelConfig
from model.adapter import SensoryFuser
from model.god_encoder import GodEncoder

class FlowMTEBWrapper:
    """
    针对 Phase 0 / Flow 0 的 GodEncoder 评测包装器。
    """
    def __init__(self, fuser, god_encoder, base_model, active_idx, emb_dims):
        self.fuser = fuser
        self.god_encoder = god_encoder
        self.base_model = base_model
        self.active_idx = active_idx
        self.emb_dims = emb_dims
        
        self.model_name = "FlowGodEncoder"
        self.revision = "1.0.0"
        self.device = "cpu"
        
        # 权重：只有当前的 active_idx 是 1.0 (或者用平均)，其他全为 0.0 防止引发干扰
        self.weights = [0.0] * len(emb_dims)
        self.weights[active_idx] = 1.0
        
    @property
    def mteb_model_meta(self):
        class MockMeta:
            name = "FlowGodEncoder"
            revision = "1.0.0"
            release_date = "2026-04-11"
            languages = ["cmn"]
            framework = []
            
            def model_name_as_path(self):
                return "FlowGodEncoder"
                
            def to_dict(self):
                return {
                    "name": self.name, 
                    "revision": self.revision,
                    "languages": self.languages,
                    "release_date": self.release_date
                }
                
            def __getattr__(self, name):
                return None
                
        return MockMeta()
        
    def encode(self, inputs, *, task_metadata=None, hf_split=None, hf_subset=None, prompt_type=None, **kwargs) -> np.ndarray:
        # 适配 MTEB >= 2.11 新版 EncoderProtocol
        sentences = inputs
        batch_size = kwargs.pop("batch_size", 64)
        if type(sentences).__name__ == "DataLoader":
            flat_sentences = []
            for batch in sentences:
                if isinstance(batch, (list, tuple)): flat_sentences.extend(batch)
                elif isinstance(batch, dict): flat_sentences.extend(list(batch.values())[0])
                else: flat_sentences.append(batch)
            sentences = flat_sentences
        elif not isinstance(sentences, list):
            sentences = list(sentences)
            
        all_z = []
        
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i : i + batch_size]
            
            # 1. 临时用基座模型抽出真正的特征
            with torch.no_grad():
                # SentenceTransformer encode
                base_embs = self.base_model.encode(batch, convert_to_numpy=True, normalize_embeddings=True)
                
            if len(base_embs.shape) == 1:
                base_embs = np.expand_dims(base_embs, axis=0)
            batch_size_actual = base_embs.shape[0]
            
            # 2. 构造 SensoryFuser 的输入 (List of MLX arrays)
            ml_embs = []
            for j, dim in enumerate(self.emb_dims):
                if j == self.active_idx:
                    ml_embs.append(mx.array(base_embs))
                else:
                    # 伪造占位符 (会被 weights 乘以 0 抹去)
                    ml_embs.append(mx.zeros((batch_size_actual, dim)))
                    
            # 3. 映射到上帝空间 z_target
            f_t = self.fuser(ml_embs, weights=self.weights)
            z_target = self.god_encoder(f_t)
            mx.eval(z_target)
            
            # 存入结果
            all_z.append(np.array(z_target))
            
        return np.concatenate(all_z, axis=0)

    def similarity(self, embeddings1, embeddings2):
        emb1 = embeddings1 / np.maximum(np.linalg.norm(embeddings1, axis=1, keepdims=True), 1e-9)
        emb2 = embeddings2 / np.maximum(np.linalg.norm(embeddings2, axis=1, keepdims=True), 1e-9)
        return emb1 @ emb2.T

    def similarity_pairwise(self, embeddings1, embeddings2):
        emb1 = embeddings1 / np.maximum(np.linalg.norm(embeddings1, axis=1, keepdims=True), 1e-9)
        emb2 = embeddings2 / np.maximum(np.linalg.norm(embeddings2, axis=1, keepdims=True), 1e-9)
        return np.sum(emb1 * emb2, axis=1)

def main():
    parser = argparse.ArgumentParser(description="Flow 0 GodEncoder MTEB Evaluator")
    parser.add_argument("--ckpt", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--tasks", type=str, nargs="+", default=["LCQMC", "BQ", "AFQMC"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--model", type=str, default="bge", choices=["bge", "qwen", "xiaobu", "youtu", "conan_v1"], help="测试哪个基座维度的对齐能力")
    parser.add_argument("--config_file", type=str, default="dataset_config.json")
    args = parser.parse_args()

    # 0. 内置模型映射表
    model_mapping = {
        "bge": "BAAI/bge-m3",
        "qwen": "iic/gte_Qwen2-7B-instruct",
        "xiaobu": "lier007/xiaobu-embedding-v2",
        "youtu": "tencent/Youtu-Embedding",
        "conan_v1": "TencentBAC/Conan-embedding-v1"
    }
    
    base_model_id = model_mapping[args.model]

    # 1. 初始化模型架构
    with open(args.config_file, "r") as f:
        ds_config = json.load(f)
    base_models = ds_config.get("base_models", ["bge", "qwen", "xiaobu", "youtu", "conan_v1"])
    emb_dims = ds_config.get("emb_dims", [1024, 3584, 1792, 2048, 1792])
    
    # 自动算出 active_idx
    if args.model in base_models:
        active_idx = base_models.index(args.model)
    else:
        raise ValueError(f"Model {args.model} 不在 dataset_config.json 的 base_models 列表中!")

    config = ModelConfig()
    d_model = config.decoder_heads * 64
    
    fuser = SensoryFuser(emb_dims, d_model)
    god_encoder = GodEncoder(d_model, config.z_dim)

    print(f"1. Loading Flow_0 weights from {args.ckpt}...")
    fuser_path = f"{args.ckpt}/sense_fuser.safetensors"
    if not os.path.exists(fuser_path):
        fuser_path = f"{args.ckpt}/sense_adapter.safetensors"
    
    fuser.load_weights(fuser_path)
    god_encoder.load_weights(f"{args.ckpt}/god_encoder.safetensors")
    mx.eval(fuser.parameters(), god_encoder.parameters())

    print(f"2. Booting up foundation model: {args.model} ({base_model_id}) via Torch/ST...")
    # 启用 trust_remote_code，否则跑 qwen 会报错
    base_st_model = SentenceTransformer(base_model_id, device="cpu", trust_remote_code=True)

    # 3. 包装模型并启动 MTEB
    fast_wrapper = FlowMTEBWrapper(fuser, god_encoder, base_st_model, active_idx, emb_dims)
    
    print(f"\n=============================================")
    print(f"开始启动 C-MTEB 评测任务：{args.tasks}")
    print(f"基线探测源：{args.model} (idx: {active_idx})")
    print(f"=============================================\n")

    active_tasks = mteb.get_tasks(tasks=args.tasks)
    eval_engine = mteb.MTEB(tasks=active_tasks)
    
    out_dir = f"mteb_flow_eval/{os.path.basename(args.ckpt)}_base_{args.model}"
    eval_engine.run(fast_wrapper, output_folder=out_dir, encode_kwargs={"batch_size": args.batch_size})
    
    print(f"\n打榜结束！成绩已经详细保存到了 {out_dir}")

if __name__ == "__main__":
    main()
