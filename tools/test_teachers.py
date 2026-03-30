"""
【脚本功能】：开源顶级 Embedding 模型测试场 (Teacher MTEB Evaluator)
【使用场景】：Phase 0 教师选择期。使用 MTEB 框架自动下载各个预选的开源大模型 (例如 GTE, BGE, RoBERTa)，并在三大经典中文短文本任务上对它们的语义提取能力进行摸底硬考，最终将战报落盘在 `distilled_emb/` 目录下。
【用法示例】：`python tools/test_teachers.py`
"""
import mteb
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

models_to_test = [
    "hfl/chinese-roberta-wwm-ext",
    "shibing624/text2vec-base-chinese",
    "thenlper/gte-large-zh",
    "BAAI/bge-large-zh-v1.5"
]

tasks = ["LCQMC", "BQ", "AFQMC"]

for model_name in models_to_test:
    print(f"\n================ Eval {model_name} ================")
    try:
        model = SentenceTransformer(model_name)
        active_tasks = mteb.get_tasks(tasks=tasks)
        eval_engine = mteb.MTEB(tasks=active_tasks)
        eval_engine.run(model, output_folder=f"distilled_emb/mteb_results/{model_name.replace('/', '_')}")
    except Exception as e:
        print(f"Failed on {model_name}: {e}")
