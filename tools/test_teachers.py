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

tasks = ["LCQMC", "BQCorpus", "AFQMC"]

for model_name in models_to_test:
    print(f"\n================ Eval {model_name} ================")
    try:
        model = SentenceTransformer(model_name)
        active_tasks = mteb.get_tasks(tasks=tasks)
        eval_engine = mteb.MTEB(tasks=active_tasks)
        eval_engine.run(model, output_folder=f"distilled_emb/mteb_results/{model_name.replace('/', '_')}")
    except Exception as e:
        print(f"Failed on {model_name}: {e}")
