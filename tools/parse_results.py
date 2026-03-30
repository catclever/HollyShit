"""
【脚本功能】：MTEB (Massive Text Embedding Benchmark) 结果嗅探器
【使用场景】：Phase 0 教师验证。当跑完 `tools/test_teachers.py` 后，在杂乱的多级 JSON 结果目录中快速剥离并打印出 LCQMC, BQ, AFQMC 这三大测试集的核心 Accuracy/Spearman 得分。
【用法示例】：`python tools/parse_results.py`
"""
import json, glob, os

models = [
    "BAAI_bge-large-zh-v1.5",
    "thenlper_gte-large-zh",
    "shibing624_text2vec-base-chinese"
]
tasks = ["LCQMC", "BQ", "AFQMC"]

for model in models:
    print(f"\n[{model}]")
    for task in tasks:
        files = glob.glob(f"distilled_emb/mteb_results/{model}/**/{task}*.json", recursive=True)
        if not files: 
            print(f"  {task}: Missing")
            continue
        
        latest_file = max(files, key=os.path.getmtime)
        with open(latest_file, 'r') as f:
            data = json.load(f)
            
        scores = data.get("scores", {})
        test_scores = scores.get("test", [{}])[0] if "test" in scores else scores.get("validation", [{}])[0]
        if not test_scores: test_scores = data.get("test", {})
        
        metric = 0.0
        if "accuracy" in test_scores: metric = test_scores["accuracy"]
        elif "cos_sim" in test_scores and "spearman" in test_scores["cos_sim"]: metric = test_scores["cos_sim"]["spearman"]
        elif "spearman" in test_scores: metric = test_scores["spearman"]
            
        print(f"  {task}: {metric*100:.2f}")
