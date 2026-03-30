"""
【脚本功能】：神算子！开源顶级 Embedding 模型摸底全自动考场与战报生成器
【使用场景】：Phase 0 教师模型挑选期。当面对海量开源大模型 (GTE, BGE, RoBERTa 等) 犹豫不决时，一键启动本脚本！
             它会自动拉起 MTEB 框架对指定模型进行极限拷问（LCQMC, BQ, AFQMC 语义评测）；
             当所有模型跑完后，它会自动剥开层层嵌套的复杂 JSON 战报，直接在前线大屏给您打印出最硬核的 Accuracy / Spearman 综合战力榜单！
【用法示例】：
    - 完整考核并生成榜单：`python tests/evaluate_mteb_teachers.py`
    - 只看现有的历史榜单（跳过极耗时间的跑分）：`python tests/evaluate_mteb_teachers.py --skip_eval`
"""
import os
import json
import glob
import argparse
import warnings

try:
    import mteb
    from sentence_transformers import SentenceTransformer
except ImportError:
    pass

warnings.filterwarnings('ignore')

TASKS = ["LCQMC", "BQ", "AFQMC"]

def get_safe_name(model_path):
    # Extracts pure model names from absolute linux paths or HF IDs elegantly
    # Examples:
    # "/hy-tmp/models/iic/gte_Qwen2-7B-instruct" -> "gte_Qwen2-7B-instruct"
    # "BAAI/bge-large-zh-v1.5" -> "bge-large-zh-v1.5"
    return os.path.basename(os.path.normpath(model_path))

def run_evaluation(models_to_test, output_root, device, batch_size):
    print(f"🔥 [STAGE 1] 启动 MTEB 终极考场！共有 {len(models_to_test)} 名开源选手参赛。")
    for model_name in models_to_test:
        print(f"\n================ 正在考核选手: {model_name} ================")
        try:
            model = SentenceTransformer(model_name, device=device)
            active_tasks = mteb.get_tasks(tasks=TASKS)
            eval_engine = mteb.MTEB(tasks=active_tasks)
            # Smart isolation of the model's base identity
            safe_name = get_safe_name(model_name)
            output_folder = os.path.join(output_root, safe_name)
            
            # The engine will automatically skip what has already been evaluated, saving massive time!
            eval_engine.run(model, output_folder=output_folder, encode_kwargs={"batch_size": batch_size})
        except Exception as e:
            print(f"[!] 选手 {model_name} 崩溃或因网络退赛: {e}")

def parse_and_report(models_to_test, output_root):
    print(f"\n🏆 [STAGE 2] 提取多维战报，生成绝密考核榜单！")
    print("=" * 60)
    for model_name in models_to_test:
        safe_name = get_safe_name(model_name)
        print(f"【参赛兵器】: {safe_name}")
        
        for task in TASKS:
            files = glob.glob(os.path.join(output_root, safe_name, "**", f"{task}*.json"), recursive=True)
            if not files: 
                print(f"   ├─ 任务 {task}: [缺考/漏考]")
                continue
            
            latest_file = max(files, key=os.path.getmtime)
            with open(latest_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            scores = data.get("scores", {})
            test_scores = scores.get("test", [{}])[0] if "test" in scores else scores.get("validation", [{}])[0]
            if not test_scores: 
                test_scores = data.get("test", {})
            
            metric = 0.0
            if "accuracy" in test_scores: 
                metric = test_scores["accuracy"]
            elif "cos_sim" in test_scores and "spearman" in test_scores["cos_sim"]: 
                metric = test_scores["cos_sim"]["spearman"]
            elif "spearman" in test_scores: 
                metric = test_scores["spearman"]
                
            print(f"   ├─ 任务 {task}: 战力得分 {metric*100:.2f} 分")
        print("-" * 60)

def main():
    parser = argparse.ArgumentParser(description="MTEB 自动化双轨考核引擎 (Evaluate + Parse)")
    parser.add_argument("--models", nargs="+", 
                        default=[
                            "hfl/chinese-roberta-wwm-ext",
                            "shibing624/text2vec-base-chinese",
                            "thenlper/gte-large-zh",
                            "BAAI/bge-large-zh-v1.5"
                        ], 
                        help="一个或多个大模型的 HuggingFace ID 或绝对本地路径 (以空格分隔)")
    parser.add_argument("--skip_eval", action="store_true", help="如果加了这个参数，将直接绕过漫长的跑分环节，只打印最后生成的战绩榜单。")
    parser.add_argument("--output_dir", type=str, default="./mteb_results", help="挂载存放测试结果集的根目录（例如 ./mteb_results 或者 ../distilled_embs）")
    parser.add_argument("--device", type=str, default="cpu", help="推理设备架构 (cpu, cuda, mps 等)")
    parser.add_argument("--batch_size", type=int, default=8, help="MTEB 提取的 Batch 大小。防 OOM 请设小。")
    args = parser.parse_args()

    # Create root directory if needed
    os.makedirs(args.output_dir, exist_ok=True)

    if not args.skip_eval:
        run_evaluation(args.models, args.output_dir, args.device, args.batch_size)
    else:
        print(">> 检测到 --skip_eval 指令，绕过现场考核跑分环节。 <<")
        
    parse_and_report(args.models, args.output_dir)
    print("\n✅ 全自动化测评战机执行完毕！")

if __name__ == "__main__":
    main()
