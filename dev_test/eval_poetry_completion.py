import os
import sys
import json
import mlx.core as mx
from mlx_lm.utils import load_model, load_tokenizer, _download
from mlx_lm.generate import stream_generate

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from gemma_prosthesis.model import SemanticProjector
from gemma_prosthesis.alignment_fuser import DynamicAlignmentEmbedding, GatedProjector
from training.core.checkpoint import Checkpointer
from distilled_emb.model import TinyCharEncoder

def generate_with_stop(model, tokenizer, prompt, max_tokens=1024):
    out = ""
    for response in stream_generate(model, tokenizer, prompt, max_tokens=max_tokens):
        out += response.text
        if "<turn|>" in out or "<end_of_turn>" in out:
            break
    return out

def setup_models():
    print("-> 加载基座模型 (Gemma 4B)...")
    model_path = _download("google/gemma-4-E4B-it")
    tokenizer = load_tokenizer(model_path)
    gemma, _ = load_model(model_path, strict=False)
    gemma.eval()
    base_embed = gemma.language_model.model.embed_tokens

    print("-> 加载视网膜 (TinyCharEncoder)...")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    vocab_path = os.path.join(base_dir, "data", "Basic_ZH", "char_vocab.json")
    with open(vocab_path, "r", encoding="utf-8") as f:
        char_vocab = json.load(f)
    unk_id = char_vocab.get("[UNK]", 0)
    
    retina = TinyCharEncoder(vocab_size=len(char_vocab))
    retina_path = os.path.join(base_dir, "checkpoints", "distilled", "distilled_retina_mlx.safetensors")
    retina.load_weights(retina_path)
    
    print("-> 恢复神经外挂权重 (step_16000)...")
    base_projector = SemanticProjector(in_features=1024, out_features=2560)
    projector = GatedProjector(base_projector)
    
    ckpt_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'prosthesis'))
    checkpointer = Checkpointer(out_dir=ckpt_dir)
    checkpointer.register_model("projector", projector)
    checkpointer.load(os.path.join(ckpt_dir, "step_16000"))
    
    prosthesis_embed = DynamicAlignmentEmbedding(base_embed=base_embed, alpha=0.3)
    prosthesis_embed.gated_projector = projector
    prosthesis_embed.tokenizer = tokenizer
    
    return gemma, tokenizer, base_embed, prosthesis_embed, retina, char_vocab, unk_id

def run_test(gemma, tokenizer, base_embed, prosthesis_embed, retina, char_vocab, unk_id, questions, allow_thought=True):
    results = []
    
    import re
    for i, q in enumerate(questions):
        print(f"\n[{i+1}/{len(questions)}] 测试: {q}")
        
        if allow_thought:
            prompt = f"请问'{q}'的下一句是什么？请直接给出下一句的原文。"
        else:
            prompt = f"请问'{q}'的下一句是什么？请直接给出下一句的原文。 IMPORTANT: Please output only the original text of the next line. Do not output your thinking process."
            
        prompt_formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        
        # 智能分离思考过程和最终答案 (Robust Parser)
        def parse_output(text):
            import re
            thought = ""
            answer = text
            
            # Match <|channel>thought ... <channel|> OR </|channel>thought
            thought_match = re.search(r'<\|channel>thought(.*?)(?:<channel\|>|</\|channel>thought|\Z)', text, re.DOTALL)
            if thought_match:
                thought = thought_match.group(1).strip()
                # Remove the entire thought block from the text to get the answer
                answer = re.sub(r'<\|channel>thought.*?(?:<channel\|>|</\|channel>thought|\Z)', '', text, flags=re.DOTALL).strip()
            elif "Thinking Process:" in text:
                parts = text.split("Thinking Process:", 1)
                before_thought = parts[0].strip()
                thought_and_after = parts[1]
                
                # Assume thought ends at double newline or end of string
                thought_parts = thought_and_after.split("\n\n", 1)
                thought = "Thinking Process:\n" + thought_parts[0].strip()
                after_thought = thought_parts[1].strip() if len(thought_parts) > 1 else ""
                
                answer = (before_thought + "\n" + after_thought).strip()
            
            # Clean up <turn|> tags which Gemma might output
            answer = answer.replace("<turn|>", "").strip()
            
            if not answer:
                answer = "[被截断或未回答]"
                print(f"\n      [DEBUG RAW TEXT]\n{text[:300]}\n      [END DEBUG]\n")
                
            thought_md = thought.replace("\n", "<br>")
            answer_md = answer.replace("\n", "<br>")
            return thought_md, answer_md

        # ========== 1. 测试纯净版 (Base) ==========
        gemma.language_model.model.embed_tokens = base_embed
        base_out_raw = generate_with_stop(gemma, tokenizer, prompt=prompt_formatted, max_tokens=1024).strip()
        base_thought, base_ans = parse_output(base_out_raw)
        print(f"   [Base]     输出: {base_ans[:30]}...")
        
        # ========== 2. 测试外挂版 (Prosthesis) ==========
        gemma.language_model.model.embed_tokens = prosthesis_embed
        
        tiny_ids = [char_vocab.get(c, unk_id) for c in prompt]
        if len(tiny_ids) == 0: tiny_ids = [unk_id]
            
        tiny_mx = mx.array([tiny_ids])
        z_seq_char_pure = retina(tiny_mx, return_seq=True)
        
        prompt_start_idx = prompt_formatted.find(prompt)
        if prompt_start_idx == -1: prompt_start_idx = 0
            
        N_c_full = len(prompt_formatted)
        z_dim = z_seq_char_pure.shape[-1]
        z_seq_char = mx.zeros((1, N_c_full, z_dim))
        
        end_idx = prompt_start_idx + len(prompt)
        z_seq_char[0, prompt_start_idx:end_idx, :] = z_seq_char_pure[0]
        
        gemma.language_model.model.embed_tokens.current_z_seq_char = z_seq_char
        gemma.language_model.model.embed_tokens.current_text = prompt_formatted
        
        pros_out_raw = generate_with_stop(gemma, tokenizer, prompt=prompt_formatted, max_tokens=1024).strip()
        pros_thought, pros_ans = parse_output(pros_out_raw)
        print(f"   [Prosthesis] 输出: {pros_ans[:30]}...")
        
        results.append({
            "quote": q,
            "base_thought": base_thought,
            "base_ans": base_ans,
            "pros_thought": pros_thought,
            "pros_ans": pros_ans
        })
        
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Poetry Completion with Prosthesis")
    parser.add_argument("--limit", type=int, default=0, help="Limit the number of test cases (0 for all)")
    parser.add_argument("--disable_thought", action="store_true", help="Disable the model's chain of thought reasoning")
    parser.add_argument("--out_dir", type=str, default="", help="Directory to save the markdown report (defaults to ../evals)")
    parser.add_argument("--out_name", type=str, default="prosthesis_poetry_completion_eval.md", help="Name of the output markdown file")
    args = parser.parse_args()

    json_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "Basic_ZH", "poems_benchmark.json"))
    with open(json_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
        
    if args.limit > 0:
        test_data = test_data[:args.limit]
        
    print(f"-> 成功加载 {len(test_data)} 条测试古诗名句。")
    
    questions = [d["q"] for d in test_data]
    gemma, tokenizer, base, pros, retina, vocab, unk = setup_models()
    res = run_test(gemma, tokenizer, base, pros, retina, vocab, unk, questions, allow_thought=not args.disable_thought)
    
    base_correct = 0
    pros_correct = 0
    
    md_table = "| 古诗名句 | 标准答案 | Base 思考链 | Base 答案 | Prosthesis 思考链 | Prosthesis 答案 |\n"
    md_table += "|---|---|---|---|---|---|\n"
    for i, r in enumerate(res):
        correct_ans = test_data[i]["a"]
        
        # Simple substring matching for accuracy
        base_is_correct = correct_ans in r['base_ans']
        pros_is_correct = correct_ans in r['pros_ans']
        
        if base_is_correct: base_correct += 1
        if pros_is_correct: pros_correct += 1
        
        base_icon = "✅" if base_is_correct else "❌"
        pros_icon = "✅" if pros_is_correct else "❌"
        
        md_table += f"| {r['quote']} | **{correct_ans}** | {r['base_thought']} | {base_icon} **{r['base_ans']}** | {r['pros_thought']} | {pros_icon} **{r['pros_ans']}** |\n"
        
    total = len(test_data)
    summary = f"## 准确率统计\n\n- **Base 模型**: {base_correct}/{total} ({base_correct/total*100:.1f}%)\n- **Prosthesis 模型**: {pros_correct}/{total} ({pros_correct/total*100:.1f}%)\n\n"
        
    # Ensure the evals directory exists
    if args.out_dir:
        evals_dir = os.path.abspath(args.out_dir)
    else:
        evals_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'evals'))
        
    os.makedirs(evals_dir, exist_ok=True)
    
    out_file = os.path.join(evals_dir, args.out_name)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("# 古诗文分词灾难 vs 视网膜修正 对比测试报告\n\n")
        f.write(summary)
        f.write(md_table)
        
    print(f"\n✅ 测试完成！已生成报告: {out_file}")
    print(summary.strip())
