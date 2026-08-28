import json
from datasets import load_dataset

print("Downloading thu-coai/lccc base dataset...")
# We use streaming to quickly grab a subset without downloading everything
ds = load_dataset("thu-coai/lccc", "base", split="train", streaming=True)

out_path = "data/Basic_ZH/lccc_dialogues.jsonl"
count = 0
target = 250000

with open(out_path, "w", encoding="utf-8") as f:
    for item in ds:
        # LCCC dialogue is usually a list of strings
        dialogue = item.get("dialogue", item.get("text", []))
        if not dialogue:
            continue
        
        # Clean up and ensure no spaces between words
        # Join with a period or just space if it's natural, but actually Chinese doesn't use spaces.
        # So we can just join turns with '。' or '\n' 
        clean_text = "。".join([str(t).replace(" ", "") for t in dialogue])
        
        if clean_text:
            f.write(json.dumps(clean_text, ensure_ascii=False) + "\n")
            count += 1
            
        if count >= target:
            break
            
print(f"Saved {count} dialogues to {out_path}")
