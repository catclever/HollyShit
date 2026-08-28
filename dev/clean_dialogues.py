import json

def clean_item(line):
    if not line.strip(): return None
    item = json.loads(line)
    
    turns = []
    
    # Extract raw turns
    if isinstance(item, list):
        if len(item) == 2 and "答案：" in item[0] and "分段：" in item[1]:
            # This is the messy translated format
            # We ONLY keep the second part (item[1]) and split by "分段："
            raw_text = item[1].replace(' ', '') # remove spaces
            parts = raw_text.split("分段：")
            turns = [p.strip() for p in parts if p.strip()]
        else:
            # Normal list of strings
            turns = [str(x).replace(' ', '').strip() for x in item]
    elif isinstance(item, dict):
        val = item.get('dialogue', item.get('text', ''))
        if isinstance(val, list):
            turns = [str(x).replace(' ', '').strip() for x in val]
        else:
            # Single string, try to split if it has "分段："
            raw_text = str(val).replace(' ', '')
            if "答案：" in raw_text and "分段：" in raw_text:
                parts = raw_text.split("答案：")[-1].split("分段：")
                turns = [p.strip() for p in parts if p.strip()]
            else:
                turns = [raw_text.strip()]
    else:
        turns = [str(item).replace(' ', '').strip()]
        
    # Filter empty turns
    turns = [t for t in turns if t]
    if not turns:
        return None
        
    # Apply dialogue markers
    formatted_turns = []
    for i, turn in enumerate(turns):
        marker = "PSR1" if i % 2 == 0 else "PSR2"
        formatted_turns.append(f"{marker}{turn}")
        
    # Join into a single string for unconditional modeling
    return "".join(formatted_turns)

with open('data/Basic_ZH/dialogues_254k.jsonl', 'r') as fin, open('data/Basic_ZH/dialogues_254k_clean.jsonl', 'w') as fout:
    count = 0
    for line in fin:
        cleaned = clean_item(line)
        if cleaned:
            fout.write(json.dumps(cleaned, ensure_ascii=False) + '\n')
            count += 1
            
print(f"Cleaned {count} items.")
