with open("training/train_phase1_1_decoder_pretrain.py", "r") as f:
    code = f.read()

# Replace the lazy string casting with proper dialogue concatenation
old_parse = """                if isinstance(item, dict):
                    text = item.get('dialogue', item.get('text', str(item)))
                    if isinstance(text, list):
                        text = " ".join(text)
                else:
                    text = str(item)"""

new_parse = """                if isinstance(item, list):
                    # Multi-turn dialogue: list of strings
                    text = " ".join([str(x) for x in item])
                elif isinstance(item, dict):
                    text = item.get('dialogue', item.get('text', ''))
                    if isinstance(text, list):
                        text = " ".join([str(x) for x in text])
                    else:
                        text = str(text)
                else:
                    text = str(item)"""

code = code.replace(old_parse, new_parse)

with open("training/train_phase1_1_decoder_pretrain.py", "w") as f:
    f.write(code)
