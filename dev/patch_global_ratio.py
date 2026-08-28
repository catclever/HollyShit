import sys

with open("training/train_decoder_pretrain.py", "r") as f:
    content = f.read()

if "--global_attention_ratio" not in content:
    content = content.replace(
        "parser.add_argument(\"--window_size\", type=int, default=None)",
        "parser.add_argument(\"--window_size\", type=int, default=None)\n    parser.add_argument(\"--global_attention_ratio\", type=float, default=0.0, help=\"Probability of using global attention instead of windowed attention per batch.\")"
    )

import_random = "import random\n"
if "import random" not in content:
    content = import_random + content

target_insert = """
                if args.window_size is not None:
                    if random.random() < args.global_attention_ratio:
                        model.window_size = None
                    else:
                        model.window_size = args.window_size
"""
if "model.window_size = None" not in content:
    content = content.replace(
        "x_np_full = np.array(x_np)",
        target_insert + "\n                x_np_full = np.array(x_np)"
    )

with open("training/train_decoder_pretrain.py", "w") as f:
    f.write(content)
print("Patched global_attention_ratio")
