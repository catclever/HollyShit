import sys
import re

def process_script(filename, is_local=False):
    with open(filename, "r") as f:
        content = f.read()

    # Add start_epoch arg
    if '--start_epoch' not in content:
        content = content.replace(
            'parser.add_argument("--resume_ckpt", type=str, default="", help="Path to checkpoint to resume from")',
            'parser.add_argument("--resume_ckpt", type=str, default="", help="Path to checkpoint to resume from")\n    parser.add_argument("--start_epoch", type=int, default=0, help="Epoch to start from (e.g. 3 if resuming from epoch 3)")'
        )

    # Initialize model with window_size if it's 1.2
    if is_local:
        if "window_size=" not in content:
            content = content.replace(
                'num_heads=16\n    )',
                'num_heads=16,\n        window_size=15\n    )'
            )
    else:
        # ensure it's removed if it was there (just in case)
        content = content.replace(',\n        window_size=15\n    )', '\n    )')

    # Add start_epoch logic to the loop
    old_loop = "for epoch in range(args.epochs):"
    new_loop = "for epoch in range(args.start_epoch, args.epochs):"
    
    if old_loop in content:
        content = content.replace(old_loop, new_loop)

    with open(filename, "w") as f:
        f.write(content)
    print(f"Patched {filename}")

process_script("training/train_phase1_1_mixed.py", is_local=False)

import shutil
shutil.copy("training/train_phase1_1_mixed.py", "training/train_phase1_2_local.py")
process_script("training/train_phase1_2_local.py", is_local=True)

