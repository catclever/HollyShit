import re
import os

with open("training/train_phase1_1_mixed.py", "r") as f:
    content = f.read()

# Remove the dumb start_epoch argument
content = content.replace(
    'parser.add_argument("--start_epoch", type=int, default=0, help="Epoch to start from (e.g. 3 if resuming from epoch 3)")\n',
    ''
)

# Add window_size argument
content = content.replace(
    'parser.add_argument("--resume_ckpt", type=str, default="", help="Path to checkpoint to resume from")',
    'parser.add_argument("--resume_ckpt", type=str, default="", help="Path to checkpoint to resume from")\n    parser.add_argument("--window_size", type=int, default=None, help="Local attention window size (default: None for full causal)")'
)

# Pass window_size to model
content = content.replace(
    'num_heads=16\n    )',
    'num_heads=16,\n        window_size=args.window_size\n    )'
)

# Add regex parsing logic for step/epoch
regex_logic = """    global_step = 0
    if args.resume_ckpt:
        print(f"Resuming from {args.resume_ckpt}...")
        model.load_weights(args.resume_ckpt)
        
        # Parse step or epoch from filename
        basename = os.path.basename(args.resume_ckpt)
        step_match = re.search(r'_step_(\d+)', basename)
        epoch_match = re.search(r'_epoch_(\d+)', basename)
        
        if step_match:
            global_step = int(step_match.group(1))
            print(f"Parsed global step: {global_step} from filename.")
        elif epoch_match:
            parsed_epoch = int(epoch_match.group(1))
            global_step = parsed_epoch * len(dataloader)
            print(f"Parsed epoch: {parsed_epoch}, calculated global step: {global_step}.")
            
    start_epoch = global_step // len(dataloader) if len(dataloader) > 0 else 0
    step = global_step
"""

# Replace the old resume logic
old_resume = """    if args.resume_ckpt:
        print(f"Resuming from {args.resume_ckpt}...")
        model.load_weights(args.resume_ckpt)
"""
content = content.replace(old_resume, regex_logic)

# Replace the loop start
old_loop = "for epoch in range(args.start_epoch, args.epochs):"
new_loop = "for epoch in range(start_epoch, args.epochs):"
content = content.replace(old_loop, new_loop)

# Save as train_decoder_pretrain.py
with open("training/train_decoder_pretrain.py", "w") as f:
    f.write(content)
print("Created train_decoder_pretrain.py")
