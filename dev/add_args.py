with open("training/train_phase3_1_latefusion.py", "r") as f:
    text = f.read()

anchor = 'parser = get_training_parser("Phase 3.1 Late Fusion Alignment")'
new_args = """
    parser.add_argument("--decoder_layers", type=int, default=8)
    parser.add_argument("--decoder_heads", type=int, default=16)"""

text = text.replace(anchor, anchor + new_args)

with open("training/train_phase3_1_latefusion.py", "w") as f:
    f.write(text)
