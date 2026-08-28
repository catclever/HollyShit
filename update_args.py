import re

with open("training/train_phase1_1_decoder_pretrain.py", "r") as f:
    code = f.read()

# Replace my ad-hoc argparse with get_training_parser
argparse_block = """    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--z_dim', type=int, default=1024)
    parser.add_argument('--output_dir', type=str, default='checkpoints/phase1_1_decoder')
    parser.add_argument('--run_name', type=str, default='decoder_grammar_pretrain')
    args = parser.parse_args()"""

replacement = """    from training.core.args import get_training_parser
    parser = get_training_parser("Phase 1.1 Decoder Grammar Pretrain")
    parser.add_argument('--run_name', type=str, default='decoder_grammar_pretrain')
    args = parser.parse_args()
    args.output_dir = args.out_dir"""

code = code.replace(argparse_block, replacement)

with open("training/train_phase1_1_decoder_pretrain.py", "w") as f:
    f.write(code)
