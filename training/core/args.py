import argparse

def get_training_parser(description: str = "Training Script"):
    """
    Returns a unified ArgumentParser pre-populated with common training arguments.
    Additional script-specific arguments can be added to it before parsing.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=3, help="Total number of epochs to train")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--save_steps", type=int, default=1000, help="Save a checkpoint every X steps")
    parser.add_argument("--z_dim", type=int, default=1024, help="Dimension of the absolute truth anchor space (z_target)")
    parser.add_argument("--fusion_alpha", type=float, default=0.7, help="Weight of the randomly selected primary embedding logic (rest divided equally)")
    parser.add_argument("--out_dir", type=str, default="checkpoints/run", help="Output directory for checkpoints")
    parser.add_argument("--ckpt_prefix", type=str, nargs="?", const="TIMESTAMP", default=None, help="Prefix for checkpoint folders. If passed empty, uses timestamp.")
    parser.add_argument("--tokenizer_id", type=str, default="Qwen/Qwen2.5-7B", help="HuggingFace Tokenizer ID")
    
    # Dual Resume Mechanism
    parser.add_argument("--continue", dest="auto_resume", action="store_true", help="Automatically find and resume from the latest checkpoint or emergency save")
    parser.add_argument("--resume_from", type=str, default=None, help="Explicitly specify the path to a checkpoint directory to resume from")
    
    parser.add_argument("--warmup_steps", type=int, default=2000, help="Learning rate linear warmup steps")
    
    return parser
