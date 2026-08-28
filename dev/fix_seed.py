with open("training/core/checkpoint.py", "r") as f:
    text = f.read()

# Fix register_args
old_register = """    def register_args(self, args: argparse.Namespace):
        \"\"\"Register parsed CLI arguments to be saved inside the checkpoint.\"\"\"
        self._args = args"""
        
new_register = """    def register_args(self, args: argparse.Namespace):
        \"\"\"Register parsed CLI arguments to be saved inside the checkpoint.\"\"\"
        if hasattr(args, 'seed') and args.seed is None:
            import time
            args.seed = int(time.time() * 1000) % 1000000
            print(f"No seed provided. Auto-generated fresh random seed: {args.seed}")
        self._args = args"""
        
text = text.replace(old_register, new_register)

# Fix setup_epoch_seed
old_setup = """    def setup_epoch_seed(self, epoch: int, base_seed: int = None):
        \"\"\"
        Sets deterministic pseudo-random seeds for the current epoch to ensure reproducible dataloader shuffling.
        If base_seed is None, falls back to 42.
        \"\"\"
        try:
            if base_seed is None:
                base_seed = 42
            epoch_seed = base_seed + epoch"""
            
new_setup = """    def setup_epoch_seed(self, epoch: int, base_seed: int = None):
        \"\"\"
        Sets deterministic pseudo-random seeds for the current epoch to ensure reproducible dataloader shuffling.
        \"\"\"
        try:
            if base_seed is None:
                raise ValueError("base_seed cannot be None! It should have been auto-generated in register_args or loaded from checkpoint.")
            epoch_seed = base_seed + epoch"""
            
text = text.replace(old_setup, new_setup)

with open("training/core/checkpoint.py", "w") as f:
    f.write(text)
print("Fixed Checkpointer seed logic!")
