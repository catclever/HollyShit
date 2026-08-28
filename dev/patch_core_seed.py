import sys

with open("training/core/checkpoint.py", "r") as f:
    content = f.read()

# Update Checkpointer.__init__ or register_args to initialize a random seed if None
# It's better to do it in register_args
old_register_args = """    def register_args(self, args):
        self._args = args"""
        
new_register_args = """    def register_args(self, args):
        if hasattr(args, "seed") and getattr(args, "seed") is None:
            import random
            args.seed = random.randint(0, 2**32 - 1)
            print(f"No seed provided. Auto-generated persistent random seed: {args.seed}")
        self._args = args"""
        
if old_register_args in content:
    content = content.replace(old_register_args, new_register_args)

with open("training/core/checkpoint.py", "w") as f:
    f.write(content)
print("Patched Checkpointer to auto-generate persistent seed if None")
