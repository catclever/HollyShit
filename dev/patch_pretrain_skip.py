import sys

with open("training/train_decoder_pretrain.py", "r") as f:
    content = f.read()

# Make sure we import random
if "import random" not in content:
    content = content.replace("import os", "import os\nimport random")

# Add random seed based on args.seed or epoch logic
loop_start = "for epoch in range(start_epoch, args.epochs):"
new_loop_start = """for epoch in range(start_epoch, args.epochs):
            # Seed the dataloader for this epoch to ensure reproducible shuffling
            # If a global seed is provided, use it. Otherwise, fallback to a stable deterministic seed.
            epoch_seed = (args.seed if args.seed is not None else 42) + epoch
            mx.random.seed(epoch_seed)
            np.random.seed(epoch_seed)
            import torch
            torch.manual_seed(epoch_seed)
            random.seed(epoch_seed)
            
            # Calculate how many steps to skip if we resumed mid-epoch
            skip_steps = 0
            if epoch == start_epoch and resume_ckpt_path:
                skip_steps = checkpointer.get_skip_steps(global_step, len(dataloader))
                if skip_steps > 0:
                    print(f"Resuming mid-epoch: skipping first {skip_steps} batches of Epoch {epoch+1}")"""

if "skip_steps = 0" not in content:
    content = content.replace(loop_start, new_loop_start)

# Add the actual skipping logic inside the dataloader iteration
inner_loop = "for x_np in pbar:"
new_inner_loop = """for step_in_epoch, x_np in enumerate(pbar):
                if skip_steps > 0 and step_in_epoch < skip_steps:
                    continue"""
                    
if "step_in_epoch < skip_steps" not in content:
    content = content.replace(inner_loop, new_inner_loop)

with open("training/train_decoder_pretrain.py", "w") as f:
    f.write(content)
print("Patched pretrain script with pseudo-random seeding and exact batch skipping")
