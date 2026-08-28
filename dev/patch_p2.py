import sys

with open("training/train_phase2_thoughtengine.py", "r") as f:
    content = f.read()

# Add load_args and wrap_dataloader
if "checkpointer.load_args" not in content:
    content = content.replace(
        "if args.auto_resume:",
        "if args.resume_from:\n        args = checkpointer.load_args(args.resume_from, args)\n        global_step = checkpointer.load(args.resume_from)\n        print(f'Resumed at global step {global_step}')\n    elif args.auto_resume:"
    )
    content = content.replace(
        "elif args.resume_from:\n        global_step = checkpointer.load(args.resume_from)",
        ""
    )
    
if "checkpointer.register_args" not in content:
    content = content.replace(
        "checkpointer.register_model(\"thought_engine\", model)",
        "checkpointer.register_model(\"thought_engine\", model)\n    checkpointer.register_args(args)"
    )

if "wrap_dataloader" not in content:
    content = content.replace(
        "for batch_idx, (w_t, w_tgt) in enumerate(dataloader):",
        "resumed_loader = checkpointer.wrap_dataloader(dataloader, epoch, global_step)\n        for batch_idx, (w_t, w_tgt) in enumerate(resumed_loader):"
    )

with open("training/train_phase2_thoughtengine.py", "w") as f:
    f.write(content)
print("Patched Phase 2 Thought Engine")
