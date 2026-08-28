import glob
import re

scripts = glob.glob("training/train_*.py")

# We want to find loops like `for epoch in range(start_epoch, args.epochs):`
# and inside them, wrap the dataloader before it's used.
# Since usage varies, we can use regex to inject it right after the `for epoch in range(...):` line.

for script in scripts:
    if "pretrain" in script: continue # already did this
    if "phase0" in script: continue # maybe skip phase0 v2 for now
    
    with open(script, "r") as f:
        content = f.read()
        
    # Check if it has a dataloader
    if "dataloader =" not in content and "DataLoader(" not in content:
        continue
        
    # We need to inject epoch_dataloader = checkpointer.wrap_dataloader(dataloader, epoch, global_step)
    # and replace occurrences of `dataloader` with `epoch_dataloader` ONLY inside the epoch loop.
    # This is tricky with simple string replace. Let's do it carefully.
    
    # 1. phase 1.5 concept decoder
    if "train_phase1_5" in script:
        content = content.replace(
            "super_gen = super_batch_generator(dataloader, encode_step, chunker, tokenizer, args)",
            "epoch_dataloader = checkpointer.wrap_dataloader(dataloader, epoch, global_step)\n            super_gen = super_batch_generator(epoch_dataloader, encode_step, chunker, tokenizer, args)"
        )
        
    # 2. phase 1.6 late fusion
    elif "train_phase1_6_latefusion" in script:
        content = content.replace(
            "for batch in dataloader:",
            "epoch_dataloader = checkpointer.wrap_dataloader(dataloader, epoch, global_step)\n            for batch in epoch_dataloader:"
        )
        
    # 3. phase 1 charencoder
    elif "train_phase1_charencoder" in script:
        content = content.replace(
            "for step_in_epoch, (w_t,) in enumerate(dataloader):",
            "epoch_dataloader = checkpointer.wrap_dataloader(dataloader, epoch, global_step)\n            for step_in_epoch, (w_t,) in enumerate(epoch_dataloader):"
        )
        content = content.replace(
            "for step_in_epoch, (w_t, ) in enumerate(dataloader):",
            "epoch_dataloader = checkpointer.wrap_dataloader(dataloader, epoch, global_step)\n            for step_in_epoch, (w_t, ) in enumerate(epoch_dataloader):"
        )
        # some might have `for step, (w_t,) in enumerate(dataloader):`
        content = content.replace(
            "for step, (w_t,) in enumerate(dataloader):",
            "epoch_dataloader = checkpointer.wrap_dataloader(dataloader, epoch, global_step)\n            for step, (w_t,) in enumerate(epoch_dataloader):"
        )
        
    # 4. phase 2 thoughtengine
    elif "train_phase2_thoughtengine" in script:
        content = content.replace(
            "for batch in dataloader:",
            "epoch_dataloader = checkpointer.wrap_dataloader(dataloader, epoch, global_step)\n            for batch in epoch_dataloader:"
        )
        
    with open(script, "w") as f:
        f.write(content)
        print(f"Patched {script}")
