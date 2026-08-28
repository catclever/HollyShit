import sys

with open("training/train_phase1_1_omni.py", "r") as f:
    content = f.read()

# We need to wrap the epoch loop in a try-except block
old_loop = """
    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for x_np in pbar:
"""

new_loop = """
    try:
        for epoch in range(args.epochs):
            model.train()
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
            for x_np in pbar:
"""

if old_loop in content:
    content = content.replace(old_loop, new_loop)
    
    # Now we need to add the except block at the end of the file, right before wandb.finish()
    old_finish = """
        model.save_weights(f"{args.out_dir}/omni_epoch_{epoch+1}.safetensors")
        
    wandb.finish()
"""
    
    new_finish = """
            model.save_weights(f"{args.out_dir}/omni_epoch_{epoch+1}.safetensors")
            
    except KeyboardInterrupt:
        print(f"\nTraining interrupted manually at step {step}. Saving emergency checkpoint...")
        model.save_weights(f"{args.out_dir}/omni_emergency_step_{step}.safetensors")
        print("Emergency checkpoint saved successfully!")
        
    wandb.finish()
"""
    content = content.replace(old_finish, new_finish)
    
    # Also indent the inner loop logic
    lines = content.split('\n')
    in_try = False
    for i, line in enumerate(lines):
        if "for epoch in range(args.epochs):" in line:
            in_try = True
        elif "except KeyboardInterrupt:" in line:
            in_try = False
            
        if in_try and i > lines.index("    try:"):
            # If it's not empty, add 4 spaces
            if line.strip():
                lines[i] = "    " + line

    with open("training/train_phase1_1_omni.py", "w") as f:
        f.write('\n'.join(lines))
    print("Patched successfully!")
else:
    print("Could not find the loop to patch.")
