import sys

with open("training/train_decoder_pretrain.py", "r") as f:
    content = f.read()

# Replace the messy loop logic
old_loop_start = """for epoch in range(start_epoch, args.epochs):
            checkpointer.setup_epoch_seed(epoch, args.seed)
            
            skip_steps = 0
            if epoch == start_epoch and resume_ckpt_path:
                skip_steps = checkpointer.get_skip_steps(global_step, len(dataloader))
                if skip_steps > 0:
                    print(f"Resuming mid-epoch: skipping first {skip_steps} batches of Epoch {epoch+1}")
            
            model.train()
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
            for step_in_epoch, x_np in enumerate(pbar):
                if skip_steps > 0 and step_in_epoch < skip_steps:
                    continue"""
                    
new_loop_start = """for epoch in range(start_epoch, args.epochs):
            model.train()
            
            # Use Core to transparently handle seeds and mid-epoch batch skipping
            epoch_dataloader = checkpointer.wrap_dataloader(dataloader, epoch, global_step)
            
            pbar = tqdm(epoch_dataloader, desc=f"Epoch {epoch+1}")
            for x_np in pbar:"""

if "checkpointer.setup_epoch_seed(epoch, args.seed)" in content:
    content = content.replace(old_loop_start, new_loop_start)
else:
    print("Warning: old loop not found exactly.")

with open("training/train_decoder_pretrain.py", "w") as f:
    f.write(content)
print("Updated pretrain script to use wrap_dataloader")
