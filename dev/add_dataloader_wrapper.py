import sys

with open("training/core/checkpoint.py", "r") as f:
    content = f.read()

wrapper_logic = """

class ResumedDataLoader:
    \"\"\"Wraps a dataloader to transparently skip a specified number of batches and maintain __len__.\"\"\"
    def __init__(self, dataloader, skip_steps):
        self.dataloader = dataloader
        self.skip_steps = skip_steps

    def __iter__(self):
        it = iter(self.dataloader)
        for _ in range(self.skip_steps):
            try:
                next(it)
            except StopIteration:
                break
        return it

    def __len__(self):
        # We return the original length so tqdm displays correctly, 
        # though ideally it could be max(0, len(self.dataloader) - self.skip_steps)
        # Keeping original length keeps global epoch math consistent in some scripts.
        return len(self.dataloader)

    def set_epoch(self, epoch):
        if hasattr(self.dataloader, 'sampler') and hasattr(self.dataloader.sampler, 'set_epoch'):
            self.dataloader.sampler.set_epoch(epoch)
"""

if "class ResumedDataLoader:" not in content:
    # Insert it before the Checkpointer class
    idx = content.find("class Checkpointer:")
    content = content[:idx] + wrapper_logic + "\n" + content[idx:]

# Now add wrap_dataloader to Checkpointer
wrap_logic = """
    def wrap_dataloader(self, dataloader, epoch: int, global_step: int):
        \"\"\"
        Sets the seed for the epoch and returns a wrapped dataloader that automatically skips 
        the already processed batches if resuming mid-epoch.
        \"\"\"
        base_seed = self._args.seed if hasattr(self._args, 'seed') else None
        self.setup_epoch_seed(epoch, base_seed)
        
        # Only skip if we are on the exact start_epoch that we resumed from
        start_epoch = global_step // len(dataloader) if len(dataloader) > 0 else 0
        skip_steps = 0
        if epoch == start_epoch:
            skip_steps = self.get_skip_steps(global_step, len(dataloader))
            
        if skip_steps > 0:
            print(f"Checkpointer: Auto-skipping first {skip_steps} batches of Epoch {epoch+1}")
            return ResumedDataLoader(dataloader, skip_steps)
        return dataloader
"""

if "def wrap_dataloader" not in content:
    content += wrap_logic

with open("training/core/checkpoint.py", "w") as f:
    f.write(content)
print("Added ResumedDataLoader and wrap_dataloader to Checkpointer")
