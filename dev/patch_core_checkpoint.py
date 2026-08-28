import sys

with open("training/core/checkpoint.py", "r") as f:
    content = f.read()

# Add get_skip_steps method to Checkpointer
skip_logic = """        if step == 0:
            import re
            m = re.search(r'step_(\d+)', load_path)
            if m:
                step = int(m.group(1))
        
        return step

    def get_skip_steps(self, resumed_step: int, dataloader_len: int) -> int:
        \"\"\"
        Calculates how many batches to skip in the current epoch when resuming.
        Assumes epoch = resumed_step // dataloader_len
        \"\"\"
        if dataloader_len == 0:
            return 0
        return resumed_step % dataloader_len
"""

if "def get_skip_steps" not in content:
    content = content.replace(
        "        if step == 0:\n            import re\n            m = re.search(r'step_(\d+)', load_path)",
        skip_logic[:skip_logic.find("        if step == 0:")] + "        if step == 0:\n            import re\n            m = re.search(r'step_(\d+)', load_path)"
    )
    # The replace above is tricky, let's just append it to the class if not found
    
    if "def get_skip_steps" not in content:
        # Find the end of the class
        content += """

    def get_skip_steps(self, resumed_step: int, dataloader_len: int) -> int:
        \"\"\"
        Calculates how many batches to skip in the current epoch when resuming.
        \"\"\"
        if dataloader_len == 0:
            return 0
        return resumed_step % dataloader_len
"""
        
with open("training/core/checkpoint.py", "w") as f:
    f.write(content)
print("Patched Checkpointer with get_skip_steps")
