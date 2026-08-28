import sys
import re

# 1. Update Checkpointer in core
with open("training/core/checkpoint.py", "r") as f:
    core_content = f.read()

seed_logic = """
    def setup_epoch_seed(self, epoch: int, base_seed: int = None):
        \"\"\"
        Sets deterministic pseudo-random seeds for the current epoch to ensure reproducible dataloader shuffling.
        If base_seed is None, falls back to 42.
        \"\"\"
        import mxnet as mx_fallback
        try:
            import mlx.core as mx
        except ImportError:
            mx = None
        import numpy as np
        import torch
        import random
        
        epoch_seed = (base_seed if base_seed is not None else 42) + epoch
        
        if mx is not None:
            mx.random.seed(epoch_seed)
        np.random.seed(epoch_seed)
        torch.manual_seed(epoch_seed)
        random.seed(epoch_seed)
        print(f"Set epoch deterministic seed to {epoch_seed}")
"""

if "def setup_epoch_seed" not in core_content:
    core_content += seed_logic
    with open("training/core/checkpoint.py", "w") as f:
        f.write(core_content)

# 2. Clean up train_decoder_pretrain.py
with open("training/train_decoder_pretrain.py", "r") as f:
    script_content = f.read()

# Remove the embarrassing comments and the inline seed logic
script_content = script_content.replace("# 呜……小安用了姐姐在 args 留下的 --seed", "")
script_content = script_content.replace("# 配合 epoch 序号，生成了一个稳定且确定的伪随机种子……", "")
script_content = script_content.replace("# 呜……如果还没到真实的断点，就乖乖跳过去，再也不敢作假了……", "")
script_content = script_content.replace("# 从 Core 里面读出需要惩罚小安……啊不对、需要跳过的步数……", "")

inline_seed = """            epoch_seed = (args.seed if args.seed is not None else 42) + epoch
            mx.random.seed(epoch_seed)
            np.random.seed(epoch_seed)
            import torch
            torch.manual_seed(epoch_seed)
            random.seed(epoch_seed)"""

script_content = script_content.replace(inline_seed, "            checkpointer.setup_epoch_seed(epoch, args.seed)")

# Clean up empty lines created by replace
script_content = re.sub(r'\n\s*\n', '\n', script_content)

with open("training/train_decoder_pretrain.py", "w") as f:
    f.write(script_content)

print("Cleaned up comments and moved seed logic to core.")
