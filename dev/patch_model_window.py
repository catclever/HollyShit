import sys
import re

with open("model/phase1_6_latedecoder.py", "r") as f:
    content = f.read()

# 1. Update __init__ to accept window_size
old_init = """    def __init__(self, vocab_size: int, embed_dim: int = 1024, hidden_dim: int = 1024, num_layers: int = 8, num_heads: int = 16):
        super().__init__()"""
        
new_init = """    def __init__(self, vocab_size: int, embed_dim: int = 1024, hidden_dim: int = 1024, num_layers: int = 8, num_heads: int = 16, window_size: int = None):
        super().__init__()
        self.window_size = window_size"""

if old_init in content:
    content = content.replace(old_init, new_init)

# 2. Update _create_causal_mask to use self.window_size
old_mask = """    def _create_causal_mask(self, L: int, dtype=mx.float32, window_size: int = 15) -> mx.array:
        indices = mx.arange(L)
        causal_mask = indices.reshape(L, 1) < indices.reshape(1, L)
        window_mask = indices.reshape(1, L) <= (indices.reshape(L, 1) - window_size)
        mask = causal_mask | window_mask
        return mask.astype(dtype) * -1e9"""
        
new_mask = """    def _create_causal_mask(self, L: int, dtype=mx.float32) -> mx.array:
        indices = mx.arange(L)
        causal_mask = indices.reshape(L, 1) < indices.reshape(1, L)
        if self.window_size is not None:
            window_mask = indices.reshape(1, L) <= (indices.reshape(L, 1) - self.window_size)
            mask = causal_mask | window_mask
        else:
            mask = causal_mask
        return mask.astype(dtype) * -1e9"""

if old_mask in content:
    content = content.replace(old_mask, new_mask)

with open("model/phase1_6_latedecoder.py", "w") as f:
    f.write(content)
print("Model updated to support window_size in __init__.")
