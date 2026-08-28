import sys

with open("model/phase1_6_latedecoder.py", "r") as f:
    content = f.read()

old_mask = """    def _create_causal_mask(self, L: int, dtype=mx.float32) -> mx.array:
        indices = mx.arange(L)
        mask = indices.reshape(L, 1) < indices.reshape(1, L)
        return mask.astype(dtype) * -1e9"""

new_mask = """    def _create_causal_mask(self, L: int, dtype=mx.float32, window_size: int = 15) -> mx.array:
        indices = mx.arange(L)
        causal_mask = indices.reshape(L, 1) < indices.reshape(1, L)
        window_mask = indices.reshape(1, L) <= (indices.reshape(L, 1) - window_size)
        mask = causal_mask | window_mask
        return mask.astype(dtype) * -1e9"""

if old_mask in content:
    content = content.replace(old_mask, new_mask)
    with open("model/phase1_6_latedecoder.py", "w") as f:
        f.write(content)
    print("Mask patched successfully.")
else:
    print("Could not find the mask method to patch.")
