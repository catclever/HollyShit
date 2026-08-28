with open("training/train_phase1_1_decoder_pretrain.py", "r") as f:
    code = f.read()

import re

# Simplify __getitem__ to just return padded tokens as 'x'
new_getitem = """    def __getitem__(self, idx):
        text = self.data[idx]
        tokens = [self.tokenizer.bos_token_id] + self.tokenizer.encode(text, add_special_tokens=False) + [self.tokenizer.eos_token_id]
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
            tokens[-1] = self.tokenizer.eos_token_id
            
        pad_len = self.max_length - len(tokens)
        
        x = tokens.copy()
        if pad_len > 0:
            x = x + [self.tokenizer.pad_token_id] * pad_len
            
        return np.array(x, dtype=np.int32)"""

# Use regex to replace the entire __getitem__
code = re.sub(r'    def __getitem__\(self, idx\):.*?(?=\n\n    def loss_fn|def loss_fn)', new_getitem + '\n\n', code, flags=re.DOTALL)

# Fix training loop to slice y correctly
old_loop_sig = """        for x_np, y_np in pbar:
            x = mx.array(x_np)
            y = mx.array(y_np)"""

new_loop_sig = """        for x_np in pbar:
            # The model shifts internally. So input is x, target is y = x[:, 1:]
            x_np_full = np.array(x_np)
            y_np = x_np_full[:, 1:].copy()
            # Replace pad_token_id with -100 in y
            y_np[y_np == dataset.tokenizer.pad_token_id] = -100
            
            x = mx.array(x_np_full)
            y = mx.array(y_np)"""

code = code.replace(old_loop_sig, new_loop_sig)

with open("training/train_phase1_1_decoder_pretrain.py", "w") as f:
    f.write(code)
