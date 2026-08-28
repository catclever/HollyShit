with open("training/train_phase1_1_decoder_pretrain.py", "r") as f:
    code = f.read()

import re

# We need to change __getitem__ to just return the full padded sequence
old_getitem = """    def __getitem__(self, idx):
        text = self.data[idx]
        tokens = [self.tokenizer.bos_token_id] + self.tokenizer.encode(text, add_special_tokens=False) + [self.tokenizer.eos_token_id]
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
            tokens[-1] = self.tokenizer.eos_token_id
            
        x = tokens[:-1]
        y = tokens[1:]
        
        pad_len = self.max_length - 1 - len(x)
        if pad_len > 0:
            x = x + [self.tokenizer.pad_token_id] * pad_len
            y = y + [-100] * pad_len
            
        return np.array(x, dtype=np.int32), np.array(y, dtype=np.int32)"""

new_getitem = """    def __getitem__(self, idx):
        text = self.data[idx]
        tokens = [self.tokenizer.bos_token_id] + self.tokenizer.encode(text, add_special_tokens=False) + [self.tokenizer.eos_token_id]
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
            tokens[-1] = self.tokenizer.eos_token_id
            
        pad_len = self.max_length - len(tokens)
        
        x = tokens.copy()
        y = tokens[1:].copy() + [-100]
        
        if pad_len > 0:
            x = x + [self.tokenizer.pad_token_id] * pad_len
            y = y[:-1] + [-100] * pad_len + [-100]
            
        return np.array(x, dtype=np.int32), np.array(y, dtype=np.int32)"""

code = code.replace(old_getitem, new_getitem)

# Now fix the training loop mock_z shape
old_loop = """            x = mx.array(x_np)
            y = mx.array(y_np)
            
            # mock_z for target concept aligned with output (L length)
            mock_z = mx.zeros((x.shape[0], x.shape[1], 1024))
            
            loss, grads = loss_and_grad_fn(model, x, y, mock_z)"""

new_loop = """            x = mx.array(x_np)
            y = mx.array(y_np)
            
            # mock_z for target concept aligned with output (L-1 length)
            mock_z = mx.zeros((x.shape[0], x.shape[1] - 1, 1024))
            
            loss, grads = loss_and_grad_fn(model, x, y, mock_z)"""

code = code.replace(old_loop, new_loop)

with open("training/train_phase1_1_decoder_pretrain.py", "w") as f:
    f.write(code)
