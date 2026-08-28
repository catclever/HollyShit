with open("training/train_phase1_1_decoder_pretrain.py", "r") as f:
    code = f.read()

import re

# We need to completely rewrite the training loop to use MLX.
new_code = """import os
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json
import wandb
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from training.core.char_tokenizer import CharTokenizer
from training.core.args import get_training_parser
from model.latedecoder import LateFusionDecoder

class DialogueDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        print(f"Loading data from {jsonl_path}...")
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                item = json.loads(line)
                if isinstance(item, dict):
                    text = item.get('dialogue', item.get('text', str(item)))
                    if isinstance(text, list):
                        text = " ".join(text)
                else:
                    text = str(item)
                self.data.append(text)
        print(f"Loaded {len(self.data)} samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
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
            
        return np.array(x, dtype=np.int32), np.array(y, dtype=np.int32)

def loss_fn(model, x, y, mock_z):
    logits = model(x, mock_z)
    B, L, V = logits.shape
    logits = logits.reshape(B * L, V)
    y = y.reshape(B * L)
    
    # MLX Cross Entropy
    losses = nn.losses.cross_entropy(logits, y)
    
    # Mask out ignore_index (-100)
    mask = (y != -100)
    loss = mx.sum(losses * mask) / mx.sum(mask)
    return loss

def main():
    parser = get_training_parser("Phase 1.1 Decoder Grammar Pretrain")
    args = parser.parse_args()

    tokenizer = CharTokenizer()
    dataset = DialogueDataset("data/Basic_ZH/dialogues_254k.jsonl", tokenizer)
    # Using PyTorch DataLoader for multiprocessing
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=4)

    model = LateFusionDecoder(
        vocab_size=tokenizer.vocab_size,
        embed_dim=1024,
        hidden_dim=1024,
        num_layers=8,
        num_heads=16
    )
    
    mx.eval(model.parameters())

    optimizer = optim.AdamW(learning_rate=args.lr)
    
    wandb.init(project="antigravity_phase1_1", name=args.ckpt_prefix)

    os.makedirs(args.out_dir, exist_ok=True)

    step = 0
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    
    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for x_np, y_np in pbar:
            x = mx.array(x_np)
            y = mx.array(y_np)
            
            # mock_z for target concept aligned with output (L length)
            mock_z = mx.zeros((x.shape[0], x.shape[1], 1024))
            
            loss, grads = loss_and_grad_fn(model, x, y, mock_z)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            
            loss_val = loss.item()
            wandb.log({"loss": loss_val, "step": step})
            pbar.set_postfix({"loss": f"{loss_val:.4f}"})
            step += 1
            
            if step % args.save_steps == 0:
                model.save_weights(f"{args.out_dir}/step_{step}.safetensors")
                
        model.save_weights(f"{args.out_dir}/epoch_{epoch+1}.safetensors")
        
    wandb.finish()

if __name__ == '__main__':
    main()
"""

with open("training/train_phase1_1_decoder_pretrain.py", "w") as f:
    f.write(new_code)
