import os
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pandas as pd
import wandb
import sys
import numpy as np
import json
import random
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from training.core.char_tokenizer import CharTokenizer
from training.core.args import get_training_parser
from training.core.checkpoint import Checkpointer
from model.latedecoder import LateFusionDecoder
class MixedDataset(Dataset):
    def __init__(self, jsonl_path, parquet_path, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        print(f"Loading dialogue data from {jsonl_path}...")
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                item = json.loads(line)
                self.data.append(str(item))
        print(f"Loading omni data from {parquet_path}...")
        df = pd.read_parquet(parquet_path)
        for chunks in df['chunks']:
            text = "".join([str(c).replace('<EOS>', '') for c in chunks])
            self.data.append(text)
        print(f"Loaded {len(self.data)} total samples.")
        random.shuffle(self.data)
        print("Data shuffled.")
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        text = self.data[idx]
        tokens = [self.tokenizer.bos_token_id] + self.tokenizer.encode(text, add_special_tokens=False) + [self.tokenizer.eos_token_id]
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
            tokens[-1] = self.tokenizer.eos_token_id
        pad_len = self.max_length - len(tokens)
        x = tokens.copy()
        if pad_len > 0:
            x = x + [self.tokenizer.pad_token_id] * pad_len
        return np.array(x, dtype=np.int32)
def loss_fn(model, x, y, mock_z):
    logits = model(x, mock_z)
    B, L, V = logits.shape
    logits = logits.reshape(B * L, V)
    y = y.reshape(B * L)
    losses = nn.losses.cross_entropy(logits, y)
    mask = (y != -100)
    loss = mx.sum(losses * mask) / mx.sum(mask)
    return loss

def worker_init_fn(worker_id):
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def main():
    parser = get_training_parser("Decoder Pretrain (Supports Phase 1.1 Global & Phase 1.2 Local)")
    parser.add_argument("--window_size", type=int, default=None, help="Local attention window size (default: None for full causal)")
    parser.add_argument("--global_attention_ratio", type=float, default=0.0, help="Probability of using global attention instead of windowed attention per batch.")
    args = parser.parse_args()
    checkpointer = Checkpointer(out_dir=args.out_dir, prefix=args.ckpt_prefix, keep_last_k=args.keep_last_k)
    resume_ckpt_path = None
    if args.auto_resume:
        resume_ckpt_path = checkpointer.get_latest_checkpoint_path()
    elif args.resume_from:
        resume_ckpt_path = args.resume_from
    if resume_ckpt_path:
        args = checkpointer.load_args(resume_ckpt_path, args)
    tokenizer = CharTokenizer()
    dataset = MixedDataset("data/Basic_ZH/dialogues_254k_clean.jsonl", "data/Basic_ZH/chunked_mixed_omni.parquet", tokenizer)


    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=4, worker_init_fn=worker_init_fn)
    model = LateFusionDecoder(
        vocab_size=tokenizer.vocab_size,
        embed_dim=1024,
        hidden_dim=1024,
        num_layers=8,
        num_heads=16,
        window_size=args.window_size
    )
    optimizer = optim.AdamW(learning_rate=args.lr)
    checkpointer.register_model("decoder", model)
    checkpointer.register_optimizer("decoder_opt", optimizer)
    checkpointer.register_args(args)
    global_step = 0
    if resume_ckpt_path:
        global_step = checkpointer.load(resume_ckpt_path)
    start_epoch = global_step // len(dataloader) if len(dataloader) > 0 else 0
    step = global_step
    mx.eval(model.parameters())
    wandb.init(project="antigravity_phase1_1", name=args.ckpt_prefix)
    os.makedirs(args.out_dir, exist_ok=True)
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    try:
        for epoch in range(start_epoch, args.epochs):
            # Seed the dataloader for this epoch to ensure reproducible shuffling
            # If a global seed is provided, use it. Otherwise, fallback to a stable deterministic seed.
            checkpointer.setup_epoch_seed(epoch, args.seed)
            # Calculate how many steps to skip if we resumed mid-epoch
            skip_steps = 0
            if epoch == start_epoch and resume_ckpt_path:
                skip_steps = checkpointer.get_skip_steps(global_step, len(dataloader))
                if skip_steps > 0:
                    print(f"Resuming mid-epoch: skipping first {skip_steps} batches of Epoch {epoch+1}")
            model.train()
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
            for step_in_epoch, x_np in enumerate(pbar):
                if skip_steps > 0 and step_in_epoch < skip_steps:
                    continue
                
                if args.window_size is not None:
                    if random.random() < args.global_attention_ratio:
                        model.window_size = None
                    else:
                        model.window_size = args.window_size

                x_np_full = np.array(x_np)
                y_np = x_np_full[:, 1:].copy()
                y_np[y_np == dataset.tokenizer.pad_token_id] = -100
                x = mx.array(x_np_full)
                y = mx.array(y_np)
                mock_z = mx.zeros((x.shape[0], x.shape[1] - 1, 1024))
                loss, grads = loss_and_grad_fn(model, x, y, mock_z)
                optimizer.update(model, grads)
                mx.eval(model.parameters(), optimizer.state)
                loss_val = loss.item()
                wandb.log({"loss": loss_val, "step": step})
                pbar.set_postfix({"loss": f"{loss_val:.4f}"})
                step += 1
                if step % args.save_steps == 0:
                    checkpointer.save(step)
            # Checkpoint at epoch end
            checkpointer.save(step)
    except KeyboardInterrupt:
        print(f"\nTraining interrupted manually at step {step}. Saving emergency checkpoint...")
        checkpointer.save(step)
        print("Emergency checkpoint saved successfully!")
    wandb.finish()
if __name__ == '__main__':
    main()
