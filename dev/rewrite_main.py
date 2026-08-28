import sys
import re

with open("training/train_decoder_pretrain.py", "r") as f:
    content = f.read()
    
# Import Checkpointer at the top if not present
if "from training.core.checkpoint import Checkpointer" not in content:
    content = content.replace("from training.core.args import get_training_parser", "from training.core.args import get_training_parser\nfrom training.core.checkpoint import Checkpointer")

# Find the start of main
main_start_idx = content.find("def main():")

new_main = """def main():
    parser = get_training_parser("Decoder Pretrain (Supports Phase 1.1 Global & Phase 1.2 Local)")
    parser.add_argument("--window_size", type=int, default=None, help="Local attention window size (default: None for full causal)")
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
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=4)

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
            model.train()
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
            for x_np in pbar:
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
        print(f"\\nTraining interrupted manually at step {step}. Saving emergency checkpoint...")
        checkpointer.save(step)
        print("Emergency checkpoint saved successfully!")
        
    wandb.finish()

if __name__ == '__main__':
    main()
"""

final_content = content[:main_start_idx] + new_main
with open("training/train_decoder_pretrain.py", "w") as f:
    f.write(final_content)
print("Rewrote main() using Checkpointer")
