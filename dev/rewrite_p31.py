import sys

with open("training/train_phase3_1_latefusion.py", "r") as f:
    content = f.read()

# 1. Imports
if "from training.core.args import get_training_parser" not in content:
    content = content.replace(
        "from training.core.checkpoint import Checkpointer",
        "from training.core.checkpoint import Checkpointer\nfrom training.core.args import get_training_parser"
    )

# 2. Worker Init
bad_worker = "    def worker_init_fn(worker_id):\n        import signal\n        signal.signal(signal.SIGINT, signal.SIG_IGN)"
if bad_worker in content:
    content = content.replace(bad_worker, "")
    
if "def worker_init_fn" not in content:
    content = content.replace("def train():", "def worker_init_fn(worker_id):\n    import signal\n    signal.signal(signal.SIGINT, signal.SIG_IGN)\n\ndef train():")

# 3. Parser Replacement
old_parser = """    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--z_dim", type=int, default=1024)
    parser.add_argument("--decoder_layers", type=int, default=4)
    parser.add_argument("--decoder_heads", type=int, default=8)
    parser.add_argument("--noise_min", type=float, default=0.0)
    parser.add_argument("--noise_max", type=float, default=0.5)
    
    parser.add_argument("--data_dir", type=str, default="data/Basic_ZH")
    parser.add_argument("--encoder_ckpt", type=str, default="checkpoints/basencoder_step_115000/student.pt")
    
    parser.add_argument("--out_dir", type=str, default="checkpoints/phase1_6")
    parser.add_argument("--ckpt_prefix", type=str, default="latefusion")
    parser.add_argument("--save_steps", type=int, default=5000)
    parser.add_argument("--continue_from", type=str, default=None)
    parser.add_argument("--continue", dest="resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()"""

new_parser = """    parser = get_training_parser("Phase 3.1 Late Fusion Alignment")
    parser.add_argument("--noise_min", type=float, default=0.0)
    parser.add_argument("--noise_max", type=float, default=0.5)
    parser.add_argument("--encoder_ckpt", type=str, default="checkpoints/basencoder_step_115000/student.pt")
    parser.add_argument("--decoder_ckpt", type=str, default="checkpoints/phase1_2_decoder/partial_window_step_45000/decoder.safetensors")
    
    args = parser.parse_args()"""

content = content.replace(old_parser, new_parser)

# 4. Checkpointer Logic
old_ckpt_logic = """    # Checkpointing
    os.makedirs(args.out_dir, exist_ok=True)
    checkpointer = Checkpointer(args.out_dir, prefix=args.ckpt_prefix, keep_last_k=3)
    checkpointer.register_model("decoder", decoder)
    checkpointer.register_optimizer("optimizer", optimizer)
    checkpointer.register_args(args)"""

new_ckpt_logic = """    # Load Phase 2.1 Decoder weights BEFORE checkpointer load
    if args.decoder_ckpt:
        print(f"Loading pretrained Phase 2.1 Decoder from {args.decoder_ckpt}...")
        decoder.load_weights(args.decoder_ckpt)
        # Optional: We usually freeze the text generation parts to only train the fusion_mlp
        # If Kaelina wants it to fine-tune, she can comment this out.
        # for block in decoder.blocks:
        #     block.freeze()
        # decoder.char_embed.freeze()

    # Checkpointing
    os.makedirs(args.out_dir, exist_ok=True)
    checkpointer = Checkpointer(args.out_dir, prefix=args.ckpt_prefix, keep_last_k=args.keep_last_k)
    checkpointer.register_model("decoder", decoder)
    checkpointer.register_optimizer("optimizer", optimizer)
    checkpointer.register_args(args)"""

content = content.replace(old_ckpt_logic, new_ckpt_logic)

# 5. Resume Logic
old_resume = """    global_step = 0
    if args.continue_from:
        global_step = checkpointer.load(args.continue_from)
    elif args.resume:
        global_step = checkpointer.load_latest()"""

new_resume = """    global_step = 0
    resume_ckpt_path = None
    if args.auto_resume:
        resume_ckpt_path = checkpointer.get_latest_checkpoint_path()
    elif args.resume_from:
        resume_ckpt_path = args.resume_from
        
    if resume_ckpt_path:
        args = checkpointer.load_args(resume_ckpt_path, args)
        global_step = checkpointer.load(resume_ckpt_path)"""

content = content.replace(old_resume, new_resume)

# 6. Dataloader wrapper and seed
old_loop = """        start_epoch = global_step // steps_per_epoch
        for epoch in range(start_epoch, args.epochs):
            epoch_dataloader = checkpointer.wrap_dataloader(dataloader, epoch, global_step)
            for batch in epoch_dataloader:"""

new_loop = """        start_epoch = global_step // steps_per_epoch
        for epoch in range(start_epoch, args.epochs):
            checkpointer.setup_epoch_seed(epoch, args.seed)
            epoch_dataloader = checkpointer.wrap_dataloader(dataloader, epoch, global_step)
            for batch in epoch_dataloader:"""

content = content.replace(old_loop, new_loop)

with open("training/train_phase3_1_latefusion.py", "w") as f:
    f.write(content)
print("Complete!")
