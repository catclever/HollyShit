import sys

with open("training/train_phase3_1_latefusion.py", "r") as f:
    content = f.read()

# Add get_training_parser if not present
if "get_training_parser" not in content:
    content = content.replace(
        "from training.core.checkpoint import Checkpointer",
        "from training.core.checkpoint import Checkpointer\nfrom training.core.args import get_training_parser"
    )

# Fix parser
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
    parser.add_argument("--encoder_ckpt", type=str, default="checkpoints/basencoder_step_115000/student.pt")
    parser.add_argument("--decoder_ckpt", type=str, default=None, help="Path to Phase 2.1 pretrained decoder safetensors")
    
    args = parser.parse_args()"""

if old_parser in content:
    content = content.replace(old_parser, new_parser)

# Fix models loading
old_load = """    # Load frozen encoder weights
    load_torch_weights_into_mlx(encoder, args.encoder_ckpt)
    
    # Disable gradient tracking on encoder entirely
    encoder.freeze()
    
    # Checkpointer
    checkpointer = Checkpointer(out_dir=args.out_dir, prefix=args.ckpt_prefix, keep_last_k=5)
    checkpointer.register_model("fusion_decoder", decoder)
    checkpointer.register_optimizer("fusion_opt", optimizer)
    
    global_step = 0
    if args.continue_from:
        global_step = checkpointer.load(args.continue_from)
    elif args.resume:
        global_step = checkpointer.load_latest()"""

new_load = """    # Load frozen encoder weights
    load_torch_weights_into_mlx(encoder, args.encoder_ckpt)
    
    # Disable gradient tracking on encoder entirely
    encoder.freeze()
    
    if args.decoder_ckpt:
        print(f"Loading pretrained Phase 2.1 Decoder from {args.decoder_ckpt}")
        decoder.load_weights(args.decoder_ckpt)
        # Optional: Freeze the core decoder blocks to only train fusion_mlp
        # for block in decoder.blocks:
        #     block.freeze()
        # decoder.char_embed.freeze()
    
    # Checkpointer
    checkpointer = Checkpointer(out_dir=args.out_dir, prefix=args.ckpt_prefix, keep_last_k=args.keep_last_k)
    checkpointer.register_model("fusion_decoder", decoder)
    checkpointer.register_optimizer("fusion_opt", optimizer)
    checkpointer.register_args(args)
    
    global_step = 0
    resume_ckpt_path = None
    if args.auto_resume:
        resume_ckpt_path = checkpointer.get_latest_checkpoint_path()
    elif args.resume_from:
        resume_ckpt_path = args.resume_from
        
    if resume_ckpt_path:
        args = checkpointer.load_args(resume_ckpt_path, args)
        global_step = checkpointer.load(resume_ckpt_path)"""

if old_load in content:
    content = content.replace(old_load, new_load)
    
# Fix loop logic
old_loop = """    for epoch in range(args.epochs):
        for batch_idx, (input_ids, _) in enumerate(dataloader):"""

new_loop = """    start_epoch = global_step // len(dataloader) if len(dataloader) > 0 else 0
    for epoch in range(start_epoch, args.epochs):
        checkpointer.setup_epoch_seed(epoch, args.seed)
        resumed_loader = checkpointer.wrap_dataloader(dataloader, epoch, global_step)
        for batch_idx, (input_ids, _) in enumerate(resumed_loader):"""
        
if old_loop in content:
    content = content.replace(old_loop, new_loop)
    
# Fix worker_init_fn
bad_fn = """    def worker_init_fn(worker_id):
        import signal
        signal.signal(signal.SIGINT, signal.SIG_IGN)"""
if bad_fn in content:
    content = content.replace(bad_fn, "")
    good_fn = "def worker_init_fn(worker_id):\n    import signal\n    signal.signal(signal.SIGINT, signal.SIG_IGN)\n\ndef train():"
    content = content.replace("def train():", good_fn)

# Add worker_init_fn to DataLoader if missing
if "worker_init_fn" not in content and "DataLoader" in content:
    content = content.replace("persistent_workers=True)", "persistent_workers=True, worker_init_fn=worker_init_fn)")

with open("training/train_phase3_1_latefusion.py", "w") as f:
    f.write(content)
print("Patched P3.1 script")
