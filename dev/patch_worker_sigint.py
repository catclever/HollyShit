import sys

with open("training/train_decoder_pretrain.py", "r") as f:
    content = f.read()

worker_init_logic = """
def worker_init_fn(worker_id):
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=4, worker_init_fn=worker_init_fn)
"""

if "worker_init_fn" not in content:
    content = content.replace(
        "dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=4)",
        "def worker_init_fn(worker_id):\n        import signal\n        signal.signal(signal.SIGINT, signal.SIG_IGN)\n\n    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=4, worker_init_fn=worker_init_fn)"
    )
    with open("training/train_decoder_pretrain.py", "w") as f:
        f.write(content)
    print("Patched DataLoader to ignore SIGINT in workers")
else:
    print("Already patched")
