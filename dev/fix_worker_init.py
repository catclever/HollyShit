import sys

with open("training/train_decoder_pretrain.py", "r") as f:
    content = f.read()

# Remove the local worker_init_fn
bad_fn = """    def worker_init_fn(worker_id):
        import signal
        signal.signal(signal.SIGINT, signal.SIG_IGN)"""
content = content.replace(bad_fn, "")

# Add it at the module level
good_fn = """
def worker_init_fn(worker_id):
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def main():"""
if "def worker_init_fn" not in content[:content.find("def main()")]:
    content = content.replace("def main():", good_fn)

with open("training/train_decoder_pretrain.py", "w") as f:
    f.write(content)

# Do the same for latefusion if it has it
try:
    with open("training/train_phase1_6_latefusion.py", "r") as f:
        content_lf = f.read()
    if bad_fn in content_lf:
        content_lf = content_lf.replace(bad_fn, "")
        if "def worker_init_fn" not in content_lf[:content_lf.find("def main()")]:
            content_lf = content_lf.replace("def main():", good_fn)
        with open("training/train_phase1_6_latefusion.py", "w") as f:
            f.write(content_lf)
except Exception:
    pass

print("Fixed worker_init_fn scope")
