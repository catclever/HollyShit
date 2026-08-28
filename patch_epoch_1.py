import re

with open('training/train_phase3_1_latefusion.py', 'r') as f:
    code = f.read()

# Replace `Epoch {epoch}` with `Epoch {epoch + 1}`
code = code.replace('print(f"Epoch {epoch} |', 'print(f"Epoch {epoch + 1} |')

with open('training/train_phase3_1_latefusion.py', 'w') as f:
    f.write(code)
