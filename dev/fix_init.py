import sys

with open("model/phase1_6_latedecoder.py", "r") as f:
    content = f.read()
    
# Remove duplicate super().__init__()
content = content.replace("        super().__init__()\n        self.window_size = window_size\n        super().__init__()", "        self.window_size = window_size\n        super().__init__()")

with open("model/phase1_6_latedecoder.py", "w") as f:
    f.write(content)
