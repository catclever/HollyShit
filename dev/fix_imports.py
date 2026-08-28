import sys

with open("training/train_phase3_1_latefusion.py", "r") as f:
    content = f.read()

path_fix = """import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
"""

if "sys.path.insert" not in content:
    # insert after standard imports
    if "import os" in content:
        content = content.replace("import os\n", "import os\n" + path_fix)
    
with open("training/train_phase3_1_latefusion.py", "w") as f:
    f.write(content)
