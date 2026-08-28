import sys
import os

with open("training/core/checkpoint.py", "r") as f:
    content = f.read()

target_logic = """
        current_args_dict = vars(current_args)
        
        # Parse raw sys.argv to find which arguments were explicitly provided by the user
        import sys
        explicit_args = set()
        for arg in sys.argv[1:]:
            if arg.startswith("--"):
                explicit_args.add(arg[2:].split('=')[0])
                
        overridden_keys = []
        preserved_keys = []
        
        for k, v in saved_args_dict.items():
            if k in current_args_dict and current_args_dict[k] != v:
                if k in explicit_args:
                    preserved_keys.append(f"{k}: {v} (saved) -> {current_args_dict[k]} (CLI override)")
                else:
                    overridden_keys.append(f"{k}: {current_args_dict[k]} (default) -> {v} (saved)")
                    current_args_dict[k] = v
                    
        if overridden_keys:
            print("Restored the following arguments from checkpoint:")
            for msg in overridden_keys:
                print(f"  {msg}")
                
        if preserved_keys:
            print("Preserved the following command-line arguments (overriding checkpoint):")
            for msg in preserved_keys:
                print(f"  {msg}")
"""

# We need to replace the old load_args inner loop
import re
pattern = r'current_args_dict = vars\(current_args\).*?current_args_dict\[k\] = v'
content = re.sub(pattern, target_logic.strip(), content, flags=re.DOTALL)

with open("training/core/checkpoint.py", "w") as f:
    f.write(content)
print("Patched Checkpointer to respect CLI args")
