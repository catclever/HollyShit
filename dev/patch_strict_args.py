import sys

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
        rejected_keys = []
        
        # Define arguments that are STRICTLY tied to model architecture or dataset shape
        # These CANNOT be changed during resume.
        STRICT_MODEL_ARGS = {"embed_dim", "hidden_dim", "num_layers", "num_heads", "window_size", "max_len"}
        
        for k, v in saved_args_dict.items():
            if k in current_args_dict and current_args_dict[k] != v:
                if k in explicit_args:
                    if k in STRICT_MODEL_ARGS:
                        # Reject user's attempt to override a strict model parameter during resume
                        rejected_keys.append(f"{k}: user requested {current_args_dict[k]}, but forced to keep saved {v}")
                        current_args_dict[k] = v
                    else:
                        # Allow user to override training hyperparameters (lr, batch_size, global_attention_ratio, etc)
                        preserved_keys.append(f"{k}: {v} (saved) -> {current_args_dict[k]} (CLI override)")
                else:
                    overridden_keys.append(f"{k}: {current_args_dict[k]} (default) -> {v} (saved)")
                    current_args_dict[k] = v
                    
        if overridden_keys:
            print("Restored the following arguments from checkpoint (defaults):")
            for msg in overridden_keys:
                print(f"  {msg}")
                
        if preserved_keys:
            print("Preserved the following command-line training arguments (overriding checkpoint):")
            for msg in preserved_keys:
                print(f"  {msg}")
                
        if rejected_keys:
            print("WARNING: Rejected attempt to change STRICT model parameters during resume:")
            for msg in rejected_keys:
                print(f"  {msg}")
"""

import re
pattern = r'current_args_dict = vars\(current_args\).*?if preserved_keys:\n.*?print\(f"  \{msg\}"\)'
content = re.sub(pattern, target_logic.strip(), content, flags=re.DOTALL)

with open("training/core/checkpoint.py", "w") as f:
    f.write(content)
print("Patched Checkpointer to enforce strict args")
