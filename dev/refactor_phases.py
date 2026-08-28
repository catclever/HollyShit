import os
import glob

# Rename map for training scripts
script_renames = {
    "training/train_decoder_pretrain.py": "training/train_phase2_1_decoder.py",
    "training/train_phase1_6_latefusion.py": "training/train_phase3_1_latefusion.py",
    "training/train_phase1_5_conceptdecoder.py": "training/train_phase3_2_crossattn.py",
    "training/train_phase1_5_conceptdecoder_cuda.py": "training/train_phase3_2_crossattn_cuda.py",
    "training/train_phase2_thoughtengine.py": "training/train_phase4_thoughtengine.py",
}

# Rename map for models (removing phase prefixes for shared models)
model_renames = {
    "model/phase1_6_latedecoder.py": "model/latedecoder.py",
    "model/phase1_5_conceptdecoder.py": "model/crossattn_decoder.py",
}

import_replacements = {
    "model.phase1_6_latedecoder": "model.latedecoder",
    "model.phase1_5_conceptdecoder": "model.crossattn_decoder",
    "train_decoder_pretrain.py": "train_phase2_1_decoder.py",
    "train_phase1_6_latefusion.py": "train_phase3_1_latefusion.py",
    "train_phase1_5_conceptdecoder": "train_phase3_2_crossattn",
    "train_phase2_thoughtengine": "train_phase4_thoughtengine"
}

for old, new in {**script_renames, **model_renames}.items():
    if os.path.exists(old):
        os.rename(old, new)
        print(f"Renamed: {old} -> {new}")

# Update imports in all python scripts
py_files = glob.glob("**/*.py", recursive=True)
for fpath in py_files:
    if "dev/" in fpath or ".venv" in fpath:
        continue
    with open(fpath, 'r') as f:
        content = f.read()
    
    new_content = content
    for old_imp, new_imp in import_replacements.items():
        new_content = new_content.replace(old_imp, new_imp)
        
    if new_content != content:
        with open(fpath, 'w') as f:
            f.write(new_content)
        print(f"Updated imports in: {fpath}")

print("Refactor complete.")
