import os
import shutil
import glob

ckpt_dir = "checkpoints/phase1_1_decoder"
safetensors_files = glob.glob(os.path.join(ckpt_dir, "*.safetensors"))

for file_path in safetensors_files:
    basename = os.path.basename(file_path)
    dir_name = basename.replace(".safetensors", "")
    new_dir_path = os.path.join(ckpt_dir, dir_name)
    
    os.makedirs(new_dir_path, exist_ok=True)
    
    new_file_path = os.path.join(new_dir_path, "decoder.safetensors")
    shutil.move(file_path, new_file_path)
    print(f"Migrated {basename} to {new_dir_path}/decoder.safetensors")

print("Migration complete!")
