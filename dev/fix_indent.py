import sys

with open("training/train_phase1_1_omni.py", "r") as f:
    lines = f.readlines()

new_lines = []
for i, line in enumerate(lines):
    if "model.save_weights(f\"{args.out_dir}/omni_epoch_{epoch+1}.safetensors\")" in line:
        # Revert the wrong indentation from the previous patch (it was indented into the inner loop)
        new_lines.append('        model.save_weights(f"{args.out_dir}/omni_epoch_{epoch+1}.safetensors")\n')
    else:
        new_lines.append(line)

with open("training/train_phase1_1_omni.py", "w") as f:
    f.writelines(new_lines)
print("Indentation fixed.")
