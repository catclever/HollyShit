import sys
import os
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from distilled_emb.model_cuda import TinyCharEncoderCUDA
from training.core.char_tokenizer import CharTokenizer
from training.core.dataloader import TextDocumentDataLoader
from safetensors.torch import load_file

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    tokenizer = CharTokenizer()
    dataloader = TextDocumentDataLoader(
        parquet_path="data/Basic_ZH/chunked_mixed_omni.parquet",
        tokenizer=tokenizer,
        batch_size=32,
        max_episode_len=128,
        backend='torch'
    )
    
    ckpt_path = "checkpoints/distilled/tinybert_pt_v1_step_100000"
    if os.path.exists(f"{ckpt_path}/model.safetensors"):
        weights = load_file(f"{ckpt_path}/model.safetensors")
    elif os.path.exists(f"{ckpt_path}/student.pt"):
        weights = torch.load(f"{ckpt_path}/student.pt", map_location='cpu', weights_only=True)
    else:
        print("Checkpoint not found!")
        return

    sniffed_z_dim = weights['out_proj.weight'].shape[0]
    print(f"Z_dim = {sniffed_z_dim}")
    
    tiny_encoder = TinyCharEncoderCUDA(vocab_size=tokenizer.vocab_size, z_dim=sniffed_z_dim)
    tiny_encoder.load_state_dict(weights, strict=False)
    tiny_encoder.eval().to(device)
    
    all_z = []
    
    print("Extracting features from a few batches...")
    with torch.no_grad():
        for i, (ids, att, sen) in enumerate(dataloader):
            if i >= 50: # Collect ~1.5k sequences
                break
            ids, att, sen = ids.to(device), att.to(device), sen.to(device)
            B, S, T = ids.shape
            
            flat_ids = ids.view(-1, T)
            flat_att = att.view(-1, T)
            flat_sen = sen.view(-1)
            
            z = tiny_encoder(flat_ids, attention_mask=flat_att)
            
            # Only keep valid sentences
            valid_z = z[flat_sen != 0]
            if len(valid_z) > 0:
                all_z.append(valid_z.cpu())
                
    if not all_z:
        print("No valid Z found!")
        return
        
    Z = torch.cat(all_z, dim=0)
    print(f"Collected {Z.shape[0]} valid semantic coordinates.")
    
    mean_z = torch.mean(Z, dim=0)
    var_z = torch.var(Z, dim=0)
    
    print(f"Global Mean range: {mean_z.min().item():.4f} to {mean_z.max().item():.4f}")
    print(f"Global Var range: {var_z.min().item():.4f} to {var_z.max().item():.4f}")
    print(f"Average Var across all dims: {var_z.mean().item():.4f}")
    
    # Simple PCA check using SVD
    U, S_vals, V = torch.svd(Z - mean_z)
    explained_variance = (S_vals ** 2) / (Z.shape[0] - 1)
    total_var = explained_variance.sum()
    explained_variance_ratio = explained_variance / total_var
    
    cum_var = torch.cumsum(explained_variance_ratio, dim=0)
    
    dims_for_95 = torch.searchsorted(cum_var, 0.95).item() + 1
    dims_for_99 = torch.searchsorted(cum_var, 0.99).item() + 1
    
    print(f"Total Dimensions: {sniffed_z_dim}")
    print(f"Dimensions needed for 95% variance: {dims_for_95}")
    print(f"Dimensions needed for 99% variance: {dims_for_99}")
    
    # Save statistics for phase 1.1
    stats = {
        'mean': mean_z,
        'std': torch.sqrt(var_z),
        'pca_components': V[:, :dims_for_99], # K top components
        'pca_mean': mean_z
    }
    torch.save(stats, "checkpoints/z_space_stats.pt")
    print("Saved statistics to checkpoints/z_space_stats.pt")

if __name__ == "__main__":
    main()
