import argparse
import sys
import os
import torch
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from distilled_emb.model_cuda import TinyCharEncoderCUDA
from training.core.char_tokenizer import CharTokenizer
from training.core.dataloader import TextDocumentDataLoader
from safetensors.torch import load_file

def main():
    parser = argparse.ArgumentParser(description="Extract Mean and Std of GodEncoder/TinyCharEncoder Latent Space")
    parser.add_argument("--data_path", type=str, default="data/Basic_ZH/chunked_mixed_omni.parquet", help="Path to the training data.")
    parser.add_argument("--tinybert_ckpt", type=str, default="checkpoints/distilled/tinybert_pt_v1_step_100000", help="Path to frozen Phase 0.5 distilled TinyBERT.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_episode_len", type=int, default=256)
    parser.add_argument("--num_batches", type=int, default=1000, help="How many batches to process for statistics calculation.")
    parser.add_argument("--out_file", type=str, default="checkpoints/z_space_stats.pt", help="Where to save the computed statistics.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    tokenizer = CharTokenizer()
    dataloader = TextDocumentDataLoader(
        parquet_path=args.data_path,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_episode_len=args.max_episode_len,
        backend='torch'
    )
    
    # Load TinyCharEncoder
    ckpt_path = args.tinybert_ckpt
    if not ckpt_path.endswith(".pt") and not ckpt_path.endswith(".safetensors"):
        if os.path.exists(f"{ckpt_path}/student.pt"):
            ckpt_path = f"{ckpt_path}/student.pt"
        elif os.path.exists(f"{ckpt_path}/model.safetensors"):
            ckpt_path = f"{ckpt_path}/model.safetensors"
        else:
            raise FileNotFoundError(f"Could not find student.pt or model.safetensors in {args.tinybert_ckpt}.")

    if ckpt_path.endswith(".safetensors"):
        weights = load_file(ckpt_path)
    else:
        weights = torch.load(ckpt_path, map_location='cpu', weights_only=True)

    try:
        sniffed_z_dim = weights['out_proj.weight'].shape[0]
        print(f"[Auto-Sniff] Successfully sniffed Z_dim = {sniffed_z_dim} from checkpoint!")
    except KeyError:
        raise ValueError("Cannot find 'out_proj.weight' in the checkpoint. Is this a valid TinyCharEncoder?")
    
    tiny_encoder = TinyCharEncoderCUDA(vocab_size=tokenizer.vocab_size, z_dim=sniffed_z_dim)
    tiny_encoder.load_state_dict(weights, strict=False)
    tiny_encoder.eval().to(device)
    
    # Extract Features
    print(f"Extracting features from {args.num_batches} batches...")
    
    # We will compute running mean and variance to save memory
    n_samples = 0
    running_mean = torch.zeros(sniffed_z_dim, dtype=torch.float64, device=device)
    running_M2 = torch.zeros(sniffed_z_dim, dtype=torch.float64, device=device)
    
    with torch.no_grad():
        for i, (ids_t, att_t, sen_t) in enumerate(tqdm(dataloader, total=args.num_batches)):
            if i >= args.num_batches:
                break
                
            ids_t, att_t, sen_t = ids_t.to(device), att_t.to(device), sen_t.to(device)
            B, S, T = ids_t.shape
            
            flat_ids = ids_t.view(-1, T)
            flat_att = att_t.view(-1, T)
            flat_sen = sen_t.view(-1)
            
            with torch.cuda.amp.autocast():
                z_flat = tiny_encoder(flat_ids, attention_mask=flat_att)
                
            # Filter valid sentences
            valid_z = z_flat[flat_sen != 0].to(torch.float64) # (N, D)
            
            # Welford's Online Algorithm for running mean and variance
            for z in valid_z:
                n_samples += 1
                delta = z - running_mean
                running_mean += delta / n_samples
                delta2 = z - running_mean
                running_M2 += delta * delta2
                
    if n_samples < 2:
        print("Not enough samples to compute variance.")
        return
        
    variance = running_M2 / (n_samples - 1)
    std_dev = torch.sqrt(variance)
    
    # Convert back to float32
    mean_z = running_mean.to(torch.float32).cpu()
    std_z = std_dev.to(torch.float32).cpu()
    
    print("\n--- Statistics Summary ---")
    print(f"Total Valid Samples Analyzed: {n_samples}")
    print(f"Global Mean range: {mean_z.min().item():.4f} to {mean_z.max().item():.4f}")
    print(f"Global Std range:  {std_z.min().item():.4f} to {std_z.max().item():.4f}")
    print(f"Average Variance (Expected MSE baseline): {variance.mean().item():.4f}")
    print("--------------------------\n")
    
    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    stats = {
        'mean': mean_z,
        'std': std_z,
        'z_dim': sniffed_z_dim,
        'n_samples': n_samples
    }
    torch.save(stats, args.out_file)
    print(f"Successfully saved statistics to {args.out_file}")
    print("You can now run Phase 1.1 with Anisotropic Gaussian Prior!")

if __name__ == "__main__":
    main()
