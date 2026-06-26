import torch
import math
from distilled_emb.model_cuda import TinyCharEncoderCUDA
model = TinyCharEncoderCUDA(vocab_size=10000, d_model=1024)
x = torch.randint(0, 1000, (1, 50))
out = model(x)
print(f"Variance of untinged output: {out.var().item():.4f}")
print(f"Variance scaled by 32: {(out * 32.0).var().item():.4f}")
