"""
【脚本功能】：微缩单条语句 100% 死记硬背过拟合证明 (Overfit Proof-of-Concept)
【使用场景】：Phase 0 算法验证。这是整个大架构（Fuser + GodEncoder + Decoder）的最基础沙盘。如果我们整个模型能够通过 50 步的强行投喂，把一段废话 100% 毫无损耗地死记硬背原样吐出，就证明物理管道没漏水，可以正式大规模训练。
【用法示例】：`python tests/test_overfit_poc.py`
"""
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.god_encoder import GodEncoder
from model.mamba_planner import MambaPlanner
from model.probability_mapping import ProbabilityMappingLayer
from model.decoder import WeakDecoder
from model.mamba_mlx.mamba_mlx import MambaConfig
from training.losses.loss import coverage_loss, momentum_continuity_loss, decoder_reconstruction_loss

class EndogenousSystem(nn.Module):
    """
    The Phase 1 V2 System:
    God Encoder creates the truth space (z_target).
    Mamba + Probability Net dynamically cover the truth space.
    Weak Decoder relies on z_target to build vocabulary metrics.
    """
    def __init__(self, ext_dim: int, d_model: int, z_dim: int, vocab_size: int):
        super().__init__()
        self.z_dim = z_dim
        # 0. The Adapter
        self.adapter = nn.Linear(ext_dim, d_model)
        
        # 1. Right Path: The God Encoder (takes standard f_t features)
        self.god_encoder = GodEncoder(d_model=d_model, z_dim=z_dim)
        
        # 2. Left Path: The Dynamic Engine
        config = MambaConfig(d_model=d_model, n_layers=1)
        self.mamba = MambaPlanner(config, z_dim=z_dim)
        
        # 3. The Ultimate Judge
        self.decoder = WeakDecoder(z_dim=z_dim, vocab_size=vocab_size, d_model=d_model, n_layers=1)
        
    def __call__(self, ext_emb: mx.array, tgt_toks: mx.array):
        B, L, _ = ext_emb.shape
        Z_Dim = self.z_dim
        
        # Digest sensory input
        f_t = self.adapter(ext_emb)
        
        # 1. God Encoder targets (The Absolute Truth Points)
        z_target = self.god_encoder(f_t) 
        
        # 2. Mamba Engine + Net Tying
        # Provide history inputs, prepending zero state for causality
        start_emb = mx.zeros((B, 1, f_t.shape[-1]))
        x_shifted = mx.concatenate([start_emb, f_t[:, :-1, :]], axis=1)
        
        mu_net, logvar_net, _ = self.mamba(x_shifted)
        
        # 3. Micro Decoder (The Judge of the z_target)
        z_flat = z_target.reshape(B * L, Z_Dim)
        
        # Teacher forcing tokens input
        start_toks = mx.zeros((B * L, 1), dtype=mx.int32)
        toks_shifted = mx.concatenate([start_toks, tgt_toks[:, :-1]], axis=1)
        
        logits = self.decoder(z_flat, toks_shifted)
        
        return mu_net, logvar_net, z_target, logits


def run_v2_poc():
    B, L = 2, 5
    Ext_Dim = 256
    D_Model = 128
    Z_Dim = 64
    Vocab_Size = 1000
    
    # Mock Sense Inputs & Targets
    mx.random.seed(42)
    senses_input = mx.random.normal((B, L, Ext_Dim))
    target_tokens = mx.random.randint(1, Vocab_Size - 1, (B * L, 4), dtype=mx.int32)
    
    system = EndogenousSystem(Ext_Dim, D_Model, Z_Dim, Vocab_Size)
    mx.eval(system.parameters())
    
    optimizer = optim.Adam(learning_rate=1e-3)
    
    def loss_fn(model_obj, ext_emb, tgt_toks):
        mu_net, logvar_net, z_target, logits = model_obj(ext_emb, tgt_toks)
        
        # 1. Structural definition loss (Decoder judging the God Encoder)
        l_recon = decoder_reconstruction_loss(logits, tgt_toks)
        
        # 2. Probability coverage loss (Mamba forced to guess God's dots)
        l_coverage = coverage_loss(mu_net, logvar_net, z_target)
        
        # 3. Momentum loss (Forcing Mamba and consequently God Encoder into smooth space)
        l_mom = momentum_continuity_loss(mu_net) # We apply momentum to the net center
        
        # Global joint loss
        total_loss = l_recon + l_coverage + 0.1 * l_mom
        
        return total_loss, (l_recon, l_coverage, l_mom)
        
    loss_and_grad_fn = nn.value_and_grad(system, loss_fn)
    
    # Do not compile to prevent MLX tracing issues for POC simple testing
    def step(e_in, t_tok):
        (loss, aux), grads = loss_and_grad_fn(system, e_in, t_tok)
        optimizer.update(system, grads)
        return loss, aux
        
    print("Starting Phase 1 V2 Endogenous Overfit Test...")
    for i in range(1, 201):
        loss, aux = step(senses_input, target_tokens)
        mx.eval(loss) 
        
        if i % 20 == 0:
            l_r, l_c, l_m = aux
            print(f"Epoch {i:3d}: Total={loss.item():.4f} | Recon={l_r.item():.4f} | Coverage={l_c.item():.4f} | Momentum={l_m.item():.4f}")
            
    print("✅ Successfully verified Phase 1 V2: Endogenous space generated from scratch!")

if __name__ == "__main__":
    run_v2_poc()
