from dataclasses import dataclass, field
from typing import List

@dataclass
class ModelConfig:
    """
    Central Configuration for the Endogenous Architecture (Phase 2).
    """
    
    # --- Multi-View Embedding Configuration (Phase 0.2) ---
    # The emb_dims list is now DEPRECATED.
    # The model dynamically sniffs the input .npy shapes at runtime!
    
    # The unified projection dimension output by the SensoryFuser
    d_model: int = 1024
    
    # The absolute dimensions of the Mamba spatial manifold (GodEncoder output)
    z_dim: int = 1024
    
    # --- Mamba Planner Configuration ---
    mamba_d_state: int = 16
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    
    # --- Weak Decoder Configuration ---
    # Needs to be a shallow decoder to force Mamba to learn the geometry
    vocab_size: int = 8000
    decoder_layers: int = 2
    decoder_heads: int = 8
    
    # --- Training Data & Batching ---
    max_seq_len: int = 512
