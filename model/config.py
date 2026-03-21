from dataclasses import dataclass, field
from typing import List

@dataclass
class ModelConfig:
    """
    Central Configuration for the Endogenous Architecture (Phase 2).
    """
    
    # --- Multi-View Embedding Configuration ---
    # We use 4 external embeddings: RoBERTa(768), GTE(1024), BGE(1024), Text2Vec(768)
    emb_dims: List[int] = field(default_factory=lambda: [768, 1024, 1024, 768])
    
    # The absolute dimensions of the Mamba spatial manifold
    z_dim: int = 1024
    
    # --- Mamba Planner Configuration ---
    mamba_d_state: int = 16
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    
    # --- Weak Decoder Configuration ---
    # Needs to be a shallow decoder to force Mamba to learn the geometry
    vocab_size: int = 65000  # Default large vocab size, replace with actual tokenizer size
    decoder_layers: int = 2
    decoder_heads: int = 8
    
    # --- Training Data & Batching ---
    max_seq_len: int = 512
