from dataclasses import dataclass, field
from typing import List

@dataclass
class WeakDecoderConfig:
    """
    Configuration specifically for the Phase 0 / Phase 0.2 Weak Decoder.
    Used for historical alignment training where the model was forced to output tokens
    to learn geometric embedding representations.
    """
    
    # --- Weak Decoder Configuration ---
    # Needs to be a shallow decoder to force GodEncoder to learn the geometry
    vocab_size: int = 8000
    decoder_layers: int = 2
    decoder_heads: int = 8
    
    # --- Training Data & Batching ---
    max_seq_len: int = 512

