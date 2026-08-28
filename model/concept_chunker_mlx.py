import mlx.core as mx
import mlx.nn as nn
import numpy as np

from dataclasses import dataclass

@dataclass
class ConceptChunkerConfig:
    threshold: float = 0.5

class ConceptChunkerMLX(nn.Module):
    """
    MLX version of the ConceptChunker, optimized with NumPy for the sequential loop.
    Since the ZOH grouping logic contains data-dependent branching that cannot be 
    easily parallelized or compiled in MLX without `mx.scan`, running it in MLX eager mode
    causes thousands of GPU syncs (.item()).
    
    The fastest solution is to pull the sequence to NumPy, run the 2048-iteration loop
    in pure C-backed NumPy in < 1ms, and push the grouped result back to MLX.
    """
    def __init__(self, config: ConceptChunkerConfig = None, **kwargs):
        super().__init__()
        if config is None:
            config = ConceptChunkerConfig(**kwargs)
        self.config = config
        self.threshold = config.threshold

    def get_config(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self.config)

    def __call__(self, h: mx.array, is_punct: mx.array = None) -> mx.array:
        """
        h: [B, L, D]
        is_punct: [B, L] boolean array. If True, forces a chunk break.
        returns: X: [B, L, D]
        """
        # 1. Single synchronization: MLX -> NumPy
        h_np = np.array(h)
        B, L, D = h_np.shape
        out_np = np.empty_like(h_np)
        
        is_punct_np = None
        if is_punct is not None:
            is_punct_np = np.array(is_punct)
        
        # 2. Fast NumPy operations
        norms = np.linalg.norm(h_np, ord=2, axis=-1, keepdims=True)
        # Avoid division by zero
        norms = np.maximum(norms, 1e-8)
        h_norm = h_np / norms
        
        for b in range(B):
            h_seq = h_np[b]
            h_norm_seq = h_norm[b]
            punct_seq = is_punct_np[b] if is_punct_np is not None else None
            
            chunk_start = 0
            current_sum = h_seq[0].copy()
            current_count = 1
            current_mean_norm = h_norm_seq[0].copy()
            
            for i in range(1, L):
                sim = np.dot(h_norm_seq[i], current_mean_norm)
                
                # Check for punctuation boundaries
                force_break = False
                if punct_seq is not None:
                    is_p = punct_seq[i]
                    was_p = punct_seq[i-1]
                    # If current is punctuation, or previous was punctuation, force a break
                    # This ensures punctuation symbols are isolated in their own chunks
                    # and concepts don't cross punctuation boundaries.
                    if is_p or was_p:
                        force_break = True
                
                if sim > self.threshold and not force_break:
                    current_sum += h_seq[i]
                    current_count += 1
                    current_mean = current_sum / current_count
                    
                    mean_norm = np.linalg.norm(current_mean)
                    current_mean_norm = current_mean / max(mean_norm, 1e-8)
                else:
                    final_mean = current_sum / current_count
                    out_np[b, chunk_start:i] = final_mean
                    
                    chunk_start = i
                    current_sum = h_seq[i].copy()
                    current_count = 1
                    current_mean_norm = h_norm_seq[i].copy()
                    
            final_mean = current_sum / current_count
            out_np[b, chunk_start:L] = final_mean
            
        # 3. Fast transfer: NumPy -> MLX
        return mx.array(out_np)

