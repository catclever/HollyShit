import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

class ConceptChunker(nn.Module):
    """
    Takes character-level dense embeddings [B, L, D] and groups consecutive 
    embeddings with cosine similarity > threshold into 'Concept Clouds'.
    Applies Zero-Order Hold (Step-Function) by repeating the mean vector 
    of the chunk over its original duration.
    """
    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold

    @torch.no_grad()
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        h: [B, L, D]
        returns: X: [B, L, D] 
        This is a non-learnable deterministic transformation.
        """
        B, L, D = h.shape
        out = torch.empty_like(h)
        
        # Normalize h once for faster cosine similarity
        h_norm = F.normalize(h, p=2, dim=-1)
        
        for b in range(B):
            h_seq = h[b]
            h_norm_seq = h_norm[b]
            
            chunk_start = 0
            # Track sum and count for fast mean updates
            current_sum = h_seq[0].clone()
            current_count = 1
            current_mean_norm = h_norm_seq[0].clone()
            
            for i in range(1, L):
                # Cosine similarity = dot product of normalized vectors
                sim = torch.dot(h_norm_seq[i], current_mean_norm).item()
                
                if sim > self.threshold:
                    current_sum += h_seq[i]
                    current_count += 1
                    # Update normalized mean
                    current_mean = current_sum / current_count
                    current_mean_norm = F.normalize(current_mean, p=2, dim=-1)
                else:
                    # Finalize chunk
                    final_mean = current_sum / current_count
                    out[b, chunk_start:i] = final_mean
                    
                    # Start new chunk
                    chunk_start = i
                    current_sum = h_seq[i].clone()
                    current_count = 1
                    current_mean_norm = h_norm_seq[i].clone()
                    
            # Finalize last chunk
            final_mean = current_sum / current_count
            out[b, chunk_start:L] = final_mean
            
        return out
        
    @torch.no_grad()
    def extract_chunks(self, h: torch.Tensor, tokens: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Used for Phase 1.5 Decoder training.
        Extracts individual concept vectors and their corresponding character token lists.
        h: [B, L, D]
        tokens: [B, L]
        returns: A list of tuples (concept_vector, chunk_tokens) 
                 where concept_vector is [D] and chunk_tokens is [seq_len]
        """
        B, L, D = h.shape
        h_norm = F.normalize(h, p=2, dim=-1)
        
        chunks = []
        
        for b in range(B):
            h_seq = h[b]
            h_norm_seq = h_norm[b]
            tok_seq = tokens[b]
            
            chunk_start = 0
            current_sum = h_seq[0].clone()
            current_count = 1
            current_mean_norm = h_norm_seq[0].clone()
            
            for i in range(1, L):
                sim = torch.dot(h_norm_seq[i], current_mean_norm).item()
                
                if sim > self.threshold:
                    current_sum += h_seq[i]
                    current_count += 1
                    current_mean = current_sum / current_count
                    current_mean_norm = F.normalize(current_mean, p=2, dim=-1)
                else:
                    final_mean = current_sum / current_count
                    chunks.append((final_mean, tok_seq[chunk_start:i]))
                    
                    chunk_start = i
                    current_sum = h_seq[i].clone()
                    current_count = 1
                    current_mean_norm = h_norm_seq[i].clone()
                    
            final_mean = current_sum / current_count
            chunks.append((final_mean, tok_seq[chunk_start:L]))
            
        return chunks

if __name__ == "__main__":
    # Quick test
    B, L, D = 2, 10, 8
    torch.manual_seed(42)
    h = torch.randn(B, L, D)
    tokens = torch.randint(0, 100, (B, L))
    chunker = ConceptChunker(threshold=0.5)
    
    h[0, 1] = h[0, 0] + 0.1 * torch.randn(D) 
    h[0, 2] = h[0, 0] + 0.1 * torch.randn(D) 
    
    chunks = chunker.extract_chunks(h, tokens)
    print(f"Extracted {len(chunks)} chunks.")
    print(f"First chunk length: {len(chunks[0][1])} (Expected 3)")
