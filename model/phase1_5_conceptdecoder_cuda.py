import torch
import torch.nn as nn

class WeakConceptDecoderCUDA(nn.Module):
    """
    Phase 1.5 Decoder: A 'Weak Translator'.
    Uses a 1-layer GRU. The Concept Vector is provided as the initial hidden state (h_0).
    Because the RNN is extremely weak (short memory, simple transitions), it forces 
    the Concept Vector to perfectly encapsulate the semantic meaning of the chunk.
    """
    def __init__(self, vocab_size: int, embed_dim: int = 512, hidden_dim: int = 512):
        super().__init__()
        self.char_embed = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(input_size=embed_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, concept_vector: torch.Tensor, target_tokens: torch.Tensor) -> torch.Tensor:
        """
        concept_vector: [B, hidden_dim] - The concept vector (initial hidden state)
        target_tokens: [B, L] - Ground truth tokens. 
                       Should start with <BOS> token.
        returns: logits [B, L-1, vocab_size]
        """
        # h_0 shape: [num_layers=1, B, hidden_dim]
        h_0 = concept_vector.unsqueeze(0) 
        
        # Teacher forcing: Input all tokens except the last one
        x = target_tokens[:, :-1]
        embeds = self.char_embed(x) # [B, L-1, embed_dim]
        
        # Pass through GRU
        out, _ = self.gru(embeds, h_0) # out: [B, L-1, hidden_dim]
        
        logits = self.fc(out) # [B, L-1, vocab_size]
        return logits
        
    @torch.no_grad()
    def generate(self, concept_vector: torch.Tensor, bos_token_id: int, max_len: int = 32) -> torch.Tensor:
        """
        Autoregressive generation at inference time.
        """
        B = concept_vector.size(0)
        device = concept_vector.device
        
        h = concept_vector.unsqueeze(0)
        current_token = torch.full((B, 1), bos_token_id, dtype=torch.long, device=device)
        
        generated_tokens = []
        
        for _ in range(max_len):
            embed = self.char_embed(current_token)
            out, h = self.gru(embed, h)
            logits = self.fc(out)
            
            # Greedy decode
            next_token = logits.argmax(dim=-1) # [B, 1]
            generated_tokens.append(next_token)
            current_token = next_token
            
        return torch.cat(generated_tokens, dim=1) # [B, max_len]

if __name__ == "__main__":
    # Test
    vocab_size = 100
    B, D = 4, 512
    L = 10
    
    decoder = WeakConceptDecoderCUDA(vocab_size=vocab_size, embed_dim=D, hidden_dim=D)
    
    concept_vec = torch.randn(B, D)
    target_tokens = torch.randint(0, vocab_size, (B, L))
    
    logits = decoder(concept_vec, target_tokens)
    print(f"Logits shape: {logits.shape} (Expected: {B}, {L-1}, {vocab_size})")
    
    gen_tokens = decoder.generate(concept_vec, bos_token_id=1, max_len=8)
    print(f"Generated tokens shape: {gen_tokens.shape} (Expected: {B}, 8)")
