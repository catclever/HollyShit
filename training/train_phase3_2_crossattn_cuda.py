import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import math
import os
import sys

# Auto-resolve the parent workspace root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir in sys.path and os.path.exists(os.path.join(script_dir, "model.py")):
    sys.path = [p for p in sys.path if p != script_dir] + [script_dir]
    
from training.core.char_tokenizer import CharTokenizer
from training.data.mixed_dataset import MixedDataset
from model.phase1_charencoder_cuda import TinyCharEncoderCUDA
from model.concept_chunker import ConceptChunker
from model.crossattn_decoder_cuda import WeakConceptDecoderCUDA
from training.core.args import get_training_parser
from training.core.checkpoint import Checkpointer

def pad_chunks(chunks, bos_token_id, eos_token_id, pad_token_id):
    """
    Takes a list of (concept_vector, chunk_tokens).
    Prepends BOS, appends EOS, and pads them to the max length.
    Returns:
      batched_c: [N, D]
      batched_t: [N, max_len]
    """
    N = len(chunks)
    D = chunks[0][0].shape[0]
    
    # Prepend BOS, append EOS
    sequences = []
    for _, tokens in chunks:
        seq = [bos_token_id] + tokens.tolist() + [eos_token_id]
        sequences.append(torch.tensor(seq, dtype=torch.long))
        
    max_len = max(len(s) for s in sequences)
    
    batched_c = torch.stack([c for c, _ in chunks])
    batched_t = torch.full((N, max_len), pad_token_id, dtype=torch.long)
    
    for i, seq in enumerate(sequences):
        batched_t[i, :len(seq)] = seq
        
    return batched_c, batched_t

def train():
    parser = get_training_parser("Phase 1.5 Decoder Training (CUDA)")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    tokenizer = CharTokenizer()
    dataset = MixedDataset(
        wiki_parquet_path="data/Basic_ZH/wiki_filtered.parquet",
        dialogues_jsonl_path="data/Basic_ZH/dialogues_254k.jsonl",
        tokenizer=tokenizer,
        seq_len=args.seq_len
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Load Phase 1 Encoder (Frozen)
    encoder = TinyCharEncoderCUDA(vocab_size=tokenizer.vocab_size).to(device)
    ckpt_path = "checkpoints/basencoder_step_115000/student.pt"
    if os.path.exists(ckpt_path):
        encoder.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        print("Loaded frozen TinyCharEncoder.")
    else:
        print(f"Warning: Could not find checkpoint {ckpt_path}. Using random initialization for Encoder.")
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
        
    chunker = ConceptChunker(threshold=0.5).to(device)
    
    # Phase 1.5 Decoder
    decoder = WeakConceptDecoderCUDA(
        vocab_size=tokenizer.vocab_size, 
        embed_dim=1024, 
        hidden_dim=1024
    ).to(device)
    
    optimizer = optim.AdamW(decoder.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    
    checkpointer = Checkpointer(out_dir=args.out_dir, prefix=args.ckpt_prefix, keep_last_k=args.keep_last_k)
    checkpointer.register_model("decoder", decoder)
    # Note: Checkpointer currently uses mx.save_safetensors for optimizers, so for PyTorch we omit it 
    # or rely on a custom PyTorch optimizer saver if we modify checkpoint.py later.
    checkpointer.register_args(args)
    
    global_step = 0
    if args.auto_resume:
        global_step = checkpointer.load_latest()
    elif args.resume_from:
        global_step = checkpointer.load(args.resume_from)
        
    start_epoch = global_step // len(dataloader) if len(dataloader) > 0 else 0
    
    for epoch in range(start_epoch, args.epochs):
        decoder.train()
        for batch_idx, (w_t, _) in enumerate(dataloader):
            w_t = w_t.to(device) # [B, L]
            
            # 1. Forward through frozen encoder
            with torch.no_grad():
                h = encoder.get_normalized_embeddings(w_t)
                for layer in encoder.layers:
                    h = layer(h, None)
                h = encoder.final_ln(h) # [B, L, D]
                
                # 2. Extract Concept Chunks
                chunks = chunker.extract_chunks(h, w_t)
            
            # 3. Format chunks for Seq2Seq
            batched_c, batched_t = pad_chunks(
                chunks, 
                tokenizer.bos_token_id, 
                tokenizer.eos_token_id, 
                tokenizer.pad_token_id
            )
            batched_c = batched_c.to(device)
            batched_t = batched_t.to(device)
            
            # 4. Train Decoder
            optimizer.zero_grad()
            logits = decoder(batched_c, batched_t) # [N, max_len-1, V]
            
            # targets are shifted by 1
            targets = batched_t[:, 1:].contiguous().view(-1)
            logits_flat = logits.view(-1, tokenizer.vocab_size)
            
            loss = criterion(logits_flat, targets)
            loss.backward()
            optimizer.step()
            
            global_step += 1
            
            if global_step % 100 == 0:
                print(f"Epoch {epoch+1} | Step {global_step} | Loss: {loss.item():.4f}")
                
            # Quick qualitative evaluation
            if global_step % 1000 == 0:
                decoder.eval()
                with torch.no_grad():
                    sample_c = batched_c[0:1]
                    sample_t = batched_t[0]
                    
                    gen_ids = decoder.generate(sample_c, tokenizer.bos_token_id, max_len=args.seq_len)
                    
                    orig_text = "".join([tokenizer.id_to_char.get(i.item(), "[UNK]") for i in sample_t if i != tokenizer.pad_token_id and i != tokenizer.bos_token_id and i != tokenizer.eos_token_id])
                    gen_text = "".join([tokenizer.id_to_char.get(i.item(), "[UNK]") for i in gen_ids[0]])
                    gen_text = gen_text.split("<EOS>")[0] if "<EOS>" in gen_text else gen_text
                    
                    print(f"\n--- EVAL STEP {global_step} ---")
                    print(f"Target: {orig_text}")
                    print(f"Generated: {gen_text}")
                    print("-" * 25 + "\n")
                decoder.train()

        # Save checkpoint
        checkpointer.save(global_step)
        print(f"Saved checkpoint for epoch {epoch+1} at global step {global_step}.")

if __name__ == "__main__":
    train()
