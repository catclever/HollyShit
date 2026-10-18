import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import random
from typing import List, Tuple
import os

class OmniDataset(Dataset):
    """
    Dataset for Omni data loading.
    Reads pre-chunked data from chunked_mixed_omni.parquet.
    """
    def __init__(self, 
                 omni_parquet_path: str, 
                 tokenizer, 
                 seq_len: int = 512, 
                 dynamic_padding: bool = False):
        
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.dynamic_padding = dynamic_padding
        self.documents = []
        
        if os.path.exists(omni_parquet_path):
            print(f"Loading Omni data from {omni_parquet_path}...")
            df = pd.read_parquet(omni_parquet_path)
            
            # Reconstruct documents by stripping <EOS> and joining chunks
            for row_chunks in df['chunks']:
                # row_chunks is a numpy array or list of strings
                doc = "".join([c.replace("<EOS>", "") for c in row_chunks])
                self.documents.append(doc)
            print(f"Loaded {len(self.documents)} Omni documents.")
            del df
        else:
            raise ValueError(f"Omni dataset not found at {omni_parquet_path}")
            
        self.punctuations = {'。', '！', '？', '；', '\n', '.', '!', '?', '…'}
        self.index = []
        self._build_index()

    def _build_index(self):
        print(f"Building Omni aligned indices (max_len={self.seq_len})...")
        for doc_idx, doc in enumerate(self.documents):
            doc_len = len(doc)
            cursor = 0
            
            while cursor < doc_len:
                end_idx = min(cursor + self.seq_len, doc_len)
                
                if end_idx < doc_len:
                    punct_found = False
                    for i in range(end_idx - 1, cursor - 1, -1):
                        if doc[i] in self.punctuations:
                            end_idx = i + 1
                            punct_found = True
                            break
                    if not punct_found:
                        end_idx = min(cursor + self.seq_len, doc_len)
                
                if end_idx - cursor > 1:
                    self.index.append((doc_idx, cursor, end_idx))
                    
                cursor = end_idx
                
        print(f"Index built! Total aligned sequences: {len(self.index)}")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        # Phase 1.5 Auto-Encoding doesn't need W_tgt, so we just return (W_t, W_t) for compatibility
        doc_idx, t_s, t_e = self.index[idx]
        doc = self.documents[doc_idx]
        
        text_t = doc[t_s : t_e]
        tokens_t = self.tokenizer.encode(text_t, add_special_tokens=False)
            
        if len(tokens_t) > self.seq_len:
            tokens_t = tokens_t[:self.seq_len]
        elif not self.dynamic_padding and len(tokens_t) < self.seq_len:
            tokens_t += [self.tokenizer.pad_token_id] * (self.seq_len - len(tokens_t))
            
        t_tensor = torch.tensor(tokens_t, dtype=torch.long)
        return t_tensor, t_tensor

    def dynamic_collate(self, batch):
        """Pads the batch dynamically to the longest sequence in this batch, up to seq_len."""
        tensors = [item[0] for item in batch]
        from torch.nn.utils.rnn import pad_sequence
        padded = pad_sequence(tensors, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        return padded, padded
