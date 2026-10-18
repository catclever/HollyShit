import torch
from torch.utils.data import Dataset
import pandas as pd
import json
import numpy as np
import random
from typing import List, Tuple
import os

class MixedDataset(Dataset):
    """
    Dataset for Phase 2 & 1.5.
    Extracts contiguous concept-aligned chunks from documents.
    Chunks are strictly bounded by `seq_len` and truncated at the last punctuation mark
    to preserve semantic completeness.
    """
    def __init__(self, 
                 wiki_parquet_path: str, 
                 dialogues_jsonl_path: str,
                 tokenizer, 
                 seq_len: int = 128,
                 target_mix_ratio: float = 0.8):
        
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.target_mix_ratio = target_mix_ratio
        
        self.documents = []
        self._load_data(wiki_parquet_path, dialogues_jsonl_path)
        
        if not self.documents:
            raise ValueError("No documents loaded!")
            
        self.punctuations = {'。', '！', '？', '；', '\n', '.', '!', '?', '…'}
        self.index = []
        self._build_index()

    def _load_data(self, wiki_path: str, dialog_path: str):
        if os.path.exists(wiki_path):
            print(f"Loading Wiki data from {wiki_path}...")
            try:
                df_wiki = pd.read_parquet(wiki_path)
                if 'completion' in df_wiki.columns:
                    self.documents.extend(df_wiki['completion'].dropna().tolist())
                print(f"Loaded {len(df_wiki)} wiki documents.")
                del df_wiki
            except Exception as e:
                print(f"Failed to load wiki data: {e}")
        else:
            print(f"Warning: {wiki_path} not found.")

        if os.path.exists(dialog_path):
            print(f"Loading Dialogue data from {dialog_path}...")
            try:
                with open(dialog_path, 'r', encoding='utf-8') as f:
                    dialogs = []
                    for line in f:
                        if not line.strip(): continue
                        turns = json.loads(line)
                        if isinstance(turns, list):
                            dialogs.append("\n".join(turns))
                self.documents.extend(dialogs)
                print(f"Loaded {len(dialogs)} dialogue documents.")
            except Exception as e:
                print(f"Failed to load dialogue data: {e}")
        else:
            print(f"Warning: {dialog_path} not found.")
            
        print(f"Total documents loaded: {len(self.documents)}")

    def _build_index(self):
        print(f"Building concept-aligned indices (max_len={self.seq_len})...")
        # We store tuples of (doc_idx, t_start, t_end, tgt_start, tgt_end)
        
        for doc_idx, doc in enumerate(self.documents):
            doc_len = len(doc)
            cursor = 0
            chunks = []
            
            while cursor < doc_len:
                end_idx = min(cursor + self.seq_len, doc_len)
                
                # If not at the very end of the document, scan backwards for punctuation
                if end_idx < doc_len:
                    punct_found = False
                    for i in range(end_idx - 1, cursor - 1, -1):
                        if doc[i] in self.punctuations:
                            end_idx = i + 1
                            punct_found = True
                            break
                    
                    # Fallback if no punctuation found in the entire window:
                    # just truncate rigidly at max_len
                    if not punct_found:
                        end_idx = min(cursor + self.seq_len, doc_len)
                
                # Minimum viable chunk size (e.g., skip empty/whitespace-only chunks)
                if end_idx - cursor > 1:
                    chunks.append((cursor, end_idx))
                    
                cursor = end_idx
                
            # Form contiguous pairs (W_t, W_tgt)
            for i in range(len(chunks) - 1):
                t_s, t_e = chunks[i]
                tgt_s, tgt_e = chunks[i+1]
                self.index.append((doc_idx, t_s, t_e, tgt_s, tgt_e))
                
        print(f"Index built! Total aligned sequence pairs: {len(self.index)}")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        doc_idx, t_s, t_e, tgt_s, tgt_e = self.index[idx]
        doc = self.documents[doc_idx]
        
        text_t = doc[t_s : t_e]
        text_tgt = doc[tgt_s : tgt_e]
        
        # Tokenize using the tokenizer's encode method which handles byte fallbacks for UNK characters
        tokens_t = self.tokenizer.encode(text_t, add_special_tokens=False)
        tokens_tgt = self.tokenizer.encode(text_tgt, add_special_tokens=False)
        
        # Target Mixing: randomly replace target with itself
        if random.random() > self.target_mix_ratio:
            tokens_tgt = tokens_t 
            
        # Pad to fixed length seq_len
        if len(tokens_t) < self.seq_len:
            tokens_t += [self.tokenizer.pad_token_id] * (self.seq_len - len(tokens_t))
        else:
            tokens_t = tokens_t[:self.seq_len]
            
        if len(tokens_tgt) < self.seq_len:
            tokens_tgt += [self.tokenizer.pad_token_id] * (self.seq_len - len(tokens_tgt))
        else:
            tokens_tgt = tokens_tgt[:self.seq_len]
            
        return torch.tensor(tokens_t, dtype=torch.long), torch.tensor(tokens_tgt, dtype=torch.long)

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
    from training.core.char_tokenizer import CharTokenizer
    
    tokenizer = CharTokenizer()
    dataset = Phase2Dataset(
        wiki_parquet_path="data/Basic_ZH/wiki_filtered.parquet",
        dialogues_jsonl_path="data/Basic_ZH/dialogues_254k.jsonl",
        tokenizer=tokenizer,
        seq_len=128
    )
    
    print(f"Dataset length: {len(dataset)}")
    for i in range(min(5, len(dataset))):
        w_t, w_tgt = dataset[i]
        
        # Strip padding for printing
        t_list = [t.item() for t in w_t if t.item() != tokenizer.pad_token_id]
        tgt_list = [t.item() for t in w_tgt if t.item() != tokenizer.pad_token_id]
        
        t_str = "".join([tokenizer.id_to_char.get(i, "<UNK>") for i in t_list])
        tgt_str = "".join([tokenizer.id_to_char.get(i, "<UNK>") for i in tgt_list])
        print(f"--- Pair {i} ---")
        print(f"W_t   (len={len(t_list)}): {t_str}")
        print(f"W_tgt (len={len(tgt_list)}): {tgt_str}")
