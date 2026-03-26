import numpy as np
import pandas as pd
import mlx.core as mx
import json
import os
from typing import List, Tuple, Dict, Any

class MultiEmbDataLoader:
    """
    Dataloader that handles:
    1. Reading text chunks from Parquet (in-memory, fast enough for 7M strings).
    2. Zero-copy / mmap loading of 4 massive .npy embedding files.
    3. Dynamic tokenization & padding.
    4. Interrupt / Resume Checkpointing (Saving the exact shuffled order).
    """
    def __init__(self, 
                 parquet_path: str,
                 emb_paths: List[str],
                 tokenizer,
                 batch_size: int = 256,
                 max_seq_len: int = 512,
                 shuffle: bool = True,
                 seed: int = 42,
                 backend: str = 'mlx'):
        
        self.backend = backend
        
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.shuffle = shuffle
        self.seed = seed
        self.tokenizer = tokenizer
        
        # 1. Load Text Data
        print(f"Loading Text Parquet from {parquet_path}...")
        df = pd.read_parquet(parquet_path)
        # Explode the lists of chunks into a flat list of 7.6M strings
        # dropna() just in case
        self.text_chunks = df['chunks'].explode().dropna().tolist()
        self.total_samples = len(self.text_chunks)
        print(f"Loaded {self.total_samples} text chunks into memory.")
        del df # free up memory of the original dataframe
        
        # 2. Memory-Map the Numpy Embeddings
        print("Memory-mapping embedding arrays...")
        self.embs = []
        for path in emb_paths:
            # mmap_mode='r' completely avoids loading the massive files into RAM
            arr = np.load(path, mmap_mode='r')
            assert arr.shape[0] == self.total_samples, f"Shape mismatch in {path}: {arr.shape[0]} != {self.total_samples}"
            self.embs.append(arr)
            
        print("Embeddings mapped successfully.")
        
        # 3. Epoch & Batch tracking
        self.rng = np.random.default_rng(seed)
        self.current_epoch = 0
        self.batch_idx = 0
        self.indices = np.arange(self.total_samples)
        
        if self.shuffle:
            self.rng.shuffle(self.indices)
            
        self.num_batches = int(np.ceil(self.total_samples / self.batch_size))
        
    def __iter__(self):
        return self
        
    def __next__(self) -> Tuple[mx.array, List[mx.array], mx.array]:
        if self.batch_idx >= self.num_batches:
            # End of Epoch
            self.current_epoch += 1
            self.batch_idx = 0
            if self.shuffle:
                self.rng.shuffle(self.indices)
            raise StopIteration
            
        start_idx = self.batch_idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.total_samples)
        
        batch_indices = self.indices[start_idx:end_idx]
        
        # 1. Fetch Texts
        batch_texts = [self.text_chunks[i] for i in batch_indices]
        
        # 2. Tokenize and Pad dynamically using Custom CharTokenizer
        encoded_list = [self.tokenizer.encode(t, add_special_tokens=True)[:self.max_seq_len] for t in batch_texts]
        max_len = max(len(seq) for seq in encoded_list)
        
        padded_ids = []
        masks = []
        for seq in encoded_list:
            pad_len = max_len - len(seq)
            padded_ids.append(seq + [self.tokenizer.pad_token_id] * pad_len)
            masks.append([1] * len(seq) + [0] * pad_len)
            
        # 3. Fetch Embeddings via Mmap
        # This triggers Disk I/O precisely for these specific indices
        if self.backend == 'torch':
            import torch
            token_inputs = torch.tensor(padded_ids, dtype=torch.long)
            attention_mask = torch.tensor(masks, dtype=torch.float32)
            batch_embs = [torch.from_numpy(np.array(arr[batch_indices])).float() for arr in self.embs]
        else:
            token_inputs = mx.array(padded_ids)
            attention_mask = mx.array(masks)
            batch_embs = [mx.array(arr[batch_indices]) for arr in self.embs]
        
        self.batch_idx += 1
        
        return token_inputs, batch_embs, attention_mask

    # --- Checkpointing / Interrupt Resume Support ---
    
    def state_dict(self) -> Dict[str, Any]:
        """Returns the internal state for resuming."""
        return {
            "current_epoch": self.current_epoch,
            "batch_idx": self.batch_idx,
            "indices": self.indices.tolist() # Safe to serialize exactly how it was shuffled
        }
        
    def load_state_dict(self, state: Dict[str, Any]):
        """Restores the exact shuffle and position."""
        self.current_epoch = state["current_epoch"]
        self.batch_idx = state["batch_idx"]
        self.indices = np.array(state["indices"])
        print(f"Resumed Dataloader from Epoch {self.current_epoch}, Batch {self.batch_idx}/{self.num_batches}")

class Phase1DataLoader:
    """
    Dataloader specifically for Phase 1 (Mamba Spatiotemporal Dynamics).
    It pulls *contiguous entire documents* dynamically instead of fixed arbitrary slices,
    to ensure Mamba learns the true, uncorrupted semantic arc of an article.
    """
    def __init__(self, 
                 parquet_path: str,
                 emb_paths: List[str],
                 batch_size: int = 16,
                 max_episode_len: int = None,
                 seed: int = 42,
                 backend: str = 'mlx'):
        
        self.backend = backend
        
        self.batch_size = batch_size
        self.max_episode_len = max_episode_len
        
        # 1. Load exact document boundaries to avoid crossing them!
        print("Reading Document Chunk Counts...")
        df = pd.read_parquet(parquet_path, columns=['chunk_count'])
        self.chunk_counts = df['chunk_count'].values
        
        # 2. Cumulative sum gives us the exact starting index for each document in the giant mmapped arrays
        self.doc_start_indices = np.concatenate(([0], np.cumsum(self.chunk_counts)[:-1]))
        self.total_docs = len(self.chunk_counts)
        print(f"Loaded strictly bounded {self.total_docs} semantic documents.")
        
        del df
        
        print("Mmapping embedding arrays for Phase 1...")
        self.embs = []
        for path in emb_paths:
            arr = np.load(path, mmap_mode='r')
            self.embs.append(arr)
            
        self.rng = np.random.default_rng(seed)
        self.current_epoch = 0
        self.batch_idx = 0
        
        self.indices = np.arange(self.total_docs)
        self.rng.shuffle(self.indices)
        self.num_batches = int(np.ceil(self.total_docs / self.batch_size))
        
    def __iter__(self):
        return self
        
    def __next__(self) -> Tuple[List[mx.array], mx.array]:
        if self.batch_idx >= self.num_batches:
            self.current_epoch += 1
            self.batch_idx = 0
            self.rng.shuffle(self.indices)
            raise StopIteration
            
        start_idx = self.batch_idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.total_docs)
        
        batch_doc_indices = self.indices[start_idx:end_idx]
        
        # 1. Fetch exactly the bounded chunks for each document
        raw_embs_list = [[] for _ in range(len(self.embs))]
        lengths = []
        
        for doc_idx in batch_doc_indices:
            start_pos = self.doc_start_indices[doc_idx]
            count = self.chunk_counts[doc_idx]
            
            if self.max_episode_len is not None:
                eff_len = min(count, self.max_episode_len)
            else:
                eff_len = min(count, 256) # Safety cap
                
            lengths.append(eff_len)
            
            for i in range(len(self.embs)):
                # Just copy the slice into memory
                raw_embs_list[i].append(np.array(self.embs[i][start_pos : start_pos+eff_len]))
                
        # 2. Pad to the exact max_episode_len if provided; otherwise pad dynamically to batch's max
        max_len = self.max_episode_len if self.max_episode_len is not None else max(lengths)
        
        # 2. Pad to the maximum document length in THIS specific batch
        padded_embs = [[] for _ in range(len(self.embs))]
        masks = []
        z_dim = raw_embs_list[0][0].shape[-1]
        
        for b in range(len(batch_doc_indices)):
            l = lengths[b]
            pad_len = max_len - l
            
            for i in range(len(self.embs)):
                arr = raw_embs_list[i][b]
                if pad_len > 0:
                    pad_array = np.zeros((pad_len, z_dim), dtype=arr.dtype)
                    arr = np.concatenate([arr, pad_array], axis=0)
                padded_embs[i].append(arr)
                
            masks.append([1] * l + [0] * pad_len)
            
        # Convert to arrays matching backend
        if self.backend == 'torch':
            import torch
            batch_embs_tensor = [torch.from_numpy(np.stack(padded_embs[i])).float() for i in range(len(self.embs))]
            masks_tensor = torch.tensor(masks, dtype=torch.float32)
            self.batch_idx += 1
            return batch_embs_tensor, masks_tensor
        else:
            batch_embs_mx = [mx.array(np.stack(padded_embs[i])) for i in range(len(self.embs))]
            masks_mx = mx.array(masks)
            self.batch_idx += 1
            return batch_embs_mx, masks_mx
        
    def state_dict(self) -> Dict[str, Any]:
        return {"batch_idx": self.batch_idx, "current_epoch": self.current_epoch, "indices": self.indices.tolist()}
        
    def load_state_dict(self, state: Dict[str, Any]):
        self.batch_idx = state["batch_idx"]
        self.current_epoch = state["current_epoch"]
        self.indices = np.array(state["indices"])
