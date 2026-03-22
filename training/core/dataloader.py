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
                 seed: int = 42):
        
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
        
        # 2. Tokenize and Pad dynamically
        # Ensure your tokenizer is set up correctly (e.g. padding=True, truncation=True)
        # Some tokenizers return PyTorch tensors if return_tensors='pt'. We use numpy/lists.
        encoded = self.tokenizer(
            batch_texts, 
            padding=True, 
            truncation=True, 
            max_length=self.max_seq_len,
            return_tensors='np'
        )
        # encoded['input_ids'] is shape (batch, padded_seq_len)
        token_inputs = mx.array(encoded['input_ids'])
        attention_mask = mx.array(encoded['attention_mask'])
        
        # 3. Fetch Embeddings via Mmap
        # This triggers Disk I/O precisely for these specific indices
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
    Unlike Phase 0 which pulls isolated sentences, this pulls *contiguous episodes*
    (L consecutive sentences) to form a dynamic trajectory for Mamba to learn momentum.
    """
    def __init__(self, 
                 emb_paths: List[str],
                 batch_size: int = 16,
                 episode_len: int = 10,
                 total_samples: int = 7619244,
                 seed: int = 42):
        
        self.batch_size = batch_size
        self.episode_len = episode_len
        self.total_samples = total_samples
        
        print("Mmapping embedding arrays for Phase 1...")
        self.embs = []
        for path in emb_paths:
            arr = np.load(path, mmap_mode='r')
            self.embs.append(arr)
            
        self.rng = np.random.default_rng(seed)
        self.current_epoch = 0
        self.step = 0
        
    def __iter__(self):
        return self
        
    def __next__(self) -> List[mx.array]:
        # To get a trajectory, we randomly select a starting index for each item in the batch
        # ensuring we have enough room to grab `episode_len` contiguous chunks.
        max_start = self.total_samples - self.episode_len
        start_indices = self.rng.integers(0, max_start, size=(self.batch_size,))
        
        # Build the exact ranges for each episode
        # Shape: (B, episode_len)
        episode_indices = start_indices[:, None] + np.arange(self.episode_len)
        
        # Fetch contiguous embeddings
        # Resulting batch_embs will be a list of mx.arrays of shape (B, L, d_emb)
        batch_embs = [mx.array(arr[episode_indices]) for arr in self.embs]
        
        self.step += 1
        return batch_embs
        
    def state_dict(self) -> Dict[str, Any]:
        return {"step": self.step, "current_epoch": self.current_epoch}
        
    def load_state_dict(self, state: Dict[str, Any]):
        self.step = state["step"]
        self.current_epoch = state["current_epoch"]
