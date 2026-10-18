import numpy as np
import pandas as pd

import json
import os
import glob
import re
import queue
import threading
from typing import List, Tuple, Dict, Any

class Phase1_MultiEmbDataLoader:
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
        self.seed = seed if seed is not None else int(__import__("time").time())
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
        
    def __next__(self) -> Tuple[Any, List[Any], Any]:
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
            import mlx.core as mx
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

class Phase1_ChunkedNpzDataLoader:
    """
    专门适配从 ModelScope (如 catclever/emb_npy) 拉取的 `.npz` 切片矩阵设计的双层乱序 DataLoader。
    机制：
    1. 不吃内存：只在内存驻留"当前正在训练的任务块"的特征（比如每次只把某个 50w 条特征解压进 RAM）。
    2. 双层乱序 (Double Shuffle)：每个 Epoch 开始时打乱所有 Chunk 文件的流转顺序；在读取单个 Chunk 时，在 Chunk 内部将其局部 batch 完全乱序。
    """
    def __init__(self, 
                 parquet_path: str,
                 models: List[str], # 对应你的模型标识，例如 ["bge", "qwen", "xiaobu"]
                 chunk_patterns: dict = None,
                 tokenizer=None,
                 ms_repo_id: str = None, 
                 chunk_size: int = 500000,
                 local_npz_dir: str = None,
                 batch_size: int = 256,
                 max_seq_len: int = 512,
                 shuffle: bool = True,
                 seed: int = 42,
                 backend: str = 'mlx',
                 lazy_start: bool = False,
                 cache_dir: str = './datas',
                 auto_cleanup: bool = True):
        
        self.backend = backend
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.shuffle = shuffle
        self.seed = seed if seed is not None else int(__import__("time").time())
        self.models = models
        self.chunk_patterns = chunk_patterns or {}
        self.ms_repo_id = ms_repo_id
        self.tokenizer = tokenizer
        self.cache_dir = cache_dir
        self.auto_cleanup = auto_cleanup
        
        # 3. 提取对应的全体文本数据 (先行提取，以获得 total_samples 决定边界)
        print(f"[DataLoader] 正在装载基准文本体系: {parquet_path}...")
        df = pd.read_parquet(parquet_path)
        self.text_chunks = df['chunks'].explode().dropna().tolist()
        self.total_samples = len(self.text_chunks)
        print(f"[DataLoader] 成功吃进文本池: {self.total_samples} 条")
        del df
        
        # 1. 自动挂载魔搭数据集或加载本地路径 (改为只验证配置，去除暴力的 snapshot_download 全本拉取)
        if ms_repo_id:
            try:
                from modelscope.hub.file_download import dataset_file_download
                print(f"[DataLoader] 已启用 ModelScope 终极按需流式串流模式: {ms_repo_id}...")
            except ImportError:
                raise RuntimeError("请安装 modelscope: pip install modelscope")
        else:
            self.npz_dir = local_npz_dir
            if not self.npz_dir or not os.path.isdir(self.npz_dir):
                raise ValueError("必须提供 ms_repo_id 或有效的 local_npz_dir")

        # 2. 按模型分别探测物理分块，按 start 索引取交集（容错不同模型尾部 end 不同）
        self.chunk_bounds = []
        self._per_model_bounds = {}  # {model_name: {start: end}}
        try:
            if self.ms_repo_id:
                from modelscope.hub.api import HubApi
                api = HubApi()
                res = api.get_dataset_files(self.ms_repo_id, recursive=True)
                chunk_files = [f.get("Path", f.get("Name", "")) for f in res]
            else:
                chunk_files = glob.glob(os.path.join(self.npz_dir, "**", "*.npz"), recursive=True)
            
            # 按模型归类，各自收集 start → end 的映射
            for model_name in self.models:
                pattern = self.chunk_patterns.get(model_name, "")
                prefix = pattern.split("/")[0] if "/" in pattern else model_name
                start_to_end = {}
                for f in chunk_files:
                    f_normalized = f.replace("\\", "/")
                    if f_normalized.endswith(".npz") and (f_normalized.startswith(prefix + "/") or f"/{prefix}/" in f_normalized):
                        match = re.search(r'chunk_(\d+)_(\d+)\.npz', os.path.basename(f))
                        if match:
                            start_to_end[int(match.group(1))] = int(match.group(2))
                self._per_model_bounds[model_name] = start_to_end
                print(f"[DataLoader]   {model_name}: {len(start_to_end)} 个分块")
            
            # 按 start 索引取交集（容错：youtu 的 end=9000000 和 bge 的 end=8656199 都算 start=8500000）
            if self._per_model_bounds:
                common_starts = set.intersection(*[set(d.keys()) for d in self._per_model_bounds.values()])
                for s in sorted(common_starts):
                    # 取所有模型中最小的 end 作为安全数据范围
                    min_end = min(self._per_model_bounds[m][s] for m in self.models)
                    self.chunk_bounds.append((s, min_end))
            
            print(f"[DataLoader] 多模型安全交集（按start容错）：{len(self.chunk_bounds)} 个共享分块。")
        except Exception as e:
            print(f"[DataLoader] 物理探测失败，回退到严格数学计算: {e}")
            self.chunk_bounds = []
            for start_idx in range(0, self.total_samples, chunk_size):
                 end_idx = min(start_idx + chunk_size, self.total_samples)
                 self.chunk_bounds.append((start_idx, end_idx))


        
        # 4. 统计与 Epoch 控制变量
        self.rng = np.random.default_rng(seed)
        self.current_epoch = 0
        
        # 宏观控制表 (Chunks)
        self.macro_chunk_indices = np.arange(len(self.chunk_bounds))
        if self.shuffle:
             self.rng.shuffle(self.macro_chunk_indices)
             
        self.active_macro_idx_ptr = 0 # 当前正训练到第几个 chunk
        
        # 微观控制表 (内部流)
        self.active_chunk_embs = None
        self.active_micro_indices = []
        self.active_micro_ptr = 0
        self._prev_chunk_paths = []  # 上一个已消费分块的缓存路径，用于清理
        
        # 异步流水线 (Async Prefetch)
        self.prefetch_queue = queue.Queue(maxsize=1) # 内存里永远只多塞一块，极限控流
        self.stop_event = threading.Event()
        self._started = False
        
        # 延迟启动：如果要从 checkpoint 恢复，不要在这里白白下载一个块
        if not lazy_start:
            self._do_start()

    def _do_start(self):
        """真正启动后台下载，首次加载第一个块"""
        if self._started:
            return
        self._started = True
        self._start_prefetching(start_macro_ptr=0)
        self._pop_next_chunk()

    def _cleanup_cached_files(self, paths: list):
        """清理已经消费完毕的 ModelScope 缓存文件，释放磁盘空间"""
        cleaned = 0
        for p in paths:
            try:
                if p and os.path.exists(p):
                    os.remove(p)
                    cleaned += 1
            except OSError:
                pass
        if cleaned:
            print(f"[DataLoader] 🧹 已清理 {cleaned} 个已消费缓存文件")

    def _start_prefetching(self, start_macro_ptr):
        """挂起旧线程，启动全新后台线程，从指定的 macro_ptr 开始默默搬取"""
        self.stop_event.set()
        
        # 清空队列残骸
        while not self.prefetch_queue.empty():
            try:
                self.prefetch_queue.get_nowait()
            except queue.Empty:
                break
                
        self.stop_event = threading.Event()
        self.prefetch_thread = threading.Thread(target=self._prefetch_worker, args=(start_macro_ptr,), daemon=True)
        self.prefetch_thread.start()
        
    def _prefetch_worker(self, start_macro_ptr):
        """高强度纯后台下载/解压打工仔，永远走在 GPU 的前面"""
        worker_macro_ptr = start_macro_ptr
        while not self.stop_event.is_set():
            if worker_macro_ptr >= len(self.macro_chunk_indices):
                # 提示时代结束
                self.prefetch_queue.put(None)
                return

            chunk_idx = self.macro_chunk_indices[worker_macro_ptr]
            start_idx, end_idx = self.chunk_bounds[chunk_idx]
            
            try:
                active_chunk_embs = []
                cached_paths = []  # 记录下载路径用于后续清理
                for model_name in self.models:
                    pattern = self.chunk_patterns.get(model_name, f"{model_name}_chunk_{'{start:07d}'}_{'{end:07d}'}.npz")
                    # 用该模型自己的真实 end（容错命名差异，如 youtu 的 9000000 vs bge 的 8656199）
                    model_end = self._per_model_bounds.get(model_name, {}).get(start_idx, end_idx)
                    cand_name = pattern.format(start=start_idx, end=model_end)
                    
                    if self.ms_repo_id:
                         from modelscope.hub.file_download import dataset_file_download
                         path = dataset_file_download(self.ms_repo_id, cand_name, cache_dir=self.cache_dir)
                    else:
                         path = os.path.join(self.npz_dir, cand_name)
                         if not os.path.exists(path):
                              raise FileNotFoundError(f"后台打工仔抓取失败！本地目标块缺失：{path}")
                    
                    arr = np.load(path)['features']
                    active_chunk_embs.append(arr)
                    cached_paths.append(path)
                    
                cur_chunk_size = end_idx - start_idx
                active_micro_indices = np.arange(cur_chunk_size)
                
                if self.shuffle:
                    # 构建独立的后台局部无序宇宙
                    rng = np.random.default_rng(self.seed + worker_macro_ptr + self.current_epoch)
                    rng.shuffle(active_micro_indices)
                    
                payload = {
                    "embs": active_chunk_embs,
                    "micro_indices": active_micro_indices,
                    "global_start": start_idx,
                    "macro_idx_ptr": worker_macro_ptr,
                    "cached_paths": cached_paths
                }
                # 这里会阻塞！直到上一块被 GPU 吃光
                self.prefetch_queue.put(payload)
                worker_macro_ptr += 1
            except Exception as e:
                print(f"\n[DataLoader] 后台打工仔抓取/加载失败 (网络可能断开了？): {e}")
                print(f"[DataLoader] 这不是致命错误！打工仔将在 10 秒后原地重新尝试拉取 Chunk {chunk_idx}...")
                import time
                time.sleep(10)
                # 不退出进程，不抛出异常。
                # 既然失败了，worker_macro_ptr 就不加 1，直接 continue，下一轮会继续死磕这个块！
                continue

    def _pop_next_chunk(self):
        """主线程调用的吸星大法，把工人装好的箱子拿走"""
        # 先清理上一个已消费的分块缓存
        if self.auto_cleanup and self._prev_chunk_paths:
            self._cleanup_cached_files(self._prev_chunk_paths)
            self._prev_chunk_paths = []
        
        payload = self.prefetch_queue.get()
        if payload is None:
            raise StopIteration
        if isinstance(payload, Exception):
            raise payload
            
        self.active_chunk_embs = payload["embs"]
        self.active_micro_indices = payload["micro_indices"]
        self.active_global_start = payload["global_start"]
        self.active_macro_idx_ptr = payload["macro_idx_ptr"]
        self.active_micro_ptr = 0
        self._prev_chunk_paths = payload.get("cached_paths", [])
        
        if not hasattr(self, 'emb_dims'):
            self.emb_dims = [arr.shape[-1] for arr in self.active_chunk_embs]
            print(f"[DataLoader] 自动侦测到多路特征维度: {self.emb_dims}")
            
        print(f"\n[DataLoader] 💥 无缝吸入后台新缓存块 (全局起点: {self.active_global_start}) - 纯异步无卡顿！")

    def __iter__(self):
        if not self._started:
            self._do_start()
        return self

    def __next__(self) -> Tuple[Any, List[Any], Any]:
        # 如果当前 Chunk 已经被彻底吸干了，无缝提取下一个
        if self.active_micro_ptr >= len(self.active_micro_indices):
            try:
                self._pop_next_chunk()
            except StopIteration:
                # 一个 Epoch 完结！
                self.current_epoch += 1
                if self.shuffle:
                    self.rng.shuffle(self.macro_chunk_indices)
                # 重新使唤工人从头开始搬砖
                self._start_prefetching(start_macro_ptr=0)
                self._pop_next_chunk()  # 直接抓下一个 Epoch 的第一块
                raise StopIteration
            
        fetch_len = min(self.batch_size, len(self.active_micro_indices) - self.active_micro_ptr)
        batch_local_indices = self.active_micro_indices[self.active_micro_ptr : self.active_micro_ptr + fetch_len]
        self.active_micro_ptr += fetch_len
        
        # 1. 映射回全局索引以抓取物理对应的文本序列
        batch_global_indices = self.active_global_start + batch_local_indices
        batch_texts = [self.text_chunks[i] for i in batch_global_indices]
        
        # 2. Tokenize & Pad (与旧版完全一致)
        encoded_list = [self.tokenizer.encode(t, add_special_tokens=True)[:self.max_seq_len] for t in batch_texts]
        max_len = max(max(len(seq) for seq in encoded_list), 1)
        
        padded_ids = []
        masks = []
        for seq in encoded_list:
            pad_len = max_len - len(seq)
            padded_ids.append(seq + [self.tokenizer.pad_token_id] * pad_len)
            masks.append([1] * len(seq) + [0] * pad_len)
            
        # 3. 局域拷贝内存数据构造 Tensor/Array
        if self.backend == 'torch':
            import torch
            token_inputs = torch.tensor(padded_ids, dtype=torch.long)
            attention_mask = torch.tensor(masks, dtype=torch.float32)
            batch_embs = [torch.from_numpy(arr[batch_local_indices].copy()).float() for arr in self.active_chunk_embs]
        else:
            import mlx.core as mx
            token_inputs = mx.array(padded_ids)
            attention_mask = mx.array(masks)
            batch_embs = [mx.array(arr[batch_local_indices]) for arr in self.active_chunk_embs]
            
        return token_inputs, batch_embs, attention_mask

    def state_dict(self) -> Dict[str, Any]:
        return {
            "current_epoch": self.current_epoch,
            "active_macro_idx_ptr": getattr(self, "active_macro_idx_ptr", 0),
            "macro_chunk_indices": getattr(self, "macro_chunk_indices", np.array([])).tolist(),
            "active_micro_ptr": getattr(self, "active_micro_ptr", 0),  # 保存当前块内部的微观进度！
            "seed_snapshot": getattr(self, "seed", int(__import__("time").time()))  # 永久锁定随机种子
        }
        
    def load_state_dict(self, state: Dict[str, Any]):
        self.current_epoch = state["current_epoch"]
        self.active_macro_idx_ptr = state["active_macro_idx_ptr"]
        self.macro_chunk_indices = np.array(state["macro_chunk_indices"])
        
        print(f"[DataLoader] 🔌 自断点恢复：正处于 Epoch {self.current_epoch},  宏观 Chunk 指针 {self.active_macro_idx_ptr}/{len(self.macro_chunk_indices)}")

        # 核心漏洞修复法则：如果是从早期未记录种子与微观游标的破损失忆版本恢复（当前情况）
        # 强行拨动宏观指针进入下一个崭新纯净块，避免在一锅被错乱打散重组的烂粥里吃重复数据
        is_legacy_state = "seed_snapshot" not in state or "active_micro_ptr" not in state
        
        if is_legacy_state:
            self.active_macro_idx_ptr = min(self.active_macro_idx_ptr + 1, len(self.macro_chunk_indices) - 1)
            print(f"[DataLoader] ⚠️ 检测到历史遗留 Checkpoint 缺乏微秩序信息，为了避免重复喂入，已强行剥离并将指针拨至下一块干净数据：{self.active_macro_idx_ptr}。")
        else:
            # 未来恢复逻辑：一切数据时空完美复刻
            self.seed = state["seed_snapshot"]
            
        # 断点恢复：直接从历史断点启动下载，跳过 __init__ 的默认启动
        self._started = True
        self._start_prefetching(start_macro_ptr=self.active_macro_idx_ptr)
        self._pop_next_chunk()
        
        if not is_legacy_state:
            # 严格对准之前在这个块里吃过的条数（只有新形态拥有这个待遇）
            self.active_micro_ptr = state["active_micro_ptr"]
            print(f"[DataLoader] 📍 已还原微观读取游标与世界线乱序定律：从当前块的第 {self.active_micro_ptr} 条语句无缝继续训练！")
        else:
            # 遗留强行跳块版本下，微观进度自然是从 0 重新算起
            self.active_micro_ptr = 0

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
        
    def __next__(self) -> Tuple[List[Any], Any]:
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
                
        # 2. Pad to the length of the longest document in THIS specific batch (Dynamic Batching)
        max_len = max(lengths)
        
        # 2. Pad to the maximum document length in THIS specific batch
        padded_embs = [[] for _ in range(len(self.embs))]
        masks = []
        
        for b in range(len(batch_doc_indices)):
            l = lengths[b]
            pad_len = max_len - l
            
            for i in range(len(self.embs)):
                arr = raw_embs_list[i][b]
                embs_dim = arr.shape[-1]
                if pad_len > 0:
                    pad_array = np.zeros((pad_len, embs_dim), dtype=arr.dtype)
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
            import mlx.core as mx
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

class TextDocumentDataLoader:
    """
    DataLoader specifically for Phase 1 End-to-End.
    Reads `.parquet` text chunks, groups them into full documents (causal sequences),
    and yields 3D token tensors: (Batch, Max_Sentences, Max_Tokens).
    """
    def __init__(self, 
                 parquet_path: str,
                 tokenizer,
                 batch_size: int = 16,
                 max_episode_len: int = None,
                 max_seq_len: int = 256,
                 seed: int = 42,
                 backend: str = 'mlx'):
        
        self.backend = backend
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_episode_len = max_episode_len
        self.max_seq_len = max_seq_len
        
        print("Reading Document Texts and Chunk Counts...")
        df = pd.read_parquet(parquet_path)
        self.text_chunks = df['chunks'].explode().dropna().tolist()
        self.chunk_counts = df['chunk_count'].values
        
        self.doc_start_indices = np.concatenate(([0], np.cumsum(self.chunk_counts)[:-1]))
        self.total_docs = len(self.chunk_counts)
        print(f"Loaded strictly bounded {self.total_docs} semantic documents (Text only).")
        
        del df
        
        self.rng = np.random.default_rng(seed)
        self.current_epoch = 0
        self.batch_idx = 0
        
        self.indices = np.arange(self.total_docs)
        self.rng.shuffle(self.indices)
        self.num_batches = int(np.ceil(self.total_docs / self.batch_size))
        
    def __iter__(self):
        return self
        
    def __next__(self) -> Tuple[Any, Any, Any]:
        if self.batch_idx >= self.num_batches:
            self.current_epoch += 1
            self.batch_idx = 0
            self.rng.shuffle(self.indices)
            raise StopIteration
            
        start_idx = self.batch_idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.total_docs)
        
        batch_doc_indices = self.indices[start_idx:end_idx]
        
        batch_docs_sentences = []
        for doc_idx in batch_doc_indices:
            start_pos = self.doc_start_indices[doc_idx]
            count = self.chunk_counts[doc_idx]
            
            if self.max_episode_len is not None:
                eff_len = min(count, self.max_episode_len)
            else:
                eff_len = min(count, 256) # Safety cap
                
            sentences = self.text_chunks[start_pos : start_pos + eff_len]
            batch_docs_sentences.append(sentences)
            
        # 强制固定句子长度维度！这是为了防止 Apple MLX 的 @mx.compile 
        # 因为每一批次的 shape 不同而疯狂重新编译，最终导致图缓存撑爆统一内存 (Jetsam Kill)。
        max_sentences = self.max_episode_len if self.max_episode_len else 256
        
        # We will flatten to tokenize efficiently, then reshape
        flat_texts = []
        for doc in batch_docs_sentences:
            flat_texts.extend(doc)
            
        encoded_list = [self.tokenizer.encode(t, add_special_tokens=True)[:self.max_seq_len] for t in flat_texts]
        max_tokens = max(max((len(seq) for seq in encoded_list), default=1), 1)
        
        # Prepare 3D tensors: (B, max_sentences, max_tokens)
        padded_ids = np.zeros((len(batch_docs_sentences), max_sentences, max_tokens), dtype=np.int32)
        padded_ids.fill(self.tokenizer.pad_token_id)
        
        attention_mask = np.zeros((len(batch_docs_sentences), max_sentences, max_tokens), dtype=np.float32)
        sentence_mask = np.zeros((len(batch_docs_sentences), max_sentences), dtype=np.float32)
        
        flat_idx = 0
        for b, doc in enumerate(batch_docs_sentences):
            for s, _ in enumerate(doc):
                seq = encoded_list[flat_idx]
                seq_len = len(seq)
                padded_ids[b, s, :seq_len] = seq
                attention_mask[b, s, :seq_len] = 1.0
                sentence_mask[b, s] = 1.0
                flat_idx += 1
                
        if self.backend == 'torch':
            import torch
            ids_t = torch.tensor(padded_ids, dtype=torch.long)
            att_t = torch.tensor(attention_mask, dtype=torch.float32)
            sen_t = torch.tensor(sentence_mask, dtype=torch.float32)
        else:
            import mlx.core as mx
            ids_t = mx.array(padded_ids)
            att_t = mx.array(attention_mask)
            sen_t = mx.array(sentence_mask)
            
        self.batch_idx += 1
        return ids_t, att_t, sen_t
        
    def state_dict(self) -> Dict[str, Any]:
        return {"batch_idx": self.batch_idx, "current_epoch": self.current_epoch, "indices": self.indices.tolist()}
        
    def load_state_dict(self, state: Dict[str, Any]):
        self.batch_idx = state["batch_idx"]
        self.current_epoch = state["current_epoch"]
        self.indices = np.array(state["indices"])
