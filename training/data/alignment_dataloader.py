import numpy as np
try:
    import pandas as pd
except ImportError:
    pd = None

import json
import os
import glob
import re
import queue
import shutil
import threading
from typing import List, Tuple, Dict, Any, Optional


def _get_hf_token() -> Optional[str]:
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or ""
    tok = tok.strip()
    return tok if tok else None


def _download_hf_dataset_file(repo_id: str, filename: str, cache_dir: str) -> str:
    """
    Downloads a single dataset file from Hugging Face Hub directly into `cache_dir`
    (`local_dir=cache_dir` in huggingface_hub >= 0.23 to avoid duplicating multi-GB blobs,
    falling back to `cache_dir=cache_dir` on older versions).
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise RuntimeError("请安装 huggingface_hub: pip install huggingface_hub")

    hf_token = _get_hf_token()
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    try:
        return hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            local_dir=cache_dir,
            token=hf_token,
        )
    except TypeError:
        return hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            cache_dir=cache_dir,
            token=hf_token,
        )


def _cleanup_hf_download_metadata_for_path(file_path: str, roots: List[Optional[str]]) -> int:
    """
    Cleans `.cache/huggingface/download/<rel>.metadata`, `.incomplete`, and `.lock`
    artifacts associated with `file_path` under any root in `roots`.
    """
    removed = 0
    if not file_path:
        return removed
    abs_p = os.path.abspath(file_path)
    seen_roots = set()
    for root in roots:
        if not root:
            continue
        abs_root = os.path.abspath(root)
        if abs_root in seen_roots:
            continue
        seen_roots.add(abs_root)

        prefix = abs_root.rstrip(os.sep) + os.sep
        if not abs_p.startswith(prefix):
            continue
        rel = os.path.relpath(abs_p, abs_root)
        rel_dir = os.path.dirname(rel)
        base_fn = os.path.basename(rel)
        dl_meta_dir = os.path.join(abs_root, ".cache", "huggingface", "download", rel_dir)
        if not os.path.isdir(dl_meta_dir):
            continue
        try:
            for fn in os.listdir(dl_meta_dir):
                is_match = fn in {
                    f"{base_fn}.metadata",
                    f"{base_fn}.incomplete",
                    f"{base_fn}.lock",
                } or (
                    fn.startswith(base_fn + ".")
                    and fn.endswith((".metadata", ".incomplete", ".lock"))
                )
                if is_match:
                    try:
                        os.remove(os.path.join(dl_meta_dir, fn))
                        removed += 1
                    except OSError:
                        pass
        except OSError:
            pass
    return removed


class Phase1_MultiEmbDataLoader:
    """
    Dataloader that handles:
    1. Reading text chunks from Parquet (in-memory, fast enough for 7M strings),
       or skipping Parquet completely when `embedding_only=True`.
    2. Zero-copy / mmap loading of massive .npy embedding files.
    3. Dynamic tokenization & padding (skipped when `embedding_only=True`).
    4. Interrupt / Resume Checkpointing (Saving the exact shuffled order).
    """
    def __init__(self, 
                 parquet_path: str,
                 emb_paths: List[str],
                 tokenizer=None,
                 batch_size: int = 256,
                 max_seq_len: int = 512,
                 shuffle: bool = True,
                 seed: int = 42,
                 backend: str = 'mlx',
                 embedding_only: bool = False):
        
        self.backend = backend
        
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.shuffle = shuffle
        self.seed = seed if seed is not None else int(__import__("time").time())
        self.tokenizer = tokenizer
        self.embedding_only = bool(embedding_only)
        
        if self.embedding_only:
            self.text_chunks = None
            if not emb_paths:
                raise ValueError("emb_paths must be non-empty when embedding_only=True")
            print("Memory-mapping embedding arrays (embedding_only=True)...")
            self.embs = []
            self.total_samples = None
            for path in emb_paths:
                arr = np.load(path, mmap_mode='r')
                if self.total_samples is None:
                    self.total_samples = int(arr.shape[0])
                assert int(arr.shape[0]) == self.total_samples, (
                    f"Shape mismatch in {path}: {arr.shape[0]} != {self.total_samples}"
                )
                self.embs.append(arr)
            print(
                f"[DataLoader] ⚡ 纯向量 mmap 模式 (embedding_only=True): "
                f"免加载 Parquet 文本池，总条数: {self.total_samples}"
            )
        else:
            # 1. Load Text Data
            if pd is None:
                import pandas as _pd
                globals()["pd"] = _pd
            print(f"Loading Text Parquet from {parquet_path}...")
            df = pd.read_parquet(parquet_path)
            self.text_chunks = df['chunks'].explode().dropna().tolist()
            self.total_samples = len(self.text_chunks)
            print(f"Loaded {self.total_samples} text chunks into memory.")
            del df
            
            # 2. Memory-Map the Numpy Embeddings
            print("Memory-mapping embedding arrays...")
            self.embs = []
            for path in emb_paths:
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
            self.current_epoch += 1
            self.batch_idx = 0
            if self.shuffle:
                self.rng.shuffle(self.indices)
            raise StopIteration
            
        start_idx = self.batch_idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.total_samples)
        batch_indices = self.indices[start_idx:end_idx]
        
        if self.embedding_only or self.text_chunks is None or self.tokenizer is None:
            if self.backend == 'torch':
                import torch
                batch_embs = [torch.from_numpy(np.array(arr[batch_indices])).float() for arr in self.embs]
            else:
                import mlx.core as mx
                batch_embs = [mx.array(arr[batch_indices]) for arr in self.embs]
            self.batch_idx += 1
            return None, batch_embs, None

        batch_texts = [self.text_chunks[i] for i in batch_indices]
        encoded_list = [self.tokenizer.encode(t, add_special_tokens=True)[:self.max_seq_len] for t in batch_texts]
        max_len = max(len(seq) for seq in encoded_list)
        
        padded_ids = []
        masks = []
        for seq in encoded_list:
            pad_len = max_len - len(seq)
            padded_ids.append(seq + [self.tokenizer.pad_token_id] * pad_len)
            masks.append([1] * len(seq) + [0] * pad_len)
            
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

    def state_dict(self) -> Dict[str, Any]:
        return {
            "current_epoch": self.current_epoch,
            "batch_idx": self.batch_idx,
            "indices": self.indices.tolist()
        }
        
    def load_state_dict(self, state: Dict[str, Any]):
        self.current_epoch = state["current_epoch"]
        self.batch_idx = state["batch_idx"]
        self.indices = np.array(state["indices"])
        print(f"Resumed Dataloader from Epoch {self.current_epoch}, Batch {self.batch_idx}/{self.num_batches}")


class Phase1_ChunkedNpzDataLoader:
    """
    专门适配从 ModelScope / Google Drive / Hugging Face Hub / 本地目录拉取的 `.npz` 切片矩阵设计的统一双层乱序 DataLoader。
    机制：
    1. 不吃内存：只在内存驻留当前正在训练的任务块的特征（比如每次只把某个 50w 条特征解压进 RAM）。
    2. 双层乱序 (Double Shuffle)：每个 Epoch 开始时打乱所有 Chunk 文件的流转顺序；在读取单个 Chunk 时，在 Chunk 内部将其局部 batch 完全乱序。
    3. 纯向量流式模式 (`embedding_only=True`)：免加载本地 Parquet 语料与逐批分词，直接由远端/本地 `.npz` 分块边界推导总样本数并流式产出多师向量。
    4. 统一多数据源架构：
       - `ms` (默认): ModelScope 零配置按需流式串流 (catclever/emb_npy)
       - `gdrive`: Google Drive 持久目录挂载源，后台异步向本地 NVMe 临时目录 (`cache_dir`) 暂存 (Staging)，消费完立即清除本地暂存，严禁误删云盘原始文件
       - `hf`: Hugging Face Hub 按需流式串流 (drunkencat/emb_npy)
       - `local`: 本地磁盘直接读取，不走临时暂存复制
    """
    def __init__(self, 
                 parquet_path: str,
                 models: List[str],
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
                 auto_cleanup: bool = True,
                 embedding_only: bool = False,
                 hf_repo_id: str = None,
                 data_source: str = "ms",
                 gdrive_dir: Optional[str] = None):
        
        self.backend = backend
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.shuffle = shuffle
        self.seed = seed if seed is not None else int(__import__("time").time())
        self.models = models
        self.chunk_patterns = chunk_patterns or {}
        self.tokenizer = tokenizer
        self.cache_dir = cache_dir
        self.auto_cleanup = auto_cleanup
        self.embedding_only = bool(embedding_only)
        self.npz_dir = local_npz_dir
        self.gdrive_dir = gdrive_dir

        # 规范化 data_source: ms (默认), gdrive, hf, local
        ds_raw = (data_source or "ms").strip().lower()
        if ds_raw in ("ms", "modelscope"):
            self.data_source = "ms"
        elif ds_raw in ("gdrive", "google_drive", "googledrive"):
            self.data_source = "gdrive"
        elif ds_raw in ("hf", "huggingface", "hugging_face"):
            self.data_source = "hf"
        elif ds_raw in ("local", "dir", "local_dir"):
            self.data_source = "local"
        else:
            raise ValueError(
                f"[DataLoader] 未知数据源 data_source={data_source!r}。仅支持: 'ms', 'gdrive', 'hf', 'local'"
            )

        self.ms_repo_id = ms_repo_id or ("catclever/emb_npy" if self.data_source == "ms" else None)
        self.hf_repo_id = hf_repo_id or ("drunkencat/emb_npy" if self.data_source == "hf" else None)
        
        # 3. 提取对应的全体文本数据 (embedding_only=True 时完全免加载 Parquet)
        if self.embedding_only:
            self.text_chunks = None
            self.total_samples = 0
        else:
            if pd is None:
                import pandas as _pd
                globals()["pd"] = _pd
            if parquet_path and os.path.isfile(parquet_path):
                print(f"[DataLoader] 正在装载基准文本体系: {parquet_path}...")
                df = pd.read_parquet(parquet_path)
                self.text_chunks = df['chunks'].explode().dropna().tolist()
                self.total_samples = len(self.text_chunks)
                print(f"[DataLoader] 成功吃进文本池: {self.total_samples} 条")
                del df
            else:
                self.text_chunks = []
                self.total_samples = 0
        
        # 1. 校验所选数据源所需依赖与路径
        if self.data_source == "ms":
            try:
                from modelscope.hub.file_download import dataset_file_download  # noqa: F401
                print(f"[DataLoader] 已启用 ModelScope 默认流式串流模式: {self.ms_repo_id}...")
            except ImportError:
                raise RuntimeError("请安装 modelscope: pip install modelscope")
        elif self.data_source == "gdrive":
            if not self.gdrive_dir or not os.path.isdir(self.gdrive_dir):
                raise FileNotFoundError(
                    f"[DataLoader] Google Drive 数据源目录不存在：{self.gdrive_dir}。请检查云盘路径或挂载状态！"
                )
            print(
                f"[DataLoader] 已启用 Google Drive 离线云盘模式 (NVMe 暂存: {self.cache_dir}): {self.gdrive_dir}..."
            )
        elif self.data_source == "hf":
            try:
                from huggingface_hub import HfApi, hf_hub_download  # noqa: F401
                print(f"[DataLoader] 已启用 Hugging Face Hub 极速按需流式串流模式: {self.hf_repo_id}...")
            except ImportError:
                raise RuntimeError("请安装 huggingface_hub: pip install huggingface_hub")
        elif self.data_source == "local":
            if not self.npz_dir or not os.path.isdir(self.npz_dir):
                raise ValueError(f"[DataLoader] local 模式必须提供有效的 local_npz_dir，当前为: {self.npz_dir!r}")
            print(f"[DataLoader] 已启用本地目录直接读取模式: {self.npz_dir}...")

        # 2. 按模型分别探测物理分块，按 start 索引取交集（容错不同模型尾部 end 不同）
        self.chunk_bounds = []
        self._per_model_bounds = {}  # {model_name: {start: end}}
        try:
            if self.data_source == "ms":
                from modelscope.hub.api import HubApi
                api = HubApi()
                ms_token = (
                    os.environ.get("MODELSCOPE_API_TOKEN")
                    or os.environ.get("MS_TOKEN")
                    or ""
                ).strip()
                if ms_token:
                    try:
                        api.login(ms_token)
                        print("[DataLoader] 🔑 已应用 ModelScope Access Token 鉴权。")
                    except Exception as login_err:
                        print(f"[DataLoader] ⚠️ ModelScope Token 登录忽略 (将继续匿名拉取): {login_err}")
                res = api.get_dataset_files(self.ms_repo_id, recursive=True)
                chunk_files = [f.get("Path", f.get("Name", "")) for f in res]
            elif self.data_source == "gdrive":
                chunk_files = glob.glob(os.path.join(self.gdrive_dir, "**", "*.npz"), recursive=True)
            elif self.data_source == "hf":
                from huggingface_hub import HfApi
                hf_token = _get_hf_token()
                if hf_token:
                    print("[DataLoader] 🔑 已应用 Hugging Face Token 鉴权。")
                chunk_files = list(
                    HfApi(token=hf_token).list_repo_files(
                        repo_id=self.hf_repo_id, repo_type="dataset"
                    )
                )
            elif self.data_source == "local":
                chunk_files = glob.glob(os.path.join(self.npz_dir, "**", "*.npz"), recursive=True)
            else:
                chunk_files = []
            
            # 按模型归类，各自收集 start → end 的映射
            for model_name in self.models:
                pattern = self.chunk_patterns.get(model_name, "")
                prefix = pattern.split("/")[0] if "/" in pattern else model_name
                start_to_end = {}
                for f in chunk_files:
                    f_normalized = f.replace("\\", "/")
                    base_fn = os.path.basename(f_normalized)
                    in_subdir = f_normalized.startswith(prefix + "/") or f"/{prefix}/" in f_normalized
                    in_flat = ("/" not in pattern) and base_fn.startswith(prefix)
                    if f_normalized.endswith(".npz") and (in_subdir or in_flat):
                        match = re.search(r'chunk_(\d+)_(\d+)\.npz', base_fn)
                        if match:
                            start_to_end[int(match.group(1))] = int(match.group(2))
                self._per_model_bounds[model_name] = start_to_end
                print(f"[DataLoader]   {model_name}: {len(start_to_end)} 个分块")
            
            if self._per_model_bounds:
                common_starts = set.intersection(*[set(d.keys()) for d in self._per_model_bounds.values()])
                for s in sorted(common_starts):
                    min_end = min(self._per_model_bounds[m][s] for m in self.models)
                    self.chunk_bounds.append((s, min_end))
            
            print(f"[DataLoader] 多模型安全交集（按start容错）：{len(self.chunk_bounds)} 个共享分块。")
        except Exception as e:
            print(f"[DataLoader] 物理探测失败，回退到严格数学计算: {e}")
            self.chunk_bounds = []
            for start_idx in range(0, self.total_samples, chunk_size):
                 end_idx = min(start_idx + chunk_size, self.total_samples)
                 self.chunk_bounds.append((start_idx, end_idx))

        if self.embedding_only:
            if not self.chunk_bounds:
                raise RuntimeError(
                    f"[DataLoader] embedding_only=True 模式下未探测到任何可用的 .npz 分块 "
                    f"(data_source={self.data_source!r}, ms_repo_id={self.ms_repo_id!r}, "
                    f"gdrive_dir={self.gdrive_dir!r}, hf_repo_id={self.hf_repo_id!r}, local_npz_dir={local_npz_dir!r})。"
                )
            self.total_samples = int(self.chunk_bounds[-1][1])
            print(
                f"[DataLoader] ⚡ 纯向量流式模式 (embedding_only=True): "
                f"免加载 Parquet 文本池，由分块边界推得总条数: {self.total_samples}"
            )
        
        # 4. 统计与 Epoch 控制变量
        self.rng = np.random.default_rng(seed)
        self.current_epoch = 0
        
        # 宏观控制表 (Chunks)
        self.macro_chunk_indices = np.arange(len(self.chunk_bounds))
        if self.shuffle:
             self.rng.shuffle(self.macro_chunk_indices)
             
        self.active_macro_idx_ptr = 0
        
        # 微观控制表 (内部流)
        self.active_chunk_embs = None
        self.active_micro_indices = []
        self.active_micro_ptr = 0
        self._prev_chunk_paths = []
        
        # 异步流水线 (Async Prefetch)
        self.prefetch_queue = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self._started = False
        
        if not lazy_start:
            self._do_start()

    @staticmethod
    def _download_hf_dataset_file(repo_id: str, filename: str, cache_dir: str) -> str:
        return _download_hf_dataset_file(repo_id=repo_id, filename=filename, cache_dir=cache_dir)

    def _do_start(self):
        """真正启动后台下载，首次加载第一个块"""
        if self._started:
            return
        self._started = True
        self._start_prefetching(start_macro_ptr=0)
        self._pop_next_chunk()

    def _cleanup_cached_files(self, paths: list):
        """
        清理已经消费完毕的缓存文件（含软链接真实 blob 与 .metadata/.incomplete），
        铁律防线：严格禁止删除 Google Drive 原始文件或 local 源文件！
        """
        cleaned = 0
        roots = [self.cache_dir, getattr(self, "npz_dir", None)]
        gdrive_abs = os.path.abspath(self.gdrive_dir) if getattr(self, "gdrive_dir", None) else None
        local_abs = os.path.abspath(self.npz_dir) if getattr(self, "npz_dir", None) else None

        for p in paths:
            if not p:
                continue
            abs_p = os.path.abspath(p)
            real_p = os.path.realpath(p)

            # 铁律拦截：绝不能删除 gdrive_dir 或 local 模式下的 npz_dir 原始文件！
            if gdrive_abs:
                if abs_p == gdrive_abs or abs_p.startswith(gdrive_abs + os.sep):
                    continue
                if real_p and (real_p == gdrive_abs or real_p.startswith(gdrive_abs + os.sep)):
                    continue
            if self.data_source == "local" and local_abs:
                if abs_p == local_abs or abs_p.startswith(local_abs + os.sep):
                    continue
                if real_p and (real_p == local_abs or real_p.startswith(local_abs + os.sep)):
                    continue

            try:
                if os.path.islink(p):
                    real_target = os.path.realpath(p)
                    try:
                        os.remove(p)
                        cleaned += 1
                    except OSError:
                        pass
                    if real_target and real_target != abs_p and os.path.exists(real_target):
                        if not (gdrive_abs and (real_target == gdrive_abs or real_target.startswith(gdrive_abs + os.sep))):
                            try:
                                os.remove(real_target)
                                cleaned += 1
                            except OSError:
                                pass
                elif os.path.exists(p):
                    os.remove(p)
                    cleaned += 1
            except OSError:
                pass
            cleaned += _cleanup_hf_download_metadata_for_path(p, roots)
        if cleaned:
            print(f"[DataLoader] 🧹 已清理 {cleaned} 个已消费缓存文件/元数据")

    def _drain_and_cleanup_prefetch_queue(self):
        """排空预取队列，并在 auto_cleanup=True 时清理队列中尚未消费分块的缓存文件"""
        while True:
            try:
                queued = self.prefetch_queue.get_nowait()
            except queue.Empty:
                break
            if (
                self.auto_cleanup
                and isinstance(queued, dict)
                and queued.get("cached_paths")
            ):
                self._cleanup_cached_files(queued["cached_paths"])

    def close(self):
        """安全终止后台预取线程，并清理预取队列与当前分块遗留的缓存文件"""
        self.stop_event.set()
        self._drain_and_cleanup_prefetch_queue()
        t = getattr(self, "prefetch_thread", None)
        if t is not None and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=5.0)
        self._drain_and_cleanup_prefetch_queue()
        if self.auto_cleanup and getattr(self, "_prev_chunk_paths", None):
            self._cleanup_cached_files(self._prev_chunk_paths)
            self._prev_chunk_paths = []

    def _start_prefetching(self, start_macro_ptr):
        """挂起并汇合旧线程，清理队列残骸后启动全新后台线程，从指定的 macro_ptr 开始默默搬取"""
        self.stop_event.set()
        self._drain_and_cleanup_prefetch_queue()
        old_thread = getattr(self, "prefetch_thread", None)
        if (
            old_thread is not None
            and old_thread.is_alive()
            and old_thread is not threading.current_thread()
        ):
            old_thread.join(timeout=3.0)
        self._drain_and_cleanup_prefetch_queue()

        self.stop_event = threading.Event()
        worker_stop_event = self.stop_event
        self.prefetch_thread = threading.Thread(
            target=self._prefetch_worker,
            args=(start_macro_ptr, worker_stop_event),
            daemon=True,
        )
        self.prefetch_thread.start()
        
    def _prefetch_worker(self, start_macro_ptr, stop_ev: Optional[threading.Event] = None):
        """高强度纯后台下载/解压打工仔，永远走在 GPU 的前面"""
        if stop_ev is None:
            stop_ev = self.stop_event
        worker_macro_ptr = start_macro_ptr
        while not stop_ev.is_set():
            if worker_macro_ptr >= len(self.macro_chunk_indices):
                while not stop_ev.is_set():
                    try:
                        self.prefetch_queue.put(None, timeout=0.2)
                        break
                    except queue.Full:
                        continue
                return

            chunk_idx = self.macro_chunk_indices[worker_macro_ptr]
            start_idx, end_idx = self.chunk_bounds[chunk_idx]
            cur_chunk_size = end_idx - start_idx
            cached_paths = []
            
            try:
                active_chunk_embs = []
                for model_name in self.models:
                    if stop_ev.is_set():
                        if self.auto_cleanup and cached_paths:
                            self._cleanup_cached_files(cached_paths)
                        return
                    pattern = self.chunk_patterns.get(
                        model_name, f"{model_name}_chunk_{'{start:07d}'}_{'{end:07d}'}.npz"
                    )
                    model_end = self._per_model_bounds.get(model_name, {}).get(start_idx, end_idx)
                    cand_name = pattern.format(start=start_idx, end=model_end)
                    
                    if self.data_source == "ms":
                        from modelscope.hub.file_download import dataset_file_download
                        path = dataset_file_download(self.ms_repo_id, cand_name, cache_dir=self.cache_dir)
                        cached_paths.append(path)
                    elif self.data_source == "gdrive":
                        # Google Drive 数据源：从 gdrive_dir 读取并暂存到 NVMe cache_dir
                        gdrive_src = os.path.join(self.gdrive_dir, cand_name)
                        if not os.path.exists(gdrive_src):
                            flat_src = os.path.join(self.gdrive_dir, os.path.basename(cand_name))
                            if os.path.exists(flat_src):
                                gdrive_src = flat_src
                            else:
                                raise FileNotFoundError(f"[DataLoader] Google Drive 源分块缺失：{gdrive_src}")
                        
                        abs_cache = os.path.abspath(self.cache_dir)
                        abs_gdrive = os.path.abspath(self.gdrive_dir)
                        if abs_cache != abs_gdrive:
                            staged_path = os.path.join(self.cache_dir, cand_name)
                            os.makedirs(os.path.dirname(staged_path), exist_ok=True)
                            if (not os.path.exists(staged_path)) or (
                                os.path.getsize(staged_path) != os.path.getsize(gdrive_src)
                            ):
                                shutil.copy2(gdrive_src, staged_path)
                            path = staged_path
                            cached_paths.append(staged_path)
                        else:
                            # 缓存目录与云盘相同，直接读取，严禁加入 cached_paths 以免被清理
                            path = gdrive_src
                    elif self.data_source == "hf":
                        path = _download_hf_dataset_file(self.hf_repo_id, cand_name, self.cache_dir)
                        cached_paths.append(path)
                    elif self.data_source == "local":
                        path = os.path.join(self.npz_dir, cand_name)
                        if not os.path.exists(path):
                            flat_path = os.path.join(self.npz_dir, os.path.basename(cand_name))
                            if os.path.exists(flat_path):
                                path = flat_path
                            else:
                                raise FileNotFoundError(f"后台打工仔抓取失败！本地目标块缺失：{path}")
                        # 本地目录持久文件，严禁加入 cached_paths

                    if stop_ev.is_set():
                        if self.auto_cleanup and cached_paths:
                            self._cleanup_cached_files(cached_paths)
                        return

                    arr = np.load(path)['features']
                    if arr.shape[0] > cur_chunk_size:
                        arr = arr[:cur_chunk_size]
                    active_chunk_embs.append(arr)
                    
                active_micro_indices = np.arange(cur_chunk_size)
                
                if self.shuffle:
                    rng = np.random.default_rng(self.seed + worker_macro_ptr + self.current_epoch)
                    rng.shuffle(active_micro_indices)
                    
                payload = {
                    "embs": active_chunk_embs,
                    "micro_indices": active_micro_indices,
                    "global_start": start_idx,
                    "macro_idx_ptr": worker_macro_ptr,
                    "cached_paths": cached_paths
                }
                queued_ok = False
                while not stop_ev.is_set():
                    try:
                        self.prefetch_queue.put(payload, timeout=0.2)
                        queued_ok = True
                        break
                    except queue.Full:
                        continue
                if not queued_ok:
                    if self.auto_cleanup and cached_paths:
                        self._cleanup_cached_files(cached_paths)
                    return
                worker_macro_ptr += 1
            except Exception as e:
                if self.auto_cleanup and cached_paths:
                    self._cleanup_cached_files(cached_paths)
                if stop_ev.is_set():
                    return
                print(f"\n[DataLoader] 后台打工仔抓取/加载失败 (网络可能断开了？): {e}")
                print(f"[DataLoader] 这不是致命错误！打工仔将在 10 秒后原地重新尝试拉取 Chunk {chunk_idx}...")
                stop_ev.wait(10.0)
                continue

    def _pop_next_chunk(self):
        """主线程调用的吸星大法，把工人装好的箱子拿走"""
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
        if self.active_micro_ptr >= len(self.active_micro_indices):
            try:
                self._pop_next_chunk()
            except StopIteration:
                self.current_epoch += 1
                if self.shuffle:
                    self.rng.shuffle(self.macro_chunk_indices)
                self._start_prefetching(start_macro_ptr=0)
                self._pop_next_chunk()
                raise StopIteration
            
        fetch_len = min(self.batch_size, len(self.active_micro_indices) - self.active_micro_ptr)
        batch_local_indices = self.active_micro_indices[self.active_micro_ptr : self.active_micro_ptr + fetch_len]
        self.active_micro_ptr += fetch_len
        
        if self.embedding_only or self.text_chunks is None or self.tokenizer is None:
            if self.backend == 'torch':
                import torch
                batch_embs = [torch.from_numpy(arr[batch_local_indices].copy()).float() for arr in self.active_chunk_embs]
            else:
                import mlx.core as mx
                batch_embs = [mx.array(arr[batch_local_indices]) for arr in self.active_chunk_embs]
            return None, batch_embs, None

        batch_global_indices = self.active_global_start + batch_local_indices
        batch_texts = [self.text_chunks[i] for i in batch_global_indices]
        
        encoded_list = [self.tokenizer.encode(t, add_special_tokens=True)[:self.max_seq_len] for t in batch_texts]
        max_len = max(max(len(seq) for seq in encoded_list), 1)
        
        padded_ids = []
        masks = []
        for seq in encoded_list:
            pad_len = max_len - len(seq)
            padded_ids.append(seq + [self.tokenizer.pad_token_id] * pad_len)
            masks.append([1] * len(seq) + [0] * pad_len)
            
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
            "active_micro_ptr": getattr(self, "active_micro_ptr", 0),
            "seed_snapshot": getattr(self, "seed", int(__import__("time").time()))
        }
        
    def load_state_dict(self, state: Dict[str, Any]):
        self.current_epoch = state["current_epoch"]
        self.active_macro_idx_ptr = state["active_macro_idx_ptr"]
        self.macro_chunk_indices = np.array(state["macro_chunk_indices"])
        
        print(f"[DataLoader] 🔌 自断点恢复：正处于 Epoch {self.current_epoch},  宏观 Chunk 指针 {self.active_macro_idx_ptr}/{len(self.macro_chunk_indices)}")

        is_legacy_state = "seed_snapshot" not in state or "active_micro_ptr" not in state
        
        if is_legacy_state:
            self.active_macro_idx_ptr = min(self.active_macro_idx_ptr + 1, len(self.macro_chunk_indices) - 1)
            print(f"[DataLoader] ⚠️ 检测到历史遗留 Checkpoint 缺乏微秩序信息，已拨至下一块干净数据：{self.active_macro_idx_ptr}。")
        else:
            self.seed = state["seed_snapshot"]
            
        self._started = True
        self._start_prefetching(start_macro_ptr=self.active_macro_idx_ptr)
        self._pop_next_chunk()
        
        if not is_legacy_state:
            self.active_micro_ptr = state["active_micro_ptr"]
            print(f"[DataLoader] 📍 已还原微观读取游标与世界线乱序定律：从当前块的第 {self.active_micro_ptr} 条语句无缝继续训练！")
        else:
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
        
        if pd is None:
            import pandas as _pd
            globals()["pd"] = _pd
        print("Reading Document Chunk Counts...")
        df = pd.read_parquet(parquet_path, columns=['chunk_count'])
        self.chunk_counts = df['chunk_count'].values
        
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
        
        raw_embs_list = [[] for _ in range(len(self.embs))]
        lengths = []
        
        for doc_idx in batch_doc_indices:
            start_pos = self.doc_start_indices[doc_idx]
            count = self.chunk_counts[doc_idx]
            
            if self.max_episode_len is not None:
                eff_len = min(count, self.max_episode_len)
            else:
                eff_len = min(count, 256)
                
            lengths.append(eff_len)
            
            for i in range(len(self.embs)):
                raw_embs_list[i].append(np.array(self.embs[i][start_pos : start_pos+eff_len]))
                
        max_len = max(lengths)
        
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
        
        if pd is None:
            import pandas as _pd
            globals()["pd"] = _pd
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
                eff_len = min(count, 256)
                
            sentences = self.text_chunks[start_pos : start_pos + eff_len]
            batch_docs_sentences.append(sentences)
            
        max_sentences = self.max_episode_len if self.max_episode_len else 256
        
        flat_texts = []
        for doc in batch_docs_sentences:
            flat_texts.extend(doc)
            
        encoded_list = [self.tokenizer.encode(t, add_special_tokens=True)[:self.max_seq_len] for t in flat_texts]
        max_tokens = max(max((len(seq) for seq in encoded_list), default=1), 1)
        
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
