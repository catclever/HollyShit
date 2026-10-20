import os
import glob
import json
from typing import List, Union, Optional, Tuple, Dict, Any
import torch
from torch.utils.data import Dataset

class GenericPairDataset(Dataset):
    """
    A unified, highly flexible dataset loader for pair/dialogue tasks (Prompt -> Target).
    Supports:
      - Multiple formats: JSONL, JSON, Parquet
      - Multiple sources: Single file, list of files, directories, glob patterns
      - Auto-detection of dialogue/instruction keys (prompt/target, instruction/output, etc.)
      - Multi-turn conversation format (OpenAI / ShareGPT 'messages' or 'conversations')
      - On-the-fly tokenization (compatible with multi-worker DataLoader)
    """

    PROMPT_CANDIDATE_KEYS = [
        "prompt", "instruction", "input", "query", "user", "human", 
        "question", "source", "context", "text_a"
    ]
    TARGET_CANDIDATE_KEYS = [
        "target", "output", "response", "assistant", "bot", 
        "answer", "completion", "text_b"
    ]

    def __init__(
        self,
        data_path: Union[str, List[str]],
        tokenizer,
        seq_len: int = 256,
        prompt_seq_len: Optional[int] = None,
        target_seq_len: Optional[int] = None,
        prompt_key: Optional[str] = None,
        target_key: Optional[str] = None,
        max_samples: Optional[int] = None,
        min_target_len: int = 0,
        verbose: bool = True
    ):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.prompt_seq_len = prompt_seq_len if prompt_seq_len is not None else seq_len
        self.target_seq_len = target_seq_len if target_seq_len is not None else seq_len
        self.prompt_key = prompt_key
        self.target_key = target_key
        self.max_samples = max_samples
        self.min_target_len = min_target_len
        self.verbose = verbose

        self.samples: List[Tuple[str, str]] = []
        self._load_data(data_path)

        if self.verbose:
            print(f"✅ Loaded {len(self.samples)} dialogue pairs successfully.")

    def _collect_file_paths(self, data_path: Union[str, List[str]]) -> List[str]:
        raw_paths = [data_path] if isinstance(data_path, str) else data_path
        files = []
        for path in raw_paths:
            if os.path.isdir(path):
                for ext in ["*.jsonl", "*.json", "*.parquet"]:
                    files.extend(glob.glob(os.path.join(path, ext)))
            elif "*" in path or "?" in path:
                files.extend(glob.glob(path))
            elif os.path.isfile(path):
                files.append(path)
            else:
                if self.verbose:
                    print(f"⚠️ Warning: Path not found: {path}")
        return sorted(list(set(files)))

    def _extract_pair_from_dict(self, item: Dict[str, Any]) -> Optional[Tuple[str, str]]:
        if not isinstance(item, dict):
            return None

        # 1. Check for OpenAI/ShareGPT message lists
        for conv_key in ["messages", "conversations"]:
            if conv_key in item and isinstance(item[conv_key], list):
                turns = item[conv_key]
                user_text, assistant_text = "", ""
                for turn in turns:
                    if not isinstance(turn, dict):
                        continue
                    role = turn.get("role", turn.get("from", "")).lower()
                    content = turn.get("content", turn.get("value", ""))
                    if role in ["user", "human"]:
                        user_text = content
                    elif role in ["assistant", "bot", "gpt", "model"]:
                        assistant_text = content
                if user_text and assistant_text:
                    return str(user_text), str(assistant_text)

        # 2. Check explicitly provided keys
        if self.prompt_key and self.target_key:
            if self.prompt_key in item and self.target_key in item:
                return str(item[self.prompt_key]), str(item[self.target_key])

        # 3. Auto-detect keys from candidates
        found_prompt_key = None
        if self.prompt_key and self.prompt_key in item:
            found_prompt_key = self.prompt_key
        else:
            for k in self.PROMPT_CANDIDATE_KEYS:
                if k in item and item[k]:
                    found_prompt_key = k
                    break

        found_target_key = None
        if self.target_key and self.target_key in item:
            found_target_key = self.target_key
        else:
            for k in self.TARGET_CANDIDATE_KEYS:
                if k in item and item[k]:
                    found_target_key = k
                    break

        if found_prompt_key and found_target_key:
            p_str, t_str = str(item[found_prompt_key]), str(item[found_target_key])
            if self.min_target_len > 0 and len(t_str.strip()) < self.min_target_len:
                return None
            return p_str, t_str

        return None

    def _load_data(self, data_path: Union[str, List[str]]):
        files = self._collect_file_paths(data_path)
        if not files:
            raise FileNotFoundError(f"No valid data files found from path(s): {data_path}")

        for file_path in files:
            if self.max_samples and len(self.samples) >= self.max_samples:
                break

            if self.verbose:
                print(f"📖 Reading dataset file: {file_path} ...")

            if file_path.endswith(".jsonl"):
                self._load_jsonl(file_path)
            elif file_path.endswith(".parquet"):
                self._load_parquet(file_path)
            elif file_path.endswith(".json"):
                self._load_json(file_path)

    def _load_jsonl(self, file_path: str):
        with open(file_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                if self.max_samples and len(self.samples) >= self.max_samples:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    pair = self._extract_pair_from_dict(obj)
                    if pair:
                        self.samples.append(pair)
                except Exception:
                    continue

    def _load_json(self, file_path: str):
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                if isinstance(data, list):
                    for obj in data:
                        if self.max_samples and len(self.samples) >= self.max_samples:
                            break
                        pair = self._extract_pair_from_dict(obj)
                        if pair:
                            self.samples.append(pair)
            except Exception as e:
                if self.verbose:
                    print(f"⚠️ Error reading JSON file {file_path}: {e}")

    def _load_parquet(self, file_path: str):
        try:
            import pyarrow.parquet as pq
            table = pq.read_table(file_path)
            # Find prompt and target columns
            col_names = table.column_names
            prompt_col = self.prompt_key or next((c for c in self.PROMPT_CANDIDATE_KEYS if c in col_names), None)
            target_col = self.target_key or next((c for c in self.TARGET_CANDIDATE_KEYS if c in col_names), None)

            if not prompt_col or not target_col:
                if self.verbose:
                    print(f"⚠️ Parquet file {file_path} missing prompt/target columns (available: {col_names})")
                return

            prompts = table[prompt_col].to_pylist()
            targets = table[target_col].to_pylist()
            for p, t in zip(prompts, targets):
                if self.max_samples and len(self.samples) >= self.max_samples:
                    break
                if p and t:
                    self.samples.append((str(p), str(t)))
        except Exception as e:
            if self.verbose:
                print(f"⚠️ Error reading Parquet file {file_path}: {e}")

    def __len__(self) -> int:
        return len(self.samples)

    def _pad_trunc(self, ids: List[int], target_len: int) -> List[int]:
        if len(ids) > target_len:
            return ids[:target_len]
        return ids + [self.tokenizer.pad_token_id] * (target_len - len(ids))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        prompt_text, target_text = self.samples[idx]

        w_prompt = self.tokenizer.encode(prompt_text, add_special_tokens=True)
        w_target = self.tokenizer.encode(target_text, add_special_tokens=True)

        w_prompt = self._pad_trunc(w_prompt, self.prompt_seq_len)
        w_target = self._pad_trunc(w_target, self.target_seq_len)

        return (
            torch.tensor(w_prompt, dtype=torch.long),
            torch.tensor(w_target, dtype=torch.long)
        )

# Convenient aliases
DialogueDataset = GenericPairDataset
