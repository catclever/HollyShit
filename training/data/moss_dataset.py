from .dialogue_dataset import GenericPairDataset

class MOSSDataset(GenericPairDataset):
    """
    Backwards-compatible alias for GenericPairDataset.
    Can be used interchangeably for MOSS and any prompt-target dataset.
    """
    def __init__(self, jsonl_path: str, tokenizer, seq_len: int = 256, **kwargs):
        super().__init__(
            data_path=jsonl_path,
            tokenizer=tokenizer,
            seq_len=seq_len,
            **kwargs
        )
