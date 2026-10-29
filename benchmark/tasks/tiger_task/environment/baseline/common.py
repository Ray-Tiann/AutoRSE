from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[1]
UPSTREAM = Path(__file__).resolve().parent / "upstream"
sys.path.insert(0, str(UPSTREAM))


def dataset_dir() -> Path:
    return Path(os.environ.get("TIGER_DATA_DIR", str(WORKSPACE / "datasets" / "beauty")))


def input_dir() -> Path:
    return dataset_dir() / "input"


def working_dir() -> Path:
    path = dataset_dir() / "working"
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass
class SequenceStore:
    user_ids: np.ndarray
    offsets: np.ndarray
    items: np.ndarray

    def rows_for(self, requested: np.ndarray) -> np.ndarray:
        rows = np.searchsorted(self.user_ids, requested)
        if np.any(rows >= len(self.user_ids)) or not np.array_equal(
            self.user_ids[rows], requested
        ):
            raise ValueError("Unknown or out-of-order user IDs")
        return rows

    def sequence(self, row: int) -> np.ndarray:
        return self.items[self.offsets[row] : self.offsets[row + 1]]


def load_sequences() -> SequenceStore:
    root = input_dir() / "train_sequences"
    return SequenceStore(
        user_ids=np.load(str(root / "user_ids.npy"), mmap_mode="r"),
        offsets=np.load(str(root / "offsets.npy"), mmap_mode="r"),
        items=np.load(str(root / "items.npy"), mmap_mode="r"),
    )


def load_metadata() -> dict:
    return json.loads((input_dir() / "metadata.json").read_text(encoding="utf-8"))


def load_semantic_ids() -> np.ndarray:
    candidate = working_dir() / "item_semantic_ids.npy"
    path = candidate if candidate.exists() else input_dir() / "item_semantic_ids.npy"
    return np.load(str(path), mmap_mode="r")


def offset_codes(codes: np.ndarray, codebook_size: int = 256) -> np.ndarray:
    offsets = np.arange(codes.shape[-1], dtype=np.int64) * codebook_size + 1
    return np.asarray(codes, dtype=np.int64) + offsets


def make_inputs(
    histories: list[np.ndarray],
    semantic_ids: np.ndarray,
    max_sequence_length: int,
    codebook_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    import torch

    semantic_dim = semantic_ids.shape[1]
    max_tokens = max_sequence_length * semantic_dim
    input_ids = np.zeros((len(histories), max_tokens), dtype=np.int64)
    attention = np.zeros_like(input_ids)
    for row, history in enumerate(histories):
        history = np.asarray(history, dtype=np.int64)[-max_sequence_length:]
        codes = np.asarray(semantic_ids[history])
        tokens = offset_codes(codes, codebook_size).reshape(-1)
        input_ids[row, -len(tokens) :] = tokens
        attention[row, -len(tokens) :] = 1
    return torch.from_numpy(input_ids), torch.from_numpy(attention)


def model_config(metadata: dict, **overrides) -> dict:
    config = {
        "num_layers": 4,
        "num_decoder_layers": 4,
        "d_model": 128,
        "d_ff": 1024,
        "num_heads": 6,
        "d_kv": 64,
        "dropout_rate": 0.1,
        "vocab_size": int(metadata["semantic_codebook_size"])
        * int(metadata["semantic_id_dimensions"])
        + 1,
        "pad_token_id": 0,
        "eos_token_id": 0,
        "feed_forward_proj": "relu",
        "sem_id_dim": int(metadata["semantic_id_dimensions"]),
    }
    config.update(overrides)
    return config


def append_jsonl(path: Path, value: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
