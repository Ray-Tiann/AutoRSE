from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[1]


def dataset_dir() -> Path:
    return WORKSPACE / "datasets" / "beauty"


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
    timestamps: np.ndarray

    def rows_for(self, requested: np.ndarray) -> np.ndarray:
        rows = np.searchsorted(self.user_ids, requested)
        if np.any(rows >= len(self.user_ids)) or not np.array_equal(
            self.user_ids[rows], requested
        ):
            raise ValueError("Requested user IDs are not present in the sequence store")
        return rows

    def sequence(self, row: int) -> np.ndarray:
        return self.items[self.offsets[row] : self.offsets[row + 1]]

    def sequence_timestamps(self, row: int) -> np.ndarray:
        return self.timestamps[self.offsets[row] : self.offsets[row + 1]]


def load_train_sequences() -> SequenceStore:
    root = input_dir() / "train_sequences"
    return SequenceStore(
        user_ids=np.load(str(root / "user_ids.npy"), mmap_mode="r"),
        offsets=np.load(str(root / "offsets.npy"), mmap_mode="r"),
        items=np.load(str(root / "items.npy"), mmap_mode="r"),
        timestamps=np.load(str(root / "timestamps.npy"), mmap_mode="r"),
    )


def load_metadata() -> dict:
    return json.loads((input_dir() / "metadata.json").read_text(encoding="utf-8"))


def padded_batch(
    sequences: list[np.ndarray],
    timestamps: list[np.ndarray],
    max_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    batch = len(sequences)
    items = np.zeros((batch, max_length), dtype=np.int64)
    times = np.zeros((batch, max_length), dtype=np.int64)
    lengths = np.zeros(batch, dtype=np.int64)
    for row, (sequence, sequence_times) in enumerate(zip(sequences, timestamps)):
        sequence = np.asarray(sequence)[-max_length:]
        sequence_times = np.asarray(sequence_times)[-max_length:]
        length = len(sequence)
        if length == 0:
            raise ValueError("Every query history must contain at least one item")
        items[row, :length] = sequence
        times[row, :length] = sequence_times
        lengths[row] = length
    return items, times, lengths


def append_jsonl(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
