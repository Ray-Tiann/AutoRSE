from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset


EVENT_FIELDS = (
    "item_id",
    "author_id",
    "category_id",
    "item_type_hash",
    "source",
    "click",
    "like",
    "follow",
    "forward",
    "search_action",
    "search_source_hash",
)

SAMPLE_FIELDS = (
    "user_id",
    "item_id",
    "author_id",
    "category_id",
    "item_type_hash",
    "upload_type_hash",
    "duration_bucket",
    "hour",
    "weekday",
    "search_active_level",
    "rec_active_level",
    "onehot_feat1",
    "onehot_feat2",
)


_EVENT_CACHE: dict[str, dict[str, np.ndarray]] = {}


class KuaiSARHistoryDataset(Dataset[dict[str, torch.Tensor]]):
    """Build fixed-length, strictly historical sequences from normalized tables.

    `history_end` always points to the first event at the current request
    timestamp, so every returned event occurred strictly before the label.
    """

    def __init__(
        self,
        processed_dir: Path,
        split: str,
        seq_len: int,
        max_rows: int | None = None,
    ) -> None:
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")
        self.seq_len = seq_len
        event_path = processed_dir / "events.parquet"
        sample_path = processed_dir / f"samples_{split}.parquet"
        if not event_path.exists() or not sample_path.exists():
            raise FileNotFoundError(
                f"Missing processed KuaiSAR files: {event_path} and/or {sample_path}"
            )

        event_columns = ["timestamp", *EVENT_FIELDS]
        sample_columns = ["history_start", "history_end", "timestamp", "click", "like", *SAMPLE_FIELDS]
        # Train/validation/test datasets all use the same immutable event
        # table. Re-reading and materializing it for every split consumed more
        # than 30 GiB across two DDP ranks on the Harbor host.
        event_cache_key = str(event_path.resolve())
        if event_cache_key not in _EVENT_CACHE:
            _EVENT_CACHE[event_cache_key] = {
                name: values
                for name, values in zip(
                    event_columns,
                    pl.read_parquet(event_path, columns=event_columns).to_numpy().T,
                    strict=True,
                )
            }
        self.events = _EVENT_CACHE[event_cache_key]
        sample_frame = pl.read_parquet(sample_path, columns=sample_columns)
        if max_rows is not None:
            sample_frame = sample_frame.head(max_rows)
        self.samples = {
            name: values
            for name, values in zip(sample_columns, sample_frame.to_numpy().T, strict=True)
        }

    def __len__(self) -> int:
        return len(self.samples["click"])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        history_start = int(self.samples["history_start"][index])
        history_end = int(self.samples["history_end"][index])
        start = max(history_start, history_end - self.seq_len)
        history_length = max(0, history_end - start)
        pad = self.seq_len - history_length

        sequence = np.zeros((self.seq_len, len(EVENT_FIELDS) + 1), dtype=np.int64)
        if history_length:
            event_slice = slice(start, history_end)
            for column_idx, name in enumerate(EVENT_FIELDS):
                sequence[pad:, column_idx] = self.events[name][event_slice]
            current_timestamp = int(self.samples["timestamp"][index])
            event_timestamps = self.events["timestamp"][event_slice].astype(np.int64, copy=False)
            gap_ms = np.maximum(current_timestamp - event_timestamps, 0)
            gap_minutes = gap_ms // 60_000
            gap_buckets = np.floor(np.log2(gap_minutes + 1)).astype(np.int64)
            sequence[pad:, -1] = np.minimum(gap_buckets, 127)

        seq_mask = np.zeros(self.seq_len, dtype=np.bool_)
        seq_mask[pad:] = True
        non_seq = np.asarray(
            [int(self.samples[name][index]) for name in SAMPLE_FIELDS],
            dtype=np.int64,
        )
        labels = np.asarray(
            [int(self.samples["click"][index]), int(self.samples["like"][index])],
            dtype=np.float32,
        )
        return {
            "non_seq": torch.from_numpy(non_seq),
            "sequence": torch.from_numpy(sequence),
            "seq_mask": torch.from_numpy(seq_mask),
            "labels": torch.from_numpy(labels),
            "user_id": torch.tensor(int(self.samples["user_id"][index]), dtype=torch.long),
            "timestamp": torch.tensor(int(self.samples["timestamp"][index]), dtype=torch.long),
        }
