"""Memory-mapped Criteo-5M datasets for the RankMixer Harbor task."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler


class CriteoDataset(Dataset):
    def __init__(self, data_dir, split: str = "train", mmap: bool = True):
        split_dir = Path(data_dir) / split
        mode = "r" if mmap else None
        self.numericals = np.load(split_dir / "numericals.npy", mmap_mode=mode)
        self.categoricals = np.load(split_dir / "categoricals.npy", mmap_mode=mode)
        label_path = split_dir / "labels.npy"
        id_path = split_dir / "sample_ids.npy"
        self.labels = np.load(label_path, mmap_mode=mode) if label_path.exists() else None
        self.sample_ids = np.load(id_path, mmap_mode=mode) if id_path.exists() else None
        self.n = len(self.numericals)
        if len(self.categoricals) != self.n:
            raise ValueError(f"{split}: numerical/categorical row mismatch")
        if self.labels is not None and len(self.labels) != self.n:
            raise ValueError(f"{split}: label row mismatch")
        if self.sample_ids is not None and len(self.sample_ids) != self.n:
            raise ValueError(f"{split}: sample ID row mismatch")
        print(f"[CriteoDataset] Loaded {split}: {self.n:,} samples (mmap={mmap})")

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        num = torch.tensor(self.numericals[idx], dtype=torch.float32)
        cat = torch.tensor(self.categoricals[idx], dtype=torch.long)
        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return num, cat, label
        if self.sample_ids is None:
            raise ValueError("unlabeled data must contain sample_ids.npy")
        return num, cat, torch.tensor(self.sample_ids[idx], dtype=torch.long)


def load_meta(data_dir) -> dict:
    return json.loads((Path(data_dir) / "meta.json").read_text(encoding="utf-8"))


def build_dataloader(
    data_dir,
    split: str,
    batch_size: int,
    num_workers: int = 4,
    mmap: bool = True,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    pin_memory: bool = True,
    seed: int = 2024,
) -> DataLoader:
    dataset = CriteoDataset(data_dir, split, mmap=mmap)
    sampler = None
    shuffle = split == "train"
    if distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=True,
        )
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=split == "train",
        persistent_workers=num_workers > 0,
    )
