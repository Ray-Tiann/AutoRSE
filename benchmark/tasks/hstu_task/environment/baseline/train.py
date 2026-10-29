#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from common import append_jsonl, input_dir, load_metadata, load_train_sequences, working_dir
from evaluate import evaluate_checkpoint
from model import HSTUModel


class AutoregressiveDataset(Dataset):
    def __init__(self, include_validation: bool) -> None:
        self.store = load_train_sequences()
        self.include_validation = include_validation
        self.validation_items = np.load(input_dir() / "validation" / "target_items.npy")
        self.validation_times = np.load(input_dir() / "validation" / "target_timestamps.npy")

    def __len__(self) -> int:
        return len(self.store.user_ids)

    def __getitem__(self, row: int) -> tuple[np.ndarray, np.ndarray]:
        items = np.asarray(self.store.sequence(row), dtype=np.int64)
        times = np.asarray(self.store.sequence_timestamps(row), dtype=np.int64)
        if self.include_validation:
            items = np.append(items, self.validation_items[row])
            times = np.append(times, self.validation_times[row])
        return items, times


def collate(examples, max_length: int):
    inputs = np.zeros((len(examples), max_length), dtype=np.int64)
    times = np.zeros((len(examples), max_length), dtype=np.int64)
    targets = np.zeros((len(examples), max_length), dtype=np.int64)
    for row, (sequence, timestamps) in enumerate(examples):
        sequence = sequence[-(max_length + 1) :]
        timestamps = timestamps[-(max_length + 1) :]
        source = sequence[:-1]
        length = len(source)
        inputs[row, :length] = source
        times[row, :length] = timestamps[:-1]
        targets[row, :length] = sequence[1:]
    return (
        torch.from_numpy(inputs),
        torch.from_numpy(times),
        torch.from_numpy(targets),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--attention-dim", type=int, default=16)
    parser.add_argument("--linear-dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=int(os.environ.get("RESEARCH_SEED", "2024")))
    parser.add_argument("--include-validation", action="store_true")
    parser.add_argument("--max-users", type=int)
    parser.add_argument("--checkpoint", type=Path, default=working_dir() / "hstu.pt")
    args = parser.parse_args()
    if args.include_validation:
        raise ValueError("Validation supervision is forbidden in this study")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metadata = load_metadata()
    model = HSTUModel(
        num_items=int(metadata["items"]),
        max_length=int(metadata["maximum_sequence_length"]),
        embedding_dim=args.embedding_dim,
        num_blocks=args.blocks,
        num_heads=args.heads,
        attention_dim=args.attention_dim,
        linear_dim=args.linear_dim,
        dropout=args.dropout,
        temperature=args.temperature,
    ).to(device)
    dataset = AutoregressiveDataset(args.include_validation)
    if args.max_users is not None:
        dataset = torch.utils.data.Subset(dataset, range(min(args.max_users, len(dataset))))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=lambda rows: collate(rows, model.max_length),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=args.weight_decay
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    log_path = working_dir() / "experiment_log.jsonl"
    best = -1.0
    stale = 0
    for epoch in range(args.epochs):
        model.train()
        losses = []
        started = time.time()
        for item_ids, timestamps, targets in loader:
            item_ids = item_ids.to(device, non_blocking=True)
            timestamps = timestamps.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            valid = targets.ne(0)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                hidden = model.encode(item_ids, timestamps)[valid]
                logits = model.score_hidden(hidden)
                loss = torch.nn.functional.cross_entropy(logits, targets[valid] - 1)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))

        state = {
            "model_state": model.state_dict(),
            "model_config": model.config(),
            "epoch": epoch,
        }
        candidate = working_dir() / "hstu_candidate.pt"
        torch.save(state, candidate)
        if args.include_validation:
            score = -float(np.mean(losses))
            result = {"train_loss": float(np.mean(losses))}
        else:
            result = evaluate_checkpoint(candidate, max_users=args.max_users)
            score = result["ndcg_at_10"]
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "seconds": time.time() - started,
            **result,
        }
        print(record, flush=True)
        append_jsonl(log_path, record)
        if score > best:
            best = score
            stale = 0
            torch.save(state, args.checkpoint)
        else:
            stale += 1
        if stale >= args.patience:
            break
    print(f"CHECKPOINT={args.checkpoint}")


if __name__ == "__main__":
    main()
