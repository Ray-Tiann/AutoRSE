#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from functools import partial
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from common import (
    UPSTREAM,
    append_jsonl,
    load_metadata,
    load_semantic_ids,
    load_sequences,
    make_inputs,
    model_config,
    offset_codes,
    working_dir,
)
from evaluate import evaluate_model

sys.path.insert(0, str(UPSTREAM))
from genrec.models.tiger import Tiger


class SlidingWindowDataset(Dataset):
    def __init__(self, max_users: int | None = None) -> None:
        self.store = load_sequences()
        users = len(self.store.user_ids) if max_users is None else min(max_users, len(self.store.user_ids))
        self.references = []
        for row in range(users):
            length = len(self.store.sequence(row))
            self.references.extend((row, target_position) for target_position in range(1, length))

    def __len__(self) -> int:
        return len(self.references)

    def __getitem__(self, index: int) -> tuple[np.ndarray, int]:
        row, target_position = self.references[index]
        sequence = self.store.sequence(row)
        return np.asarray(sequence[:target_position], dtype=np.int64), int(sequence[target_position])


def collate(examples, semantic_ids: np.ndarray, max_length: int, codebook_size: int):
    histories = [history for history, _ in examples]
    input_ids, attention = make_inputs(histories, semantic_ids, max_length, codebook_size)
    targets = np.asarray([target for _, target in examples], dtype=np.int64)
    labels = offset_codes(np.asarray(semantic_ids[targets]), codebook_size)
    return input_ids, attention, torch.from_numpy(labels)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--infer-batch-size", type=int, default=96)
    parser.add_argument("--beam-size", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=int(os.environ.get("RESEARCH_SEED", "2024")))
    parser.add_argument("--max-users", type=int)
    parser.add_argument("--checkpoint", type=Path, default=working_dir() / "tiger.pt")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.infer_batch_size, args.beam_size) < 1:
        parser.error("epochs, batch sizes and beam size must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metadata = load_metadata()
    semantic_ids = load_semantic_ids()
    config = model_config(metadata)
    model = Tiger(config).to(device)
    dataset = SlidingWindowDataset(args.max_users)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=partial(collate, semantic_ids=semantic_ids,
                           max_length=int(metadata["maximum_sequence_length"]),
                           codebook_size=int(metadata["semantic_codebook_size"])),
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    log_path = working_dir() / "experiment_log.jsonl"
    if len(loader) == 0:
        raise ValueError("No full training batch; reduce --batch-size")
    best = -1.0
    stale = 0
    for epoch in range(args.epochs):
        model.train()
        losses = []
        started = time.time()
        for input_ids, attention, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss, _ = model(
                input_ids=input_ids.to(device, non_blocking=True),
                attention_mask=attention.to(device, non_blocking=True),
                labels=labels.to(device, non_blocking=True),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        result = evaluate_model(
            model,
            config,
            args.infer_batch_size,
            args.beam_size,
            device,
            args.max_users,
        )
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "seconds": time.time() - started,
            **result,
        }
        print(record, flush=True)
        append_jsonl(log_path, record)
        score = result["ndcg_at_10"]
        if score > best:
            best = score
            stale = 0
            torch.save(
                {"model_state": model.state_dict(), "model_config": config, "epoch": epoch},
                args.checkpoint,
            )
        else:
            stale += 1
        if stale >= args.patience:
            break
    print(f"CHECKPOINT={args.checkpoint}")


if __name__ == "__main__":
    main()
