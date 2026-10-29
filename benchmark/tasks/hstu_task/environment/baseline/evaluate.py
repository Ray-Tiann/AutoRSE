#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from common import input_dir, load_train_sequences, padded_batch, working_dir
from model import HSTUModel


def load_model(checkpoint: Path, device: torch.device) -> HSTUModel:
    state = torch.load(checkpoint, map_location="cpu")
    model = HSTUModel(**state["model_config"])
    model.load_state_dict(state["model_state"])
    return model.to(device).eval()


@torch.no_grad()
def recommend(
    model: HSTUModel,
    user_ids: np.ndarray,
    include_validation: bool,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    store = load_train_sequences()
    rows = store.rows_for(user_ids)
    validation_items = np.load(input_dir() / "validation" / "target_items.npy")
    validation_times = np.load(input_dir() / "validation" / "target_timestamps.npy")
    output = np.empty((len(user_ids), 10), dtype=np.int32)
    for start in range(0, len(user_ids), batch_size):
        stop = min(start + batch_size, len(user_ids))
        sequences = []
        timestamps = []
        for output_row, source_row in enumerate(rows[start:stop], start=start):
            sequence = np.asarray(store.sequence(int(source_row)), dtype=np.int64)
            times = np.asarray(store.sequence_timestamps(int(source_row)), dtype=np.int64)
            if include_validation:
                sequence = np.append(sequence, validation_items[output_row])
                times = np.append(times, validation_times[output_row])
            sequences.append(sequence)
            timestamps.append(times)
        items_np, times_np, lengths_np = padded_batch(
            sequences, timestamps, model.max_length
        )
        items = torch.from_numpy(items_np).to(device)
        times = torch.from_numpy(times_np).to(device)
        lengths = torch.from_numpy(lengths_np).to(device)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            query = model.query(items, times, lengths)
            scores = model.score_hidden(query)
        for row, sequence in enumerate(sequences):
            seen = torch.as_tensor(np.unique(sequence), device=device, dtype=torch.long)
            scores[row, seen - 1] = -torch.inf
        output[start:stop] = torch.topk(scores, k=10, dim=-1).indices.cpu().numpy() + 1
    return output


def metrics(top_items: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    matches = top_items == targets[:, None]
    found = matches.any(axis=1)
    ranks = np.argmax(matches, axis=1)
    result = {}
    for cutoff in (5, 10):
        hits = found & (ranks < cutoff)
        result[f"recall_at_{cutoff}"] = float(np.mean(hits))
        result[f"ndcg_at_{cutoff}"] = float(
            np.mean(np.where(hits, 1.0 / np.log2(ranks + 2), 0.0))
        )
    result["mrr_at_10"] = float(
        np.mean(np.where(found, 1.0 / (ranks + 1), 0.0))
    )
    return result


def evaluate_checkpoint(
    checkpoint: Path, batch_size: int = 256, max_users: int | None = None
) -> dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    user_ids = np.load(input_dir() / "validation" / "user_ids.npy")
    targets = np.load(input_dir() / "validation" / "target_items.npy")
    if max_users is not None:
        user_ids = user_ids[:max_users]
        targets = targets[:max_users]
    model = load_model(checkpoint, device)
    top_items = recommend(model, user_ids, False, batch_size, device)
    return metrics(top_items, targets)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=working_dir() / "hstu.pt")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-users", type=int)
    args = parser.parse_args()
    result = evaluate_checkpoint(args.checkpoint, args.batch_size, args.max_users)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
