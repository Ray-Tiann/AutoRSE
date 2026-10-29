#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from common import dataset_dir, input_dir, load_sequences, working_dir
from evaluate import load_model, recommend


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=working_dir() / "tiger.pt")
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--beam-size", type=int, default=30)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    user_ids = np.load(input_dir() / "test" / "user_ids.npy")
    validation_users = np.load(input_dir() / "validation" / "user_ids.npy")
    validation_items = np.load(input_dir() / "validation" / "target_items.npy")
    validation = dict(zip(map(int, validation_users), map(int, validation_items)))
    store = load_sequences()
    rows = store.rows_for(user_ids)
    histories = [
        np.append(np.asarray(store.sequence(int(row)), dtype=np.int64), validation[int(user_ids[index])])
        for index, row in enumerate(rows)
    ]
    model, config = load_model(args.checkpoint, device)
    top_items = recommend(
        model, config, histories, args.batch_size, args.beam_size, device
    )
    output = dataset_dir() / "working" / "submission.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, user_ids=user_ids, top_items=top_items)
    print(f"SUBMISSION={output}")


if __name__ == "__main__":
    main()
