#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from common import dataset_dir, input_dir, working_dir
from evaluate import load_model, recommend


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=working_dir() / "hstu.pt")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    user_ids = np.load(input_dir() / "test" / "user_ids.npy")
    model = load_model(args.checkpoint, device)
    top_items = recommend(model, user_ids, True, args.batch_size, device)
    output = dataset_dir() / "working" / "submission.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, user_ids=user_ids, top_items=top_items)
    print(f"SUBMISSION={output}")


if __name__ == "__main__":
    main()
