#!/usr/bin/env python3
"""Generate a Harbor submission from a RankMixer checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.dataset import build_dataloader, load_meta
from model.rankmixer import RankMixer


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    cfg = checkpoint["cfg"]
    meta = load_meta(args.data_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RankMixer(
        num_numerical=meta["num_numerical"],
        cat_vocab_sizes=meta["vocab_sizes"],
        embed_dim=cfg["embed_dim"],
        num_tokens=cfg["num_tokens"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        ffn_ratio=cfg["ffn_ratio"],
        num_tasks=1,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    loader = build_dataloader(
        args.data_dir,
        "test",
        batch_size=args.batch_size,
        num_workers=4,
        distributed=False,
    )
    all_ids: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    for x_num, x_cat, sample_ids in tqdm(loader, desc="Predict test"):
        logits = model(x_num.to(device), x_cat.to(device)).squeeze(-1)
        all_scores.append(torch.sigmoid(logits).cpu().numpy())
        all_ids.append(sample_ids.numpy())

    ids = np.concatenate(all_ids).astype(np.int64, copy=False)
    scores = np.concatenate(all_scores).astype(np.float32, copy=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, sample_ids=ids, scores=scores)
    print(f"SUBMISSION={args.output} rows={len(ids):,}")


if __name__ == "__main__":
    main()
