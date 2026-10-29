#!/usr/bin/env python3
"""Generate predictions on the unlabeled test split from a trained checkpoint.

Output: CSV with columns sample_id,prediction. sample_id is the row index in
the test split (0-based), so it is unique and order-stable by construction.
The prediction column is the sigmoid of the click logit (a probability).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_kuaisar import amp_settings, build_model
from utils.kuaisar_data import KuaiSARHistoryDataset


MODEL_KEYS = (
    "seq_len",
    "ns_len",
    "d_model",
    "num_heads",
    "ffn_hidden",
    "num_layers",
    "pyramid_align",
    "mask_type",
    "activation_checkpoint",
    "optimizer",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict with a OneTrans checkpoint")
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=("test", "val", "train"))
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp-dtype", choices=("fp16", "bf16", "none"), default="bf16")
    parser.add_argument("--max-rows", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state = torch.load(args.checkpoint, map_location=args.device)
    config = state.get("config") or {}

    model_kwargs = {}
    for key in MODEL_KEYS:
        if key in config:
            value = config[key]
            if key in ("seq_len", "ns_len", "d_model", "num_heads", "ffn_hidden",
                       "num_layers", "pyramid_align"):
                value = int(value)
            model_kwargs[key] = value
    if "mask_type" not in model_kwargs:
        model_kwargs["mask_type"] = "paper_causal"
    if "activation_checkpoint" not in model_kwargs:
        model_kwargs["activation_checkpoint"] = False

    model_args = SimpleNamespace(
        model="onetrans",
        **model_kwargs,
    )
    model = build_model(model_args)
    model.load_state_dict(state["model"])
    model.to(args.device)
    model.eval()
    amp_enabled, amp_dtype = amp_settings(args)

    dataset = KuaiSARHistoryDataset(
        processed_dir=args.processed_dir,
        split=args.split,
        seq_len=int(model_kwargs.get("seq_len", 128)),
        max_rows=args.max_rows,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )

    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            non_seq = batch["non_seq"].to(args.device)
            sequence = batch["sequence"].to(args.device)
            seq_mask = batch["seq_mask"].to(args.device)
            with torch.autocast(
                device_type=torch.device(args.device).type,
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                output = model(non_seq, sequence, seq_mask)
            predictions.append(torch.sigmoid(output["click"]).float().cpu().numpy())

    if predictions:
        probs = np.concatenate(predictions)
    else:
        probs = np.zeros(0, dtype=np.float32)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["sample_id", "prediction"])
        for idx, prob in enumerate(probs):
            writer.writerow([idx, f"{float(prob):.9f}"])
    print(f"[predict] wrote {len(probs)} rows to {args.output}", flush=True)


if __name__ == "__main__":
    main()
