"""
Standalone evaluation script.

Usage:
    python evaluate.py --checkpoint checkpoints/rankmixer_100m/best.pt \
                       --data_dir data/processed \
                       --split val
"""

import argparse
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.rankmixer import RankMixer
from data.dataset import build_dataloader, load_meta


@torch.no_grad()
def evaluate(checkpoint_path: str, data_dir: str, split: str = "val", batch_size: int = 4096):
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg  = ckpt["cfg"]

    meta = load_meta(data_dir)

    # Build model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RankMixer(
        num_numerical   = meta["num_numerical"],
        cat_vocab_sizes = meta["vocab_sizes"],
        embed_dim       = cfg["embed_dim"],
        num_tokens      = cfg["num_tokens"],
        hidden_dim      = cfg["hidden_dim"],
        num_layers      = cfg["num_layers"],
        ffn_ratio       = cfg["ffn_ratio"],
        num_tasks       = 1,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    loader = build_dataloader(
        data_dir    = data_dir,
        split       = split,
        batch_size  = batch_size,
        num_workers = 4,
        distributed = False,
    )

    all_preds  = []
    all_labels = []

    for x_num, x_cat, labels in tqdm(loader, desc=f"Evaluating {split}"):
        x_num = x_num.to(device)
        x_cat = x_cat.to(device)

        logits = model(x_num, x_cat).squeeze(-1)
        preds  = torch.sigmoid(logits).cpu().numpy()

        all_preds.append(preds)
        all_labels.append(labels.numpy())

    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels).astype(int)

    auc = roc_auc_score(all_labels, all_preds)
    print(f"\n[Evaluate] {split} AUC = {auc:.6f}")
    return auc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir",   type=str, default="./data/processed")
    parser.add_argument("--split",      type=str, default="val")
    parser.add_argument("--batch_size", type=int, default=4096)
    args = parser.parse_args()

    evaluate(args.checkpoint, args.data_dir, args.split, args.batch_size)
