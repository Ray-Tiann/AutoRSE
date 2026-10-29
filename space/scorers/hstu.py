#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def score(workspace: Path, private_dir: Path) -> dict:
    private = private_dir / "beauty"
    expected_users = np.load(private / "expected_user_ids.npy")
    targets = np.load(private / "test_targets.npy")
    catalog = np.load(private / "catalog.npy")
    submission_path = workspace / "datasets" / "beauty" / "working" / "submission.npz"
    with np.load(submission_path, allow_pickle=False) as submission:
        if set(submission.files) != {"user_ids", "top_items"}:
            raise ValueError("submission.npz must contain user_ids and top_items")
        user_ids = submission["user_ids"]
        top_items = submission["top_items"]
    if not np.array_equal(user_ids, expected_users):
        raise ValueError("User IDs or row order do not match the hidden test queries")
    if top_items.shape != (len(expected_users), 10):
        raise ValueError(f"top_items must have shape ({len(expected_users)}, 10)")
    if not np.issubdtype(top_items.dtype, np.integer):
        raise ValueError("top_items must use an integer dtype")
    if np.any(top_items < int(catalog[0])) or np.any(top_items > int(catalog[-1])):
        raise ValueError("Submission contains an item outside the catalog")
    if np.any(np.diff(np.sort(top_items, axis=1), axis=1) == 0):
        raise ValueError("Each row must contain ten distinct items")
    matches = top_items == targets[:, None]
    found = matches.any(axis=1)
    ranks = np.argmax(matches, axis=1)
    result = {"users": len(expected_users), "items": len(catalog)}
    for cutoff in (5, 10):
        hits = found & (ranks < cutoff)
        result[f"recall_at_{cutoff}"] = float(np.mean(hits))
        result[f"ndcg_at_{cutoff}"] = float(
            np.mean(np.where(hits, 1.0 / np.log2(ranks + 2), 0.0))
        )
    result["mrr_at_10"] = float(np.mean(np.where(found, 1.0 / (ranks + 1), 0.0)))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path("/workspace"))
    parser.add_argument("--private-dir", type=Path, default=Path("/tests/private"))
    parser.add_argument("--reward-path", type=Path, default=Path("/logs/verifier/reward.json"))
    parser.add_argument("--report-path", type=Path, default=Path("/logs/verifier/metrics.json"))
    args = parser.parse_args()
    try:
        metrics = score(args.workspace, args.private_dir)
        reward = {"reward": metrics["ndcg_at_10"], **metrics}
        report = {
            "protocol": "Beauty chronological leave-two-out, full-catalog Top-10",
            "metrics": metrics,
        }
    except Exception as error:
        reward = {"reward": 0.0, "ndcg_at_10": 0.0, "recall_at_10": 0.0}
        report = {"error": f"{type(error).__name__}: {error}"}
    write_json(args.reward_path, reward)
    write_json(args.report_path, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
