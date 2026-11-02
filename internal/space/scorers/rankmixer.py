#!/usr/bin/env python3
"""Score RankMixer probabilities against hidden Criteo-5M labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def auc_score(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("AUC requires both classes")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    rank_sum = float(ranks[labels == 1].sum())
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path("/workspace"))
    parser.add_argument("--private-dir", type=Path, default=Path("/tests/private"))
    parser.add_argument("--reward-path", type=Path, default=Path("/logs/verifier/reward.json"))
    parser.add_argument("--report-path", type=Path, default=Path("/logs/verifier/metrics.json"))
    args = parser.parse_args()

    try:
        submission_path = args.workspace / "datasets/criteo5m/working/submission.npz"
        private = args.private_dir / "criteo5m"
        expected_ids = np.load(private / "expected_sample_ids.npy", allow_pickle=False)
        labels = np.load(private / "test_labels.npy", allow_pickle=False).astype(np.int8)
        with np.load(submission_path, allow_pickle=False) as submission:
            if set(submission.files) != {"sample_ids", "scores"}:
                raise ValueError("submission must contain exactly sample_ids and scores")
            sample_ids = submission["sample_ids"]
            scores = submission["scores"].astype(np.float64)
        if sample_ids.dtype != np.int64 or sample_ids.shape != expected_ids.shape:
            raise ValueError("sample_ids must be int64 with the expected shape")
        if not np.array_equal(sample_ids, expected_ids):
            raise ValueError("sample IDs or row order differ from public test")
        if scores.shape != labels.shape:
            raise ValueError("score shape differs from hidden labels")
        if not np.isfinite(scores).all() or np.any(scores < 0.0) or np.any(scores > 1.0):
            raise ValueError("scores must be finite probabilities in [0,1]")

        auc = float(auc_score(labels, scores))
        clipped = np.clip(scores, 1e-15, 1.0 - 1e-15)
        logloss = float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped)))
        reference_path = private / "reference_metrics.json"
        reference = json.loads(reference_path.read_text(encoding="utf-8")) if reference_path.exists() else {}
        reference_auc = reference.get("test_auc")
        report = {
            "dataset": "Criteo-5M",
            "test_samples": int(len(labels)),
            "auc": auc,
            "logloss": logloss,
            "reference_auc": reference_auc,
            "absolute_improvement": None if reference_auc is None else auc - float(reference_auc),
        }
        reward = {"reward": auc, "auc": auc, "logloss": logloss}
    except Exception as error:
        report = {"error": f"{type(error).__name__}: {error}"}
        reward = {"reward": 0.0, "auc": 0.0, "logloss": 0.0}

    write_json(args.reward_path, reward)
    write_json(args.report_path, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
