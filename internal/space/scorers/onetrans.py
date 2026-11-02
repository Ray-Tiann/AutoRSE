#!/usr/bin/env python3
"""Score OneTrans click probabilities against hidden KuaiSAR v2 labels."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


REFERENCE_AUC = 0.6767


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(np.int8, copy=False)
    positives = int(labels.sum())
    negatives = int(labels.size - positives)
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(labels.size, dtype=np.float64)
    ends = np.r_[np.flatnonzero(np.diff(sorted_scores) != 0), labels.size - 1]
    starts = np.r_[0, ends[:-1] + 1]
    sizes = ends - starts + 1
    ranks[order] = np.repeat((starts + ends) / 2.0 + 1.0, sizes)
    rank_sum = float(ranks[labels == 1].sum())
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def weighted_user_auc(labels: np.ndarray, scores: np.ndarray, users: np.ndarray) -> float:
    order = np.argsort(users, kind="mergesort")
    labels, scores, users = labels[order], scores[order], users[order]
    weighted_sum = 0.0
    total_weight = 0
    start = 0
    while start < users.size:
        end = start + 1
        while end < users.size and users[end] == users[start]:
            end += 1
        auc = binary_auc(labels[start:end], scores[start:end])
        if not math.isnan(auc):
            weight = end - start
            weighted_sum += auc * weight
            total_weight += weight
        start = end
    return weighted_sum / total_weight if total_weight else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path("/workspace"))
    parser.add_argument("--private-dir", type=Path, default=Path("/tests/private"))
    parser.add_argument("--reward-path", type=Path, default=Path("/logs/verifier/reward.json"))
    parser.add_argument("--report-path", type=Path, default=Path("/logs/verifier/metrics.json"))
    args = parser.parse_args()

    try:
        submission_path = args.workspace / "datasets/kuaisar_v2/working/submission.csv"
        private_path = args.private_dir / "kuaisar_v2/labels.npz"
        with np.load(private_path, allow_pickle=False) as data:
            expected_ids = data["sample_ids"]
            labels = data["click"].astype(np.int8)
            users = data["user_id"].astype(np.int64)

        sample_ids: list[int] = []
        predictions: list[float] = []
        with submission_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != ["sample_id", "prediction"]:
                raise ValueError("submission header must be sample_id,prediction")
            for row in reader:
                sample_ids.append(int(row["sample_id"]))
                predictions.append(float(row["prediction"]))

        ids = np.asarray(sample_ids, dtype=np.int64)
        scores = np.asarray(predictions, dtype=np.float64)
        if not np.array_equal(ids, expected_ids):
            raise ValueError("sample IDs or row order differ from public test")
        if scores.shape != labels.shape:
            raise ValueError("prediction row count differs from hidden labels")
        if not np.isfinite(scores).all() or np.any(scores < 0.0) or np.any(scores > 1.0):
            raise ValueError("predictions must be finite probabilities in [0,1]")

        auc = float(binary_auc(labels, scores))
        uauc = float(weighted_user_auc(labels, scores, users))
        clipped = np.clip(scores, 1e-15, 1.0 - 1e-15)
        logloss = float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped)))
        report = {
            "dataset": "KuaiSAR v2",
            "test_samples": int(labels.size),
            "auc": auc,
            "uauc": uauc,
            "logloss": logloss,
            "reference_auc": REFERENCE_AUC,
            "absolute_improvement": auc - REFERENCE_AUC,
        }
        reward = {"reward": auc, "auc": auc, "uauc": uauc, "logloss": logloss}
    except Exception as error:
        report = {"error": f"{type(error).__name__}: {error}"}
        reward = {"reward": 0.0, "auc": 0.0, "uauc": 0.0, "logloss": 0.0}

    write_json(args.reward_path, reward)
    write_json(args.report_path, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
