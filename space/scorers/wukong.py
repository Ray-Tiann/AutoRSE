#!/usr/bin/env python3
"""Validate WuKong click probabilities on hidden MovieLens Latest labels."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np


REFERENCE_AUC = 0.9645398857526273


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def load_labels(path: Path) -> tuple[list[str], np.ndarray]:
    sample_ids: list[str] = []
    labels: list[int] = []
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["sample_id", "label"]:
            raise ValueError("private MovieLens Latest label schema is invalid")
        for row in reader:
            sample_ids.append(row["sample_id"])
            labels.append(int(row["label"]))
    return sample_ids, np.asarray(labels, dtype=np.int8)


def load_predictions(path: Path) -> tuple[list[str], np.ndarray]:
    sample_ids: list[str] = []
    scores: list[float] = []
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["sample_id", "score"]:
            raise ValueError("submission columns must be sample_id,score")
        for row in reader:
            sample_ids.append(row["sample_id"])
            scores.append(float(row["score"]))
    return sample_ids, np.asarray(scores, dtype=np.float64)


def auc_score(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("AUC requires both positive and negative labels")

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
    return (rank_sum - positives * (positives + 1) / 2.0) / (
        positives * negatives
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path("/workspace"))
    parser.add_argument("--private-dir", type=Path, default=Path("/tests/private"))
    parser.add_argument("--reward-path", type=Path, default=Path("/logs/verifier/reward.json"))
    parser.add_argument("--report-path", type=Path, default=Path("/logs/verifier/metrics.json"))
    args = parser.parse_args()

    try:
        submission = (
            args.workspace / "datasets/movielenslatest/working/submission.csv.gz"
        )
        expected_ids, labels = load_labels(
            args.private_dir / "movielenslatest/test_data_labeled.csv"
        )
        sample_ids, scores = load_predictions(submission)
        if sample_ids != expected_ids:
            raise ValueError("sample IDs or row order differ from the public test input")
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError("sample IDs must be unique")
        if not np.isfinite(scores).all():
            raise ValueError("scores contain NaN or infinity")
        if np.any(scores < 0.0) or np.any(scores > 1.0):
            raise ValueError("scores must be probabilities in [0, 1]")

        auc = float(auc_score(labels, scores))
        clipped = np.clip(scores, 1e-15, 1.0 - 1e-15)
        logloss = float(
            -np.mean(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped))
        )
        reward = {"reward": auc, "auc": auc, "logloss": logloss}
        report = {
            "dataset": "MovielensLatest_x1",
            "test_samples": int(len(labels)),
            "auc": auc,
            "logloss": logloss,
            "reference_auc": REFERENCE_AUC,
            "absolute_improvement": auc - REFERENCE_AUC,
        }
    except Exception as error:
        reward = {"reward": 0.0, "auc": 0.0, "logloss": 0.0}
        report = {"error": f"{type(error).__name__}: {error}"}

    write_json(args.reward_path, reward)
    write_json(args.report_path, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
