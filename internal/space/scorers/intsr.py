#!/usr/bin/env python3
"""Validate and score the hidden IntSR Amazon candidate-ranking test."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


TASK_NAMES = {0: "rec", 1: "src"}
REFERENCE_MACRO_NDCG = 0.694032


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def expected_metrics(scores: np.ndarray, positive_mask: np.ndarray) -> dict[str, float]:
    if scores.shape != positive_mask.shape:
        raise ValueError("score and positive-mask shapes differ")
    if not np.isfinite(scores).all():
        raise ValueError("scores contain NaN or infinity")
    if not np.all(positive_mask.sum(axis=1) == 1):
        raise ValueError("each row must contain its hidden positive exactly once")

    positive_scores = scores[positive_mask]
    greater = (scores > positive_scores[:, None]).sum(axis=1)
    tied = (scores == positive_scores[:, None]).sum(axis=1)
    discounts = np.zeros(scores.shape[1] + 1, dtype=np.float64)
    discounts[1:] = 1.0 / np.log2(np.arange(1, scores.shape[1] + 1) + 1)
    cumulative = np.cumsum(discounts)
    values: dict[str, float] = {}
    for cutoff in (1, 5, 10):
        visible = np.clip(cutoff - greater, 0, tied)
        values[f"HR@{cutoff}"] = float(np.mean(visible / tied))
        lower = np.minimum(greater, cutoff)
        upper = np.minimum(greater + tied, cutoff)
        dcg = (cumulative[upper] - cumulative[lower]) / tied
        values[f"NDCG@{cutoff}"] = float(np.mean(dcg))
    return {
        key: values[key]
        for key in ("HR@1", "HR@5", "HR@10", "NDCG@5", "NDCG@10")
    }


def load_labels(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sample_ids = []
    task_ids = []
    positive_items = []
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["sample_id", "task", "positive_item_id"]:
            raise ValueError("private label schema is invalid")
        for row in reader:
            sample_ids.append(int(row["sample_id"]))
            try:
                task_ids.append({"rec": 0, "src": 1}[row["task"]])
            except KeyError as error:
                raise ValueError("private task value is invalid") from error
            positive_items.append(int(row["positive_item_id"]))
    return (
        np.asarray(sample_ids, dtype=np.int64),
        np.asarray(task_ids, dtype=np.uint8),
        np.asarray(positive_items, dtype=np.int64),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path("/workspace"))
    parser.add_argument("--private-dir", type=Path, default=Path("/tests/private"))
    parser.add_argument("--reward-path", type=Path, default=Path("/logs/verifier/reward.json"))
    parser.add_argument("--report-path", type=Path, default=Path("/logs/verifier/metrics.json"))
    args = parser.parse_args()

    try:
        submission_path = (
            args.workspace / "datasets/Amazon/working/submission.npz"
        )
        private = args.private_dir / "Amazon"
        with np.load(private / "expected_submission.npz", allow_pickle=False) as expected:
            expected_ids = expected["sample_ids"]
            expected_tasks = expected["task_ids"]
            expected_candidates = expected["candidate_items"]
        with np.load(submission_path, allow_pickle=False) as submission:
            required = {"sample_ids", "task_ids", "candidate_items", "scores"}
            missing = required - set(submission.files)
            if missing:
                raise ValueError(f"submission is missing arrays: {sorted(missing)}")
            sample_ids = np.asarray(submission["sample_ids"], dtype=np.int64)
            task_ids = np.asarray(submission["task_ids"], dtype=np.uint8)
            candidates = np.asarray(submission["candidate_items"], dtype=np.int32)
            scores = np.asarray(submission["scores"], dtype=np.float64)

        if not np.array_equal(sample_ids, expected_ids):
            raise ValueError("sample IDs or row order differ from the public test input")
        if not np.array_equal(task_ids, expected_tasks):
            raise ValueError("task IDs differ from the public test input")
        if not np.array_equal(candidates, expected_candidates):
            raise ValueError("candidate items or candidate order were modified")
        if scores.shape != expected_candidates.shape:
            raise ValueError("scores must have shape [133391, 100]")

        label_ids, label_tasks, positives = load_labels(
            private / "test_data_labeled.csv"
        )
        if not np.array_equal(label_ids, expected_ids) or not np.array_equal(
            label_tasks, expected_tasks
        ):
            raise ValueError("trusted label ordering is inconsistent")

        task_metrics: dict[str, dict[str, float]] = {}
        for task_id, name in TASK_NAMES.items():
            selected = task_ids == task_id
            positive_mask = candidates[selected] == positives[selected, None]
            task_metrics[name] = expected_metrics(scores[selected], positive_mask)

        macro_ndcg = float(
            np.mean([task_metrics[name]["NDCG@10"] for name in ("rec", "src")])
        )
        reward = {
            "reward": macro_ndcg,
            "macro_ndcg_at_10": macro_ndcg,
            "rec_ndcg_at_10": task_metrics["rec"]["NDCG@10"],
            "src_ndcg_at_10": task_metrics["src"]["NDCG@10"],
        }
        report = {
            "dataset": "Amazon Kindle",
            "protocol": "100 candidates: one hidden positive plus 99 negatives",
            "queries": int(len(sample_ids)),
            "macro_ndcg_at_10": macro_ndcg,
            "reference_macro_ndcg_at_10": REFERENCE_MACRO_NDCG,
            "absolute_improvement": macro_ndcg - REFERENCE_MACRO_NDCG,
            "tasks": task_metrics,
        }
    except Exception as error:
        reward = {
            "reward": 0.0,
            "macro_ndcg_at_10": 0.0,
            "rec_ndcg_at_10": 0.0,
            "src_ndcg_at_10": 0.0,
        }
        report = {"error": f"{type(error).__name__}: {error}"}

    write_json(args.reward_path, reward)
    write_json(args.report_path, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
