from __future__ import annotations

import math

import numpy as np


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(np.int8, copy=False)
    positives = int(labels.sum())
    negatives = int(labels.size - positives)
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(labels.size, dtype=np.float64)
    group_ends = np.r_[np.flatnonzero(np.diff(sorted_scores) != 0), labels.size - 1]
    group_starts = np.r_[0, group_ends[:-1] + 1]
    group_sizes = group_ends - group_starts + 1
    average_ranks = (group_starts + group_ends) / 2.0 + 1.0
    ranks[order] = np.repeat(average_ranks, group_sizes)
    positive_rank_sum = ranks[labels == 1].sum()
    return float((positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives))


def weighted_user_auc(labels: np.ndarray, scores: np.ndarray, users: np.ndarray) -> float:
    order = np.argsort(users, kind="mergesort")
    labels = labels[order]
    scores = scores[order]
    users = users[order]
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


def macro_group_auc(labels: np.ndarray, scores: np.ndarray, groups: np.ndarray) -> float:
    values = [binary_auc(labels[groups == group], scores[groups == group]) for group in np.unique(groups)]
    finite = [value for value in values if not math.isnan(value)]
    return float(np.mean(finite)) if finite else float("nan")


def macro_group_weighted_user_auc(
    labels: np.ndarray,
    scores: np.ndarray,
    users: np.ndarray,
    groups: np.ndarray,
) -> float:
    values = [
        weighted_user_auc(labels[groups == group], scores[groups == group], users[groups == group])
        for group in np.unique(groups)
    ]
    finite = [value for value in values if not math.isnan(value)]
    return float(np.mean(finite)) if finite else float("nan")


def binary_logloss(labels: np.ndarray, scores: np.ndarray) -> float:
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(scores, -40, 40)))
    probabilities = np.clip(probabilities, 1e-7, 1 - 1e-7)
    labels = labels.astype(np.float64, copy=False)
    return float(-(labels * np.log(probabilities) + (1 - labels) * np.log(1 - probabilities)).mean())
