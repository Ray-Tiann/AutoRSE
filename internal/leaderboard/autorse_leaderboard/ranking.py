"""Build one reviewed, comparable six-task leaderboard from validated summaries."""
from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
import math
from urllib.parse import urlsplit

DATA_REVISION = "autorse-public-v2-20261029"
TASK_ORDER = ("HSTU", "IntSR", "TIGER", "OneTrans", "RankMixer", "WuKong")
TASKS = (
    {"name": "HSTU", "metric": "NDCG@10"},
    {"name": "IntSR", "metric": "Macro NDCG@10"},
    {"name": "TIGER", "metric": "NDCG@10"},
    {"name": "OneTrans", "metric": "Click AUC"},
    {"name": "RankMixer", "metric": "AUC"},
    {"name": "WuKong", "metric": "AUC"},
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite(value) -> bool:
    if type(value) not in (float, int):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _label(value, name: str) -> str:
    _require(isinstance(value, str) and bool(value.strip()), f"Missing {name}")
    return value


def _source_url(value) -> str:
    if value in (None, ""):
        return ""
    _require(isinstance(value, str), "Invalid source_url")
    parsed = urlsplit(value)
    _require(parsed.scheme == "https" and bool(parsed.netloc) and not parsed.username
             and not parsed.password and not any(c.isspace() for c in value), "Invalid source_url")
    return value


def _score_key(value) -> Decimal:
    # Decimal half-up makes the published precision policy explicit, independent
    # of binary floating-point rounding or the client language's formatting.
    return Decimal(str(value)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)


def build_leaderboard(summaries: list[dict]) -> dict:
    """Return the old row format, ranked only within one data-revision cohort.

    Input arrays follow TASK_ORDER. Every summary must be a validated benchmark
    submission with exactly three independent trials per task; demo, incomplete,
    duplicate-ID and mixed-version inputs fail rather than being silently pooled.
    Empty input is an honest public-v2 board with zero rows.
    """
    _require(isinstance(summaries, list), "Summaries must be a list")
    ids, versions, rows = set(), set(), []
    for summary in summaries:
        _require(isinstance(summary, dict), "Invalid summary")
        identifier = _label(summary.get("id"), "submission id")
        _require(identifier not in ids, "Duplicate submission id")
        ids.add(identifier)
        _require(summary.get("kind") == "benchmark", "Only benchmark submissions are eligible")
        _require(summary.get("publishable") is True, "Submission is not publishable")
        revision = _label(summary.get("data_revision"), "data revision")
        versions.add(revision)
        _require(len(versions) == 1, "Mixed data revisions cannot share a cohort")
        _label(summary.get("participant"), "participant")
        configuration = summary.get("configuration")
        _require(isinstance(configuration, dict), "Missing configuration")
        model = _label(configuration.get("model"), "model")
        harness = _label(configuration.get("harness"), "harness")
        scores, deviations, counts = (summary.get(key) for key in ("scores", "sd", "trial_counts"))
        _require(all(isinstance(values, list) and len(values) == 6
                     for values in (scores, deviations, counts)), "Six complete task statistics are required")
        _require(all(_finite(score) for score in scores), "Scores must be finite numbers")
        _require(all(_finite(sd) and sd >= 0 for sd in deviations), "Standard deviations must be finite and nonnegative")
        _require(all(type(count) is int and count == 3 for count in counts),
                 "Exactly three independent trials per task are required")
        rows.append({"id": identifier, "harness": harness, "model": model,
                     "scores": list(scores), "sd": list(deviations), "trial_counts": list(counts),
                     "provenance": "Reviewed community", "source_url": _source_url(summary.get("source_url")),
                     "task_ranks": [0.0] * 6})

    for index in range(6):
        ordered = sorted(rows, key=lambda row: (-_score_key(row["scores"][index]), row["id"]))
        start = 0
        while start < len(ordered):
            end = start + 1
            value = _score_key(ordered[start]["scores"][index])
            while end < len(ordered) and _score_key(ordered[end]["scores"][index]) == value:
                end += 1
            rank = (start + 1 + end) / 2
            for row in ordered[start:end]:
                row["task_ranks"][index] = rank
            start = end

    for row in rows:
        row["rank_sum"] = sum(row["task_ranks"])
        row["mean_task_rank"] = row["rank_sum"] / 6
    rows.sort(key=lambda row: (row["rank_sum"], row["model"].casefold(), row["harness"].casefold(), row["id"]))
    previous, rank = None, 0
    for position, row in enumerate(rows, 1):
        if row["rank_sum"] != previous:
            rank = position
        row["overall_rank"] = rank
        previous = row["rank_sum"]

    revision = next(iter(versions), DATA_REVISION)
    return {"metadata": {"version": 3, "schema_version": 1,
                         "title": "AutoRSE reviewed community leaderboard",
                         "cohort": "community:" + revision, "data_revision": revision,
                         "provenance": "Reviewed community", "configuration_count": len(rows),
                         "overall_cohort_count": len(rows), "task_count": 6,
                         "required_trials_per_task": 3,
                         "overall_ranking_metric": "mean_task_rank",
                         "overall_ranking_direction": "ascending",
                         "task_tie_policy": "Scores rounded to four decimals with decimal half-up; equal scores receive midranks.",
                         "overall_tie_policy": "Equal exact rank sums share a competition rank (1, 1, 3).",
                         "filter_policy": "Filters hide rows without recalculating ranks or changing the cohort.",
                         "comparison_policy": "Only reviewed complete benchmark submissions from this data revision; historical paper references are separate."},
            "task_order": list(TASK_ORDER), "tasks": [dict(task) for task in TASKS], "rows": rows}
