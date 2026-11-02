"""Offline checks for public AutoRSE submission evidence.

The reader only fetches pinned bytes. This module never executes participant
code or interprets a trajectory as proof that an experiment actually happened.
Passing these checks makes a submission eligible for maintainer review, not an
approved leaderboard result.
"""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
import math
import re
import shlex
import statistics
from urllib.parse import urlsplit

from . import receipts
from .protocol import (
    TASK_ORDER, DATA_REVISION, PUBLIC_BUNDLE_SHA256, PRIMARY_METRICS,
    RESEARCH_BUDGET_SECONDS as RESEARCH_SECONDS,
    FINALIZATION_GRACE_SECONDS as FINALIZATION_SECONDS,
)


MAX_ARTIFACT_BYTES = 160 * 1024 * 1024
ARTIFACT_KINDS = ("code", "report", "trajectory", "config", "provenance")
_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_SLUG = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")
_REPO_PART = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}\Z")


def _text(value, limit=512):
    return (type(value) is str and 0 < len(value) <= limit
            and value == value.strip() and not any(ord(c) < 32 for c in value))


def _integer(value, expected=None):
    return type(value) is int and (expected is None or value == expected)


def _number(value):
    return type(value) is int or (type(value) is float and math.isfinite(value))


def _hex(value, expression):
    return type(value) is str and expression.fullmatch(value) is not None


def _json_problem(value, depth=0):
    """Exclude non-JSON Python types, non-finite numbers and excessive nesting."""
    if depth > 40:
        return "JSON nesting exceeds the supported limit."
    if value is None or type(value) in (bool, int, str):
        return None
    if type(value) is float:
        return None if math.isfinite(value) else "JSON numbers must be finite."
    if type(value) is list:
        for child in value:
            problem = _json_problem(child, depth + 1)
            if problem:
                return problem
        return None
    if type(value) is dict:
        for key, child in value.items():
            if type(key) is not str:
                return "JSON object keys must be strings."
            problem = _json_problem(child, depth + 1)
            if problem:
                return problem
        return None
    return "Only JSON objects, arrays and scalar values are accepted."


def _safe_repo(value):
    if type(value) is not str:
        return False
    parts = value.split("/")
    return (len(parts) == 2 and all(_REPO_PART.fullmatch(p) for p in parts)
            and all(".." not in p and "--" not in p and not p.endswith((".", ".git"))
                    for p in parts))


def _safe_path(value):
    if not _text(value, 1024) or value.startswith("/"):
        return False
    # Forbid both filesystem traversal and URL interpretation by a resolver.
    if any(c in value for c in "\\:%?#"):
        return False
    return all(part not in ("", ".", "..") for part in value.split("/"))


def _artifact_ref_problem(ref):
    if type(ref) is not dict:
        return "Every artifact requires a pinned file reference."
    if not _safe_repo(ref.get("repo_id")):
        return "Artifact repository must be an owner/name identifier."
    if not _hex(ref.get("revision"), _HEX40):
        return "Artifact revision must be a full lowercase Git commit."
    if not _safe_path(ref.get("path")):
        return "Artifact path must be a safe relative file path."
    if not _hex(ref.get("sha256"), _HEX64):
        return "Artifact SHA256 must be a lowercase 64-character digest."
    if not _integer(ref.get("size")) or not 0 < ref["size"] <= MAX_ARTIFACT_BYTES:
        return "Artifact size must be between 1 byte and 160 MiB."
    return None


def _https_url(value):
    if not _text(value, 2048) or any(c.isspace() for c in value) or "\\" in value:
        return False
    try:
        url = urlsplit(value)
        return (url.scheme == "https" and bool(url.hostname)
                and url.username is None and url.password is None)
    except ValueError:
        return False


def _timestamp(value):
    if not _text(value, 80):
        raise ValueError("Invalid timestamp")
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError("A timezone is required")
    return result


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _json_bytes(data):
    def reject_constant(_):
        raise ValueError("Non-finite JSON number")
    value = json.loads(data.decode("utf-8"), object_pairs_hook=_unique_pairs,
                       parse_constant=reject_constant)
    if _json_problem(value):
        raise ValueError("Invalid JSON structure")
    return value


def _config_problem(config, run, configuration):
    if type(config) is not dict or not config:
        return "Execution configuration must be a nonempty JSON object."
    expected = {
        "research_budget_seconds": RESEARCH_SECONDS,
        "minimum_runtime_sec": RESEARCH_SECONDS,
        "gpus": 1,
        "seed": run["seed"],
    }
    for key, value in expected.items():
        if not _integer(config.get(key), value):
            return "Configuration must declare the fixed budget, one GPU and this run's seed."
    grace = config.get("finalization_grace_sec")
    if not _integer(grace) or not 0 <= grace <= FINALIZATION_SECONDS:
        return "Configuration finalization grace must be at most 300 seconds."
    if config.get("harness") != configuration["harness"] or config.get("model") != configuration["model"]:
        return "Configuration harness and model must match the submitted configuration."
    command = config.get("command")
    if type(command) is not list or not command or not all(_text(v, 8192) for v in command):
        return "Configuration command must be a nonempty array of arguments."

    budgets = {"research_budget_seconds", "research_seconds", "minimum_runtime_sec",
               "minimum_runtime_seconds", "minimum_research_seconds"}
    graces = {"finalization_grace_sec", "finalization_grace_seconds"}
    gpu_counts = {"gpus", "gpus_per_run", "num_gpus", "n_gpus", "gpu_count"}
    # Ordinary setup/task timeouts are intentionally not treated as agent time.
    def inspect(value):
        if type(value) is list:
            return next((p for item in value if (p := inspect(item))), None)
        if type(value) is not dict:
            return None
        for key, child in value.items():
            normalized = key.lower().replace("-", "_")
            empty = child is None or child is False or child == "" or child == [] or child == {}
            if "override" in normalized and not empty:
                return "Timeout and resource override settings are not permitted."
            if normalized in budgets and not _integer(child, RESEARCH_SECONDS):
                return "Configuration contains a conflicting research budget."
            if normalized in graces and (not _integer(child) or not 0 <= child <= FINALIZATION_SECONDS):
                return "Configuration contains a conflicting finalization grace."
            if normalized in gpu_counts and not _integer(child, 1):
                return "Configuration must allocate exactly one GPU per run."
            if normalized == "timeout_multiplier" and (not _number(child) or child != 1):
                return "Timeout multipliers other than one are not permitted."
            if normalized == "no_persistence" and child is not False:
                return "The fixed research window must not be disabled."
            if normalized == "persistence" and child is not True:
                return "The fixed research window must not be disabled."
            problem = inspect(child)
            if problem:
                return problem
        return None

    problem = inspect(config)
    if problem:
        return problem
    # Inspect arguments as data; do not execute or evaluate shell expressions.
    tokens = []
    for argument in command:
        try:
            tokens.extend(shlex.split(argument))
        except ValueError:
            return "Configuration command contains malformed argument quoting."
    fixed_flags = {"--minimum-runtime-sec": RESEARCH_SECONDS, "--research-budget-seconds": RESEARCH_SECONDS,
                   "--gpus": 1, "--gpus-per-run": 1, "--n-concurrent": 1, "--n-attempts": 1}
    for index, token in enumerate(tokens):
        flag, separator, inline = token.partition("=")
        value = inline if separator else (tokens[index + 1] if index + 1 < len(tokens) else None)
        if flag == "--no-persistence" or flag.startswith(("--override-", "--override_")):
            return "Configuration command contains a prohibited override."
        if flag in fixed_flags and value != str(fixed_flags[flag]):
            return "Configuration command conflicts with the fixed execution protocol."
        if flag == "--finalization-grace-sec":
            if value is None or not value.isdigit() or not 0 <= int(value) <= FINALIZATION_SECONDS:
                return "Configuration command exceeds the finalization grace."
        if flag == "--timeout-multiplier" and value not in ("1", "1.0"):
            return "Configuration command contains a timeout multiplier."
        if flag == "--gpu" and (not value or any(c in value for c in ", \t\n")):
            return "Configuration command must select exactly one GPU."
        if flag == "--agent-kwarg" and value and "=" in value:
            key, raw = value.split("=", 1)
            normalized = key.lower().replace("-", "_")
            if "override" in normalized or normalized in ("no_persistence", "timeout_multiplier"):
                return "Configuration agent arguments contain a prohibited override."
            if normalized in budgets and raw != str(RESEARCH_SECONDS):
                return "Configuration agent arguments conflict with the fixed research budget."
            if normalized in graces and (not raw.isdigit() or not 0 <= int(raw) <= FINALIZATION_SECONDS):
                return "Configuration agent arguments exceed the finalization grace."
    return None


def _provenance_problem(value, run):
    if type(value) is not dict or value.get("evaluated_code_commit") != run["code_commit"]:
        return "Provenance must identify the evaluated code commit."
    nested = value.get("run", {})
    if type(nested) is not dict:
        return "Provenance run record must be an object."
    predictions = []
    for container in (value, nested):
        if "prediction" in container:
            prediction = container["prediction"]
            if type(prediction) is not dict or prediction.get("sha256") != run["prediction_sha256"]:
                return "Provenance prediction digest must match the signed evaluation."
            predictions.append(prediction)
    if not predictions:
        return "Provenance must bind the prediction digest to the evaluated code."
    for key in ("code_commit", "run_id", "seed"):
        if key in nested and (type(nested[key]) is not type(run[key]) or nested[key] != run[key]):
            return "Provenance run identifiers must match the submission record."
    return None


def _trajectory_problem(value):
    if type(value) is dict:
        sequences = [value.get(key) for key in ("steps", "messages", "events", "trajectory")]
    elif type(value) is list:
        sequences = [value]
    else:
        sequences = []
    for sequence in sequences:
        if type(sequence) is list and sequence and all(type(item) is dict and item for item in sequence):
            return None
    return "Trajectory must contain a nonempty array of structured steps, messages or events."


def validate_submission(submission, trusted_keys: dict, *, artifact_reader=None, allow_test=False) -> dict:
    """Return safe automatic-check results without executing submitted artifacts.

    ``artifact_reader(ref)`` must return the referenced public file as bytes. A
    metadata-only call returns ``valid=False`` with ``metadata_valid=True`` when
    appropriate: missing file verification can never silently authorize a row.
    ``kind=test`` is accepted only with ``allow_test=True`` and is never publishable.
    """
    checks = []

    def check(code, passed, message):
        checks.append({"code": code, "passed": bool(passed), "message": message})
        return bool(passed)

    def result(summary=None):
        automatic_valid = bool(checks) and all(c["passed"] for c in checks)
        metadata_valid = all(c["passed"] for c in checks if not c["code"].startswith("artifacts."))
        summary = summary or {}
        publishable = automatic_valid and summary.get("kind") == "benchmark"
        if summary:
            summary.update(publishable=publishable, requires_maintainer_review=True)
        return {"valid": automatic_valid, "metadata_valid": metadata_valid,
                "publishable": publishable, "checks": checks, "summary": summary}

    problem = _json_problem(submission)
    if not check("schema.json", problem is None and type(submission) is dict,
                 problem or "Submission must be a JSON object with finite numbers."):
        return result()
    check("schema.version", _integer(submission.get("schema_version"), 1), "Submission schema version must be 1.")
    identifier = submission.get("id")
    check("schema.id", type(identifier) is str and len(identifier) <= 80 and bool(_SLUG.fullmatch(identifier)),
          "Submission ID must be a lowercase slug of at most 80 characters.")
    kind = submission.get("kind")
    check("schema.kind", kind == "benchmark" or (allow_test is True and kind == "test"),
          "Only benchmark submissions, or explicitly enabled pipeline tests, are accepted.")

    configuration = submission.get("configuration")
    configuration_ok = type(configuration) is dict
    if configuration_ok:
        configuration_ok = all(_text(configuration.get(k)) for k in (
            "name", "agent", "harness", "agent_version", "model", "model_version"))
        configuration_ok = (configuration_ok and _hex(configuration.get("framework_commit"), _HEX40)
                            and _https_url(configuration.get("code_url")))
    check("schema.configuration", configuration_ok,
          "Configuration requires named, versioned agent/model/harness, a full framework commit and an HTTPS code URL.")
    protocol = submission.get("protocol")
    protocol_ok = (type(protocol) is dict and protocol.get("data_revision") == DATA_REVISION
                   and protocol.get("public_bundle_sha256") == PUBLIC_BUNDLE_SHA256
                   and _integer(protocol.get("research_budget_seconds"), RESEARCH_SECONDS)
                   and _integer(protocol.get("gpus_per_run"), 1))
    check("protocol.pins", protocol_ok, "Data revision, public bundle, 12-hour research budget and one-GPU protocol must match.")
    acknowledgements = submission.get("acknowledgements")
    check("schema.acknowledgements", type(acknowledgements) is dict
          and acknowledgements.get("public_evidence") is True and acknowledgements.get("protocol_followed") is True,
          "Public-evidence and protocol acknowledgements must both be true.")

    signed_receipts = submission.get("receipts")
    if not check("receipts.count", type(signed_receipts) is list and len(signed_receipts) == 3,
                 "Exactly three complete, independently evaluated six-task receipts are required."):
        return result()
    # Verify every signature before inspecting or aggregating any receipt fields.
    verified = []
    for index, envelope in enumerate(signed_receipts, 1):
        try:
            payload = receipts.verify_receipt(envelope, trusted_keys)
            if type(payload) is not dict or _json_problem(payload):
                raise ValueError("Invalid verified payload")
        except (ValueError, TypeError, KeyError):
            check(f"receipts.signature.{index}", False, "Receipt signature is missing, invalid or not trusted.")
            verified.append(None)
        else:
            check(f"receipts.signature.{index}", True, "Receipt signature verified against a trusted evaluator key.")
            verified.append(payload)
    if any(payload is None for payload in verified):
        return result()

    receipt_by_id, participants = {}, []
    for index, payload in enumerate(verified, 1):
        code = f"receipts.protocol.{index}"
        evaluation_id = payload.get("evaluation_id")
        participant = payload.get("participant")
        receipt_ok = (_integer(payload.get("schema_version"), 1) and payload.get("complete") is True
                      and payload.get("status") == "complete" and _integer(payload.get("task_count"), 6)
                      and payload.get("task_order") == list(TASK_ORDER)
                      and payload.get("usage") == "benchmark" and _text(evaluation_id, 128)
                      and _text(participant, 96) and bool(_REPO_PART.fullmatch(participant))
                      and payload.get("data_revision") == DATA_REVISION
                      and payload.get("public_bundle_sha256") == PUBLIC_BUNDLE_SHA256
                      and _hex(payload.get("batch_sha256"), _HEX64)
                      and _hex(payload.get("evaluator_code_sha256"), _HEX64))
        check(code, receipt_ok, "Receipt must be complete, participant-bound, non-demo and pinned to the public protocol.")
        participants.append(participant)
        tasks = payload.get("tasks")
        tasks_ok = type(tasks) is dict and set(tasks) == set(TASK_ORDER)
        task_values = {}
        if tasks_ok:
            for task in TASK_ORDER:
                item = tasks[task]
                if type(item) is not dict:
                    tasks_ok = False
                    break
                metric = PRIMARY_METRICS[task]
                metrics = item.get("metrics")
                score = metrics.get(metric) if type(metrics) is dict else None
                item_ok = (item.get("task") == task and item.get("metric") == metric
                           and item.get("data_revision") == DATA_REVISION
                           and item.get("public_bundle_sha256") == PUBLIC_BUNDLE_SHA256
                           and _hex(item.get("submission_sha256"), _HEX64)
                           and _number(score) and 0 <= score <= 1)
                if type(metrics) is dict and "reward" in metrics:
                    item_ok = item_ok and _number(metrics["reward"]) and metrics["reward"] == score
                if not item_ok:
                    tasks_ok = False
                    break
                task_values[task] = score
        check(f"receipts.scores.{index}", tasks_ok, "Each receipt requires all six finite primary metrics and task prediction digests.")
        if _text(evaluation_id, 128):
            if evaluation_id in receipt_by_id:
                check("receipts.unique", False, "Receipt evaluation IDs must be distinct.")
            else:
                receipt_by_id[evaluation_id] = {"payload": payload, "scores": task_values}
    check("receipts.participant", all(type(p) is str for p in participants) and len(set(participants)) == 1,
          "All three receipts must belong to the same participant account.")

    runs = submission.get("runs")
    if not check("runs.count", type(runs) is list and len(runs) == 18,
                 "Exactly 18 runs are required: six tasks and three independent trials."):
        return result()
    run_ids, pairs, seeds = set(), set(), {task: set() for task in TASK_ORDER}
    trial_receipts = {trial: set() for trial in (1, 2, 3)}
    valid_run_records = []
    for index, run in enumerate(runs, 1):
        if type(run) is not dict:
            check(f"runs.record.{index}", False, "Each run must be an object.")
            continue
        task, trial = run.get("task"), run.get("trial_index")
        basic_ok = (type(task) is str and task in TASK_ORDER and _integer(trial) and trial in (1, 2, 3)
                    and _text(run.get("run_id"), 160) and _integer(run.get("seed"))
                    and _text(run.get("receipt_evaluation_id"), 128)
                    and _hex(run.get("prediction_sha256"), _HEX64)
                    and _hex(run.get("code_commit"), _HEX40) and _integer(run.get("gpus"), 1))
        check(f"runs.record.{index}", basic_ok, "Run identifiers, trial, seed, code/prediction digests and one-GPU declaration are required.")
        if not basic_ok:
            continue
        check(f"runs.unique.{index}", run["run_id"] not in run_ids and (task, trial) not in pairs,
              "Run IDs and task/trial pairs must be unique.")
        run_ids.add(run["run_id"])
        pairs.add((task, trial))
        seeds[task].add(run["seed"])
        trial_receipts[trial].add(run["receipt_evaluation_id"])
        receipt = receipt_by_id.get(run["receipt_evaluation_id"])
        match = receipt is not None
        if match:
            receipt_tasks = receipt["payload"].get("tasks")
            item = receipt_tasks.get(task) if type(receipt_tasks) is dict else None
            match = type(item) is dict and item.get("submission_sha256") == run["prediction_sha256"]
        check(f"runs.receipt_binding.{index}", match, "Run prediction digest must match its trusted, complete evaluation receipt.")
        try:
            finished = _timestamp(run.get("agent_finished_at"))
            elapsed = (finished - _timestamp(run.get("agent_started_at"))).total_seconds()
            duration_ok = RESEARCH_SECONDS <= elapsed <= RESEARCH_SECONDS + FINALIZATION_SECONDS
        except (ValueError, TypeError, OverflowError):
            duration_ok = False
        check(f"runs.declared_budget.{index}", duration_ok,
              "Declared agent timestamps must cover 12 hours plus at most 300 seconds of finalization; actual execution needs review.")
        try:
            evaluated = _timestamp(receipt["payload"].get("evaluated_at")) if receipt else None
            evaluation_order_ok = evaluated is not None and _timestamp(run.get("agent_finished_at")) <= evaluated
        except (ValueError, TypeError, OverflowError):
            evaluation_order_ok = False
        check(f"runs.evaluation_order.{index}", evaluation_order_ok,
              "Declared agent completion must not be later than the associated signed evaluation time.")
        artifacts = run.get("artifacts")
        refs_ok = type(artifacts) is dict and set(ARTIFACT_KINDS).issubset(artifacts)
        if refs_ok:
            refs_ok = all(_artifact_ref_problem(artifacts[kind]) is None for kind in ARTIFACT_KINDS)
        check(f"runs.artifact_refs.{index}", refs_ok,
              "Code, report, trajectory, configuration and provenance require safe, commit-pinned, size/hash-bound file references.")
        if refs_ok:
            valid_run_records.append((index, run))
    expected_pairs = {(task, trial) for task in TASK_ORDER for trial in (1, 2, 3)}
    check("runs.coverage", pairs == expected_pairs and len(run_ids) == 18,
          "All six tasks must be represented exactly once in each of three trials.")
    check("runs.seeds", all(len(values) == 3 for values in seeds.values()),
          "Each task requires three distinct integer seeds.")
    group_ids = [next(iter(ids)) for ids in trial_receipts.values() if len(ids) == 1]
    check("runs.trialsets", len(group_ids) == 3 and len(set(group_ids)) == 3
          and set(group_ids) == set(receipt_by_id),
          "Each six-task trial must use one distinct complete receipt, and all three receipts must be used.")

    metadata_ok = all(c["passed"] for c in checks)
    summary = {}
    if metadata_ok:
        task_scores = {task: [receipt_by_id[next(iter(trial_receipts[trial]))]["scores"][task]
                              for trial in (1, 2, 3)] for task in TASK_ORDER}
        means = [statistics.mean(task_scores[task]) for task in TASK_ORDER]
        deviations = [statistics.stdev(task_scores[task]) for task in TASK_ORDER]
        summary = {"id": identifier, "kind": kind, "configuration": dict(configuration),
                   "participant": participants[0], "data_revision": DATA_REVISION,
                   "public_bundle_sha256": PUBLIC_BUNDLE_SHA256, "task_order": list(TASK_ORDER),
                   "scores": means, "sd": deviations, "trial_counts": [3] * 6,
                   "score_by_task": dict(zip(TASK_ORDER, means)), "task_scores": task_scores,
                   "evaluation_ids": group_ids}
    if artifact_reader is None:
        check("artifacts.pending", False, "Pinned evidence bytes have not been fetched; this metadata-only report cannot publish a result.")
        return result(summary)
    if not callable(artifact_reader):
        check("artifacts.reader", False, "Artifact reader must be a callable returning pinned file bytes.")
        return result(summary)
    if not metadata_ok:
        return result(summary)

    cache = {}
    for index, run in valid_run_records:
        for artifact_kind in ARTIFACT_KINDS:
            ref = run["artifacts"][artifact_kind]
            key = tuple(ref[k] for k in ("repo_id", "revision", "path", "sha256", "size"))
            code = f"artifacts.integrity.{index}.{artifact_kind}"
            if key not in cache:
                try:
                    data = artifact_reader(dict(ref))
                    if type(data) is not bytes or len(data) != ref["size"] or hashlib.sha256(data).hexdigest() != ref["sha256"]:
                        raise ValueError("Artifact integrity mismatch")
                    cache[key] = data
                except Exception:
                    cache[key] = None
            data = cache[key]
            if not check(code, data is not None, "Pinned artifact must be readable and match its declared size and SHA256."):
                continue
            problem = None
            if artifact_kind in ("config", "provenance", "trajectory"):
                try:
                    value = _json_bytes(data)
                    if artifact_kind == "config":
                        problem = _config_problem(value, run, configuration)
                    elif artifact_kind == "provenance":
                        problem = _provenance_problem(value, run)
                    else:
                        problem = _trajectory_problem(value)
                except (ValueError, UnicodeError, TypeError, RecursionError):
                    problem = "Structured evidence must be unambiguous, finite UTF-8 JSON."
            elif artifact_kind == "report":
                try:
                    text = data.decode("utf-8")
                    if len(text.strip()) < 20 or "\x00" in text:
                        problem = "Report must be readable, nonempty UTF-8 text describing the run."
                except UnicodeError:
                    problem = "Report must be readable UTF-8 text."
            if artifact_kind != "code":
                check(f"artifacts.format.{index}.{artifact_kind}", problem is None,
                      problem or "Evidence structure is valid; this check does not certify the claimed experiment or originality.")
    return result(summary)
