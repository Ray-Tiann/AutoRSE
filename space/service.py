"""Hosted scoring boundary. Private labels are never returned or served as files."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from autorse.submissions import validate
from autorse.tasks import TASKS, task_name
from autorse.release import DATA_REVISION, PUBLIC_BUNDLE_SHA256

PRIVATE_FILES = {
    "hstu": ["beauty/expected_user_ids.npy", "beauty/catalog.npy", "beauty/test_targets.npy"],
    "tiger": ["beauty/expected_user_ids.npy", "beauty/catalog.npy", "beauty/test_targets.npy"],
    "intsr": ["Amazon/expected_submission.npz", "Amazon/test_data_labeled.csv"],
    "wukong": ["movielenslatest/test_data_labeled.csv"],
    "rankmixer": ["criteo5m/expected_sample_ids.npy", "criteo5m/test_labels.npy"],
    "onetrans": ["kuaisar_v2/labels.npz"],
}
PRIVATE_ROOT = Path(os.environ.get("AUTORSE_PRIVATE_DIR", "/tmp/autorse-private"))


class SubmissionRejected(Exception):
    pass


class EvaluatorUnavailable(Exception):
    pass


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def provision_private_data() -> None:
    """Only the Space owner's read token can fetch the private dataset."""
    repo = os.environ.get("AUTORSE_PRIVATE_REPO")
    revision = os.environ.get("AUTORSE_PRIVATE_REVISION")
    expected_manifest_sha = os.environ.get("AUTORSE_PRIVATE_MANIFEST_SHA256")
    token = os.environ.get("HF_TOKEN")
    if not all((repo, revision, expected_manifest_sha, token)):
        raise EvaluatorUnavailable("Private evaluator configuration is incomplete")
    from huggingface_hub import HfApi, hf_hub_download
    try:
        info = HfApi(token=token).dataset_info(repo, revision=revision)
        if not info.private or len(revision) != 40 or info.sha != revision:
            raise ValueError("Evaluator must use a private repository pinned to a commit")
        PRIVATE_ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
        def fetch(name):
            return Path(hf_hub_download(repo, filename=name, repo_type="dataset", revision=revision,
                                       token=token, cache_dir=PRIVATE_ROOT / ".hub"))
        manifest_path = fetch("private-manifest.json")
        if sha256(manifest_path) != expected_manifest_sha:
            raise ValueError("Private manifest differs")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("public_bundle_sha256") != PUBLIC_BUNDLE_SHA256:
            raise ValueError("Private evaluator is paired with a different public data revision")
        rows = manifest["files"]
        by_name = {row["path"]: row for row in rows}
        expected = {f"{task}/{rel}" for task, paths in PRIVATE_FILES.items() for rel in paths}
        if len(rows) != len(by_name) or set(by_name) != expected:
            raise ValueError("Private inventory differs")
        for name, row in by_name.items():
            source = fetch(name)
            if source.stat().st_size != row["size"] or sha256(source) != row["sha256"]:
                raise ValueError("Private file verification failed")
            destination = PRIVATE_ROOT / name
            destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            shutil.copyfile(source, destination)
            destination.chmod(0o600)
    except Exception:
        # No Hub URLs, credentials, paths or label values enter the user response.
        raise EvaluatorUnavailable("Private evaluator initialization failed") from None


def score_submission(task: str, path: Path, *, private_root: Path = PRIVATE_ROOT) -> dict:
    task = task_name(task)
    try:
        validate(task, path)
    except Exception:
        raise SubmissionRejected("Invalid prediction format; check the public submission contract.") from None
    label_root = Path(private_root) / task
    if not all((label_root / rel).is_file() for rel in PRIVATE_FILES[task]):
        raise EvaluatorUnavailable("This task's evaluator is unavailable")
    digest = sha256(path)
    with tempfile.TemporaryDirectory(prefix="autorse-score-") as directory:
        work = Path(directory)
        spec = TASKS[task]
        prediction = work / "workspace" / "datasets" / spec["dataset"] / "working" / spec["file"]
        prediction.parent.mkdir(parents=True)
        shutil.copyfile(path, prediction)
        # The scorer is trusted, fixed source. No user code/checkpoint/pickle is executed.
        command = [sys.executable, str(Path(__file__).parent / "scorers" / f"{task}.py"),
                   "--workspace", str(work / "workspace"), "--private-dir", str(label_root),
                   "--reward-path", str(work / "reward.json"), "--report-path", str(work / "metrics.json")]
        child_env = {"PATH": os.defpath, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
                     "MKL_NUM_THREADS": "1", "PYTHONIOENCODING": "utf-8"}
        try:
            completed = subprocess.run(command, env=child_env, cwd=work, stdout=subprocess.DEVNULL,
                                       stderr=subprocess.DEVNULL, timeout=180, check=False)
            if completed.returncode != 0:
                raise ValueError("Scorer failed")
            report = json.loads((work / "metrics.json").read_text())
            if "error" in report:
                raise SubmissionRejected("Prediction identities or values do not match the task contract.")
            reward = json.loads((work / "reward.json").read_text())
            # Return only aggregates computed by fixed scorers, never arbitrary reports/errors.
            allowed = {"reward", "auc", "uauc", "logloss", "ndcg_at_5", "ndcg_at_10",
                       "recall_at_5", "recall_at_10", "mrr_at_10", "macro_ndcg_at_10",
                       "rec_ndcg_at_10", "src_ndcg_at_10"}
            import math
            metrics = {k: float(v) for k, v in reward.items()
                       if k in allowed and isinstance(v, (int, float)) and math.isfinite(v)}
            if spec["metric"] not in metrics:
                raise ValueError("Missing task metric")
        except SubmissionRejected:
            raise
        except Exception:
            raise EvaluatorUnavailable("Scoring did not complete; contact the benchmark maintainer.") from None
    return {"task": task, "data_revision": DATA_REVISION, "public_bundle_sha256": PUBLIC_BUNDLE_SHA256,
            "submission_sha256": digest, "metric": spec["metric"], "metrics": metrics}
