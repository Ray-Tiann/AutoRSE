#!/usr/bin/env python3
"""Create a label-free, format-valid IntSR submission."""

from pathlib import Path
import pickle
import sys

import numpy as np


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "datasets/Amazon/input/test_data_unlabeled.pkl"
OUTPUT = ROOT / "datasets/Amazon/working/submission.npz"


def load_numpy_compatible_pickle(path):
    """Load NumPy 1.x/2.x pickles without upgrading the PyTorch runtime."""
    with path.open("rb") as handle:
        try:
            return pickle.load(handle)
        except ModuleNotFoundError as exc:
            if not (exc.name or "").startswith("numpy._core"):
                raise
            handle.seek(0)
            import pandas  # noqa: F401
            import numpy.core.multiarray as legacy_multiarray
            import numpy.core.numeric as legacy_numeric

            sys.modules.setdefault("numpy._core.multiarray", legacy_multiarray)
            sys.modules.setdefault("numpy._core.numeric", legacy_numeric)
            return pickle.load(handle)


def main() -> None:
    payload = load_numpy_compatible_pickle(INPUT)

    sample_ids = []
    task_ids = []
    candidates = []
    scores = []
    position_scores = np.linspace(1.0, 0.0, 100, dtype=np.float32)
    for task_id, name in enumerate(("rec", "src")):
        frame = payload[name]
        task_candidates = np.stack(frame["candidate_items"].to_numpy()).astype(
            np.int32
        )
        sample_ids.append(frame["sample_id"].to_numpy(dtype=np.int64))
        task_ids.append(np.full(len(frame), task_id, dtype=np.uint8))
        candidates.append(task_candidates)
        scores.append(np.broadcast_to(position_scores, task_candidates.shape).copy())

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT,
        sample_ids=np.concatenate(sample_ids),
        task_ids=np.concatenate(task_ids),
        candidate_items=np.concatenate(candidates),
        scores=np.concatenate(scores),
    )
    print(f"EXAMPLE_SUBMISSION={OUTPUT}")


if __name__ == "__main__":
    main()
