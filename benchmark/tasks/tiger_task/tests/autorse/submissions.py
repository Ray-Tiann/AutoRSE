"""Bounded, non-executing validation of prediction files, without hidden scores."""
from __future__ import annotations

import csv
import gzip
import math
from pathlib import Path
import zipfile

import numpy as np

from .tasks import TASKS, task_name

MAX_UPLOAD_BYTES = 160 * 1024**2
MAX_EXPANDED_BYTES = 512 * 1024**2


def check_npz(path: Path, spec: dict) -> None:
    """Check ZIP membership and NPY headers before allocating arrays."""
    expected = spec["arrays"]
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        if {i.filename for i in infos} != {f"{key}.npy" for key in expected} or len(infos) != len(expected):
            raise ValueError("Unexpected or duplicate NPZ arrays")
        if sum(i.file_size for i in infos) > MAX_EXPANDED_BYTES:
            raise ValueError("Expanded submission exceeds limit")
        for info in infos:
            if info.flag_bits & 1:
                raise ValueError("Encrypted submissions are unsupported")
            with archive.open(info) as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version == (2, 0):
                    shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise ValueError("Unsupported NPY format")
                name = info.filename[:-4]
                if shape != expected[name] or dtype.hasobject or dtype.kind not in "iuf":
                    raise ValueError("Invalid array shape or dtype")
                if dtype.itemsize > 8 or math.prod(shape) * dtype.itemsize != info.file_size - stream.tell():
                    raise ValueError("Invalid array payload length")
    with np.load(path, allow_pickle=False) as data:
        for name in expected:
            array = data[name]
            if name != "scores" and array.dtype.kind not in "iu":
                raise ValueError("Identity and item arrays must be integers")
            if not np.isfinite(array).all():
                raise ValueError("Non-finite values")
        if "scores" in data and spec["metric"] == "auc":
            if np.any(data["scores"] < 0) or np.any(data["scores"] > 1):
                raise ValueError("Scores must be probabilities in [0, 1]")
        if "top_items" in data:
            items = data["top_items"]
            if np.any(items < 1) or np.any(items > 12101) or np.any(np.diff(np.sort(items, axis=1), axis=1) == 0):
                raise ValueError("Top-10 must contain distinct catalog items")


def check_csv(path: Path, spec: dict) -> None:
    opener = gzip.open if spec["file"].endswith(".gz") else open
    size = 0
    count = 0
    # Iterate text lines so a highly compressible gzip cannot allocate unbounded data.
    def bounded_lines(stream):
        nonlocal size
        while line := stream.readline(4097):
            size += len(line.encode("utf-8"))
            if len(line) > 4096 or size > MAX_EXPANDED_BYTES:
                raise ValueError("CSV exceeds line or expanded-size limit")
            yield line
    with opener(path, "rt", newline="", encoding="utf-8") as stream:
        reader = csv.reader(bounded_lines(stream))
        if next(reader, None) != spec["columns"]:
            raise ValueError("Incorrect CSV columns")
        for row in reader:
            count += 1
            if len(row) != 2 or not row[0] or count > spec["rows"]:
                raise ValueError("Invalid CSV rows")
            value = float(row[1])
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Predictions must be finite probabilities in [0, 1]")
    if count != spec["rows"]:
        raise ValueError("Incorrect prediction row count")


def validate(task: str, path: Path) -> dict:
    task = task_name(task)
    spec = TASKS[task]
    path = Path(path)
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= MAX_UPLOAD_BYTES:
        raise ValueError("Missing, unsafe or oversized submission")
    suffix = ".csv.gz" if spec["file"].endswith(".gz") else Path(spec["file"]).suffix
    if not path.name.endswith(suffix):
        raise ValueError("Incorrect submission file type")
    if "arrays" in spec:
        check_npz(path, spec)
    else:
        check_csv(path, spec)
    return {"task": task, "format_valid": True, "rows": spec["rows"],
            "hidden_test_scored": False, "note": "Format checks do not verify hidden IDs or compute a score."}
