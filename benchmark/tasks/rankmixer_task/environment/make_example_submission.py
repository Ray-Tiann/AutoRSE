#!/usr/bin/env python3
from pathlib import Path

import numpy as np


root = Path(__file__).resolve().parent
ids = np.load(root / "datasets/criteo5m/input/test/sample_ids.npy", mmap_mode="r")
output = root / "datasets/criteo5m/working/submission.npz"
output.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(
    output,
    sample_ids=np.asarray(ids, dtype=np.int64),
    scores=np.full(len(ids), 0.5, dtype=np.float32),
)
print(output)
