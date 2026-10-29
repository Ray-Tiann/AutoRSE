#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

from common import dataset_dir, input_dir, load_sequences


public = input_dir()
user_ids = np.load(public / "test" / "user_ids.npy")
validation = np.load(public / "validation" / "target_items.npy")
popularity = np.load(public / "item_popularity.npy")
order = np.argsort(-popularity[1:], kind="stable") + 1
store = load_sequences()
rows = store.rows_for(user_ids)
top_items = np.empty((len(user_ids), 10), dtype=np.int32)
for output_row, source_row in enumerate(rows):
    seen = set(map(int, store.sequence(int(source_row))))
    seen.add(int(validation[output_row]))
    selected = []
    for item in order:
        item = int(item)
        if item not in seen:
            selected.append(item)
            if len(selected) == 10:
                break
    top_items[output_row] = selected
output = dataset_dir() / "working" / "submission.npz"
output.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(output, user_ids=user_ids, top_items=top_items)
print(f"SUBMISSION={output}")
