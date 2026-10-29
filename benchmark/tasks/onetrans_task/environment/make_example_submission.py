#!/usr/bin/env python3
import csv
from pathlib import Path

rows = 1_460_572
path = Path("datasets/kuaisar_v2/working/submission.csv")
path.parent.mkdir(parents=True, exist_ok=True)
with path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow(["sample_id", "prediction"])
    writer.writerows((index, "0.5") for index in range(rows))
print(path)
