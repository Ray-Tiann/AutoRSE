#!/usr/bin/env python3
"""Create a label-free constant-probability WuKong submission."""

import csv
import gzip
from pathlib import Path


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "datasets/movielenslatest/input/test_data_unlabeled.csv"
OUTPUT = ROOT / "datasets/movielenslatest/working/submission.csv.gz"


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with INPUT.open(newline="", encoding="utf-8") as source, gzip.open(
        OUTPUT, "wt", newline="", encoding="utf-8"
    ) as destination:
        reader = csv.DictReader(source)
        if not reader.fieldnames or reader.fieldnames[0] != "sample_id":
            raise ValueError("unlabeled test data must begin with sample_id")
        writer = csv.writer(destination)
        writer.writerow(["sample_id", "score"])
        for row in reader:
            writer.writerow([row["sample_id"], "0.5"])
    print(f"EXAMPLE_SUBMISSION={OUTPUT}")


if __name__ == "__main__":
    main()
