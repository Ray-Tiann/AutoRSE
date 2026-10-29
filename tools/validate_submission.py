#!/usr/bin/env python3
"""Validate prediction format locally. This never computes a hidden-test score."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from autorse.submissions import validate
from autorse.tasks import TASKS

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", required=True, choices=TASKS)
    p.add_argument("--submission", required=True, type=Path)
    a = p.parse_args()
    try:
        print(json.dumps(validate(a.task, a.submission), indent=2))
    except (ValueError, OSError, EOFError) as e:
        p.exit(2, f"Invalid submission: {e}\n")
