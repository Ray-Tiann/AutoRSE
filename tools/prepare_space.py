#!/usr/bin/env python3
"""Prepare a minimal HF Space upload directory, without datasets or secrets."""
import argparse
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]

def prepare(output):
    if output.exists():
        raise ValueError("Output exists; choose a new directory")
    output.mkdir(parents=True)
    for name in ("autorse", "space"):
        shutil.copytree(ROOT / name, output / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copyfile(ROOT / "space/Dockerfile", output / "Dockerfile")
    shutil.copyfile(ROOT / "space/README.md", output / "README.md")
    shutil.copytree(ROOT / "docs", output / "docs")
    for name in ("LICENSE", "THIRD_PARTY_NOTICES.md"):
        if (ROOT / name).exists():
            shutil.copyfile(ROOT / name, output / name)
    print(f"Space source prepared: {output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    prepare(parser.parse_args().output.resolve())
