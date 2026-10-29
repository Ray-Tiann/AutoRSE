#!/usr/bin/env python3
"""Maintainer only: prepare labels for a PRIVATE HF dataset, never the public dataset."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from autorse.release import PUBLIC_BUNDLE_SHA256, INTSR_IDENTITIES_SHA256

FILES = {
    "hstu": ["beauty/expected_user_ids.npy", "beauty/catalog.npy", "beauty/test_targets.npy"],
    "tiger": ["beauty/expected_user_ids.npy", "beauty/catalog.npy", "beauty/test_targets.npy"],
    "intsr": ["Amazon/expected_submission.npz", "Amazon/test_data_labeled.csv"],
    "wukong": ["movielenslatest/test_data_labeled.csv"],
    "rankmixer": ["criteo5m/expected_sample_ids.npy", "criteo5m/test_labels.npy"],
    "onetrans": ["kuaisar_v2/labels.npz"],
}


def package(source, output, intsr_identities):
    if output.exists():
        raise ValueError("Output must be a new, separate PRIVATE directory")
    sources = [(task, rel, source / f"tasks/{task}_task/tests/private" / rel)
               for task, paths in FILES.items() for rel in paths]
    with intsr_identities.open("rb") as f:
        if hashlib.file_digest(f, "sha256").hexdigest() != INTSR_IDENTITIES_SHA256:
            raise ValueError("IntSR public identities do not match the reviewed v2 revision")
    sources = [(task, rel, intsr_identities if task == "intsr" and rel == "Amazon/expected_submission.npz" else path)
               for task, rel, path in sources]
    if any(p.is_symlink() or not p.is_file() for _, _, p in sources):
        raise ValueError("Required evaluator files missing; do not substitute labels")
    # Check the entire inventory before creating any output. Files are copied and
    # hashed as opaque bytes: this utility does not deserialize the hidden data.
    output.mkdir(parents=True, mode=0o700)
    rows = []
    for task, relative, source_file in sources:
        name = f"{task}/{relative}"
        destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        shutil.copyfile(source_file, destination)
        destination.chmod(0o600)
        with destination.open("rb") as f:
            digest = hashlib.file_digest(f, "sha256").hexdigest()
        rows.append({"path": name, "size": destination.stat().st_size, "sha256": digest})
    manifest = output / "private-manifest.json"
    manifest.write_text(json.dumps({"schema_version": 1, "visibility": "PRIVATE_EVALUATOR_ONLY",
                                    "public_bundle_sha256": PUBLIC_BUNDLE_SHA256, "files": rows}, indent=2) + "\n")
    manifest.chmod(0o600)
    (output / "README.md").write_text("---\npretty_name: AutoRSE Private Evaluator\n---\n\n# PRIVATE evaluator inputs\n\nKeep this repository PRIVATE. Never upload this folder to the public Dataset, GitHub, or Space source.\nOnly the scoring Space owner's read token should access it.\n")
    print(json.dumps({"private_files": len(rows), "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                      "visibility": "PRIVATE_EVALUATOR_ONLY", "contents_parsed": False}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-benchmark", type=Path, required=True)
    p.add_argument("--private-output", type=Path, required=True)
    p.add_argument("--intsr-identities", type=Path, required=True,
                   help="Pinned intsr-test-identity.npz from the public v2 HF upload folder")
    args = p.parse_args()
    package(args.source_benchmark.resolve(), args.private_output.resolve(), args.intsr_identities.resolve())
