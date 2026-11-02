#!/usr/bin/env python3
"""Validate and package one collected six-task trial; no hidden-test scoring."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmark"))
from autorse.submissions import validate
from autorse.tasks import TASKS

RUN_LABEL = re.compile(r'[A-Za-z0-9][A-Za-z0-9_-]*\Z')


def check_source(path, runs_dir):
    """Reject redirected job, trial, artifact, and prediction paths."""
    for part in (path, *path.parents):
        if part == runs_dir:
            break
        if part.is_symlink():
            raise ValueError(f'Symlink prediction path is not accepted: {path}')
    if not path.is_file() or not path.resolve().is_relative_to(runs_dir):
        raise ValueError(f'Missing or unsafe prediction: {path}')


def package_predictions(runs_dir, run_label, seed, output=None, overwrite=False):
    if not isinstance(run_label, str) or not RUN_LABEL.fullmatch(run_label):
        raise ValueError('--run-label must contain only letters, digits, hyphens and underscores, starting with a letter or digit')
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError('--seed must be an integer')
    runs_dir = Path(runs_dir)
    if runs_dir.is_symlink() or not runs_dir.is_dir():
        raise ValueError('--runs-dir must be a regular directory')
    runs_dir = runs_dir.resolve()
    output = Path(output) if output is not None else Path(f'{run_label}-s{seed}.zip')
    output = output.absolute()
    if output.suffix.lower() != '.zip' or output.is_symlink():
        raise ValueError('--output must be a regular .zip path, not a symlink')
    if output.exists() and (not output.is_file() or not overwrite):
        raise ValueError('Output already exists; choose a new path or use --overwrite for a regular file')
    if not output.parent.is_dir():
        raise ValueError('Output parent directory does not exist')

    predictions = []
    for task, spec in TASKS.items():
        job = runs_dir / f'{run_label}-{task}-s{seed}'
        workspaces = list(job.glob('*/artifacts/workspace'))
        if len(workspaces) != 1:
            raise ValueError(f'{job}: expected exactly one collected trial, found {len(workspaces)}')
        source = workspaces[0] / 'datasets' / spec['dataset'] / 'working' / spec['file']
        check_source(source, runs_dir)
        if output.resolve() == source.resolve() or (output.exists() and output.samefile(source)):
            raise ValueError('Output cannot overwrite a prediction source')
        predictions.append((task, f"{task}/{spec['file']}", source))

    # A failed or incomplete task must never publish a partial six-task ZIP.
    for task, member, source in predictions:
        validate(task, source)

    descriptor, name = tempfile.mkstemp(prefix=f'.{output.name}.', suffix='.tmp', dir=output.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for task, member, source in predictions:
                check_source(source, runs_dir)
                archive.write(source, arcname=member)
        if output.is_symlink() or (output.exists() and not overwrite):
            raise ValueError('Output appeared during packaging; choose a new path or use --overwrite')
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs-dir', type=Path, default=ROOT / 'runs')
    parser.add_argument('--run-label', required=True)
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    try:
        print(package_predictions(args.runs_dir, args.run_label, args.seed, args.output, args.overwrite))
    except (ValueError, OSError, EOFError, zipfile.BadZipFile, zipfile.LargeZipFile) as error:
        parser.exit(2, f'Cannot package predictions: {error}\n')


if __name__ == '__main__':
    main()
