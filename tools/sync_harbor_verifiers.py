#!/usr/bin/env python3
"""Refresh label-free Harbor verifier copies from the public format validator."""
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
TASKS = ('hstu', 'tiger', 'intsr', 'wukong', 'rankmixer', 'onetrans')


def sync():
    for task in TASKS:
        target = ROOT / f'benchmark/tasks/{task}_task/tests/autorse'
        target.mkdir(parents=True, exist_ok=True)
        for name in ('__init__.py', 'tasks.py', 'submissions.py'):
            shutil.copyfile(ROOT / 'autorse' / name, target / name)


if __name__ == '__main__':
    sync()
