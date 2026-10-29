#!/usr/bin/env python3
"""Train a public-data baseline locally or in one GPU-enabled Docker container."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]
DATASETS = {'hstu': 'beauty', 'tiger': 'beauty', 'intsr': 'Amazon',
            'wukong': 'movielenslatest', 'rankmixer': 'criteo5m', 'onetrans': 'kuaisar_v2'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=DATASETS, required=True)
    parser.add_argument('--local', action='store_true', help='Use the current Python/CUDA environment')
    parser.add_argument('--image', default='autorse-agent:latest')
    parser.add_argument('--gpu', default='0', help='One NVIDIA GPU device ID')
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('training_args', nargs=argparse.REMAINDER, help='Extra wrapper options after --')
    args = parser.parse_args()
    if not args.gpu or any(c in args.gpu for c in ', \t\n'):
        parser.error('--gpu must name exactly one GPU')
    env_path = ROOT / f'benchmark/tasks/{args.task}_task/environment'
    extra = args.training_args[1:] if args.training_args[:1] == ['--'] else args.training_args
    wrapper = ['bash', 'run_baseline.sh', DATASETS[args.task], *extra]
    if args.local:
        command = wrapper
        environment = {**os.environ, 'CUDA_VISIBLE_DEVICES': args.gpu, 'RESEARCH_SEED': str(args.seed)}
    else:
        command = ['docker', 'run', '--rm', '--gpus', f'device={args.gpu}', '--ipc=host',
                   '--mount', f'type=bind,source={env_path},target=/workspace',
                   '--workdir', '/workspace', '--env', f'RESEARCH_SEED={args.seed}',
                   args.image, *wrapper]
        environment = os.environ.copy()
    if args.dry_run:
        print(json.dumps({'task': args.task, 'cwd': str(env_path), 'command': shlex.join(command),
                          'seed': args.seed, 'gpu': args.gpu, 'training_started': False}))
        return
    if not (env_path / f'datasets/{DATASETS[args.task]}/input').is_dir():
        parser.error('Public data is missing; run tools/install_public_data.py first')
    subprocess.run(command, cwd=env_path, env=environment, check=True)


if __name__ == '__main__':
    main()
