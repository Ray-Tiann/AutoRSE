#!/usr/bin/env python3
"""Launch a Harbor research agent; local verifier reports submission format only."""
from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / 'benchmark'
TASKS = ('hstu', 'tiger', 'intsr', 'wukong', 'rankmixer', 'onetrans')
ADAPTERS = {'codex': 'tools.persistent_codex:PersistentCodex',
            'claude-code': 'tools.persistent_claude_code:PersistentClaudeCode',
            'terminus-2': 'tools.persistent_terminus_2:PersistentTerminus2'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=TASKS, required=True)
    parser.add_argument('--agent', choices=ADAPTERS, required=True)
    parser.add_argument('--model', required=True, help='Model identifier accepted by the selected Harbor harness')
    parser.add_argument('--api-env-file', type=Path, help='Private credentials outside the checkout')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--minimum-runtime-sec', type=int, default=43200)
    parser.add_argument('--finalization-grace-sec', type=int, default=300)
    parser.add_argument('--no-persistence', action='store_true')
    parser.add_argument('--jobs-dir', type=Path, default=ROOT / 'runs')
    parser.add_argument('--job-name', help='Optional Harbor job name; choose a new name for each run')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('harbor_args', nargs=argparse.REMAINDER, help='Additional Harbor flags after --')
    args = parser.parse_args()
    if not args.gpu or any(c in args.gpu for c in ', \t\n'):
        parser.error('--gpu must name exactly one GPU')
    if min(args.minimum_runtime_sec, args.finalization_grace_sec) <= 0:
        parser.error('Research and finalization durations must be positive')
    if args.minimum_runtime_sec + args.finalization_grace_sec > 43500:
        parser.error('Combined durations must fit the 43500-second task research/finalization window')
    adapter = args.agent if args.no_persistence else ADAPTERS[args.agent]
    command = ['harbor', 'run', '--path', str(BENCHMARK / f'tasks/{args.task}_task'),
               '--agent', adapter, '--model', args.model, '--env', 'docker',
               '--n-concurrent', '1', '--n-attempts', '1', '--jobs-dir', str(args.jobs_dir.resolve()), '--yes',
               '--agent-env', f'RESEARCH_SEED={args.seed}']
    if args.job_name:
        command += ['--job-name', args.job_name]
    if args.api_env_file:
        command += ['--env-file', str(args.api_env_file.expanduser().resolve())]
    if not args.no_persistence:
        for key, value in [('minimum_runtime_sec', args.minimum_runtime_sec),
                           ('finalization_grace_sec', args.finalization_grace_sec),
                           ('research_seed', args.seed)]:
            command += ['--agent-kwarg', f'{key}={value}']
    if args.agent == 'claude-code':
        command += ['--agent-kwarg', 'disallowed_tools=WebSearch,WebFetch']
    extra = args.harbor_args[1:] if args.harbor_args[:1] == ['--'] else args.harbor_args
    command += extra
    environment = {**os.environ, 'GPU_DEVICE_0': args.gpu,
                   'LITELLM_LOCAL_MODEL_COST_MAP': 'True',
                   'PYTHONPATH': str(BENCHMARK) + os.pathsep + os.environ.get('PYTHONPATH', '')}
    if args.dry_run:
        print(json.dumps({'cwd': str(BENCHMARK), 'command': shlex.join(command),
                          'gpu': args.gpu, 'research_seconds': args.minimum_runtime_sec,
                          'persistence': not args.no_persistence,
                          'local_verifier': 'format_only_not_official_score', 'agent_started': False}))
        return
    try:
        installed = version('harbor')
    except PackageNotFoundError:
        parser.error('Install the optional Harbor dependency from requirements-harbor.txt')
    if installed != '0.18.0':
        parser.error(f'This integration targets Harbor 0.18.0; installed version is {installed}')
    if args.api_env_file and not args.api_env_file.expanduser().is_file():
        parser.error('--api-env-file does not exist')
    from sync_harbor_verifiers import sync
    sync()
    subprocess.run(command, cwd=BENCHMARK, env=environment, check=True)
    print('Harbor local reward is format_valid only. Submit predictions to the hosted evaluator for official scoring.')


if __name__ == '__main__':
    main()
