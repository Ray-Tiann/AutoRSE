#!/usr/bin/env python3
"""Export selected source/configuration and checksums, without weights or data."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
import tempfile

SOURCE_SUFFIXES = {'.py', '.sh', '.yaml', '.yml', '.toml', '.ini', '.json',
                   '.md', '.txt', '.c', '.cc', '.cpp', '.h', '.hpp', '.cu', '.cuh'}
EXCLUDED_PARTS = {'__pycache__', 'datasets', 'private', 'checkpoints', 'weights',
                  'logs', 'results', 'experiments', 'cache', 'caches'}


def source_text(path):
    """Reject common credential containers and binary files with text suffixes."""
    name = path.name.lower()
    if any(word in name for word in ('credential', 'secret')) or name.startswith('.env'):
        return False
    try:
        text = path.read_text(encoding='utf-8')
    except (UnicodeError, OSError):
        return False
    return '\0' not in text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=Path.cwd())
    parser.add_argument('--command', required=True, help='Exact training/inference commands, recorded as text')
    parser.add_argument('--config', type=Path, action='append', default=[],
                        help='Additional text configuration path within the workspace (repeatable)')
    parser.add_argument('--helper', type=Path, action='append', default=[],
                        help='Necessary source helper outside baseline, within the workspace (repeatable)')
    parser.add_argument('--validation', default='', help='Public validation result and selection rationale')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    source = workspace / 'baseline'
    output = workspace / 'artifact'
    if not source.is_dir() or source.is_symlink():
        parser.error('Expected a regular baseline source directory')
    if output.is_symlink() or (output.exists() and not args.overwrite):
        parser.error('artifact already exists; use --overwrite to replace the previous selected artifact')
    with tempfile.TemporaryDirectory(prefix='.artifact-', dir=workspace) as temporary:
        stage = Path(temporary) / 'selected'
        stage.mkdir()
        copied = 0
        for path in sorted(source.rglob('*')):
            relative = path.relative_to(source)
            if any(part.startswith('.') or part.lower() in EXCLUDED_PARTS for part in relative.parts):
                continue
            if path.is_symlink():
                parser.error(f'Source symlink is not exportable: {relative}')
            if not path.is_file() or not (path.suffix.lower() in SOURCE_SUFFIXES or path.name.startswith('LICENSE')):
                continue
            if not source_text(path):
                continue
            target = stage / 'source/baseline' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            copied += 1
        if not copied:
            parser.error('No source files selected')
        for name in ('run_baseline.sh', 'requirements.txt'):
            path = workspace / name
            if path.is_file() and not path.is_symlink():
                target = stage / 'source' / name
                shutil.copyfile(path, target)
        for config, kind in [(p, 'config') for p in args.config] + [(p, 'source') for p in args.helper]:
            path = config if config.is_absolute() else workspace / config
            if path.is_symlink() or not path.is_file():
                parser.error('Configuration must be a regular text file')
            try:
                relative = path.resolve().relative_to(workspace)
            except ValueError:
                parser.error('Configuration must be inside the workspace')
            if path.suffix.lower() not in SOURCE_SUFFIXES:
                parser.error('Weights and binary data cannot be copied as configuration')
            if not source_text(path) or any(part.lower() in EXCLUDED_PARTS for part in relative.parts):
                parser.error('Credential containers, binary data and generated artifacts are not exportable')
            target = stage / kind / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        (stage / 'README.md').write_text(
            '# Selected source artifact\n\n'
            'The selected source and necessary text configuration are included. '
            'Install the public task data separately; weights and data caches are excluded.\n\n'
            'Commands (run from source/, after restoring additional config/ files and installing the public data):\n\n'
            f'```bash\n{args.command}\n```\n\n'
            f'Public-validation selection: {args.validation or "Record before submitting."}\n\n'
            'Prediction files are collected separately. No hidden-test score is recorded here.\n')
        sums = []
        for path in sorted(stage.rglob('*')):
            if path.is_file():
                sums.append(hashlib.sha256(path.read_bytes()).hexdigest() + '  ' + path.relative_to(stage).as_posix())
        (stage / 'SHA256SUMS').write_text('\n'.join(sums) + '\n')
        if output.exists():
            shutil.rmtree(output)
        shutil.move(str(stage), output)
    print(f'ARTIFACT={output}')


if __name__ == '__main__':
    main()
