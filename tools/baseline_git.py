#!/usr/bin/env python3
"""Independent, source-only baseline Git repositories and experiment provenance."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import signal
import subprocess
import sys
import tempfile
import uuid

INITIAL_TAG = 'autorse-baseline'
MAX_FILE_BYTES = 4 * 1024**2
MAX_HISTORY_BYTES = 64 * 1024**2
MAX_BUNDLE_BYTES = 64 * 1024**2
SOURCE_SUFFIXES = {'.py', '.sh', '.yaml', '.yml', '.toml', '.ini', '.json', '.md', '.txt', '.c', '.cc', '.cpp', '.h', '.hpp', '.cu', '.cuh'}
BLOCKED = {'datasets', 'private', 'checkpoints', 'weights', 'logs', 'results', 'cache', 'caches', '.git', '.ssh', '.aws'}
RUNTIME_FILES = ('run_baseline.sh', 'train_baseline.sh', 'requirements.txt', 'baseline_git.py', 'export_artifact.py')
PREDICTIONS = {'hstu': 'datasets/beauty/working/submission.npz', 'tiger': 'datasets/beauty/working/submission.npz',
               'intsr': 'datasets/Amazon/working/submission.npz', 'wukong': 'datasets/movielenslatest/working/submission.csv.gz',
               'rankmixer': 'datasets/criteo5m/working/submission.npz', 'onetrans': 'datasets/kuaisar_v2/working/submission.csv'}
SECRET = re.compile(r'(?:\b(?:gh[pousr]_[A-Za-z0-9]{30,}|hf_[A-Za-z0-9]{25,}|sk-(?:proj-)?[A-Za-z0-9_-]{24,})\b|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)')
IGNORE = '''# AutoRSE baseline repository: code and small public-validation records only.
__pycache__/
*.pyc
.venv/
.env*
*credentials*
*secrets*
*.key
*.pem
*.pt
*.pth
*.ckpt
*.safetensors
*.bin
*.npy
*.npz
*.pkl
*.pickle
*.parquet
*.csv
*.gz
*.zip
*.tar
*.bundle
*.log
logs/
results/
checkpoints/
weights/
datasets/
cache/
caches/
.DS_Store
'''


class GitError(ValueError):
    pass


def git(repo, *arguments, input=None):
    environment = {k: v for k, v in os.environ.items() if not k.startswith('GIT_')}
    environment.update(GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL=os.devnull, GIT_TERMINAL_PROMPT='0')
    result = subprocess.run(['git', '-c', f'safe.directory={repo.resolve()}', '-c', 'core.hooksPath=/dev/null',
                             '-c', 'commit.gpgsign=false', '-C', str(repo), *arguments],
                            input=input, capture_output=True, env=environment)
    if result.returncode:
        # Git command output may contain local configuration. Do not reproduce it.
        raise GitError('Git operation failed: ' + arguments[0])
    return result.stdout


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, indent=2)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def safe_path(name):
    parts = PurePosixPath(name).parts
    if not parts or name.startswith('/') or '\\' in name or ':' in name or any(ord(c) < 32 for c in name):
        raise GitError('Unsafe source path')
    if '..' in parts or any(p.lower() in BLOCKED for p in parts):
        raise GitError('Data, generated outputs or private paths are not Git source: ' + name)
    if any(p.startswith('.') for p in parts) and name != '.gitignore':
        raise GitError('Hidden source path is not exportable: ' + name)
    leaf = parts[-1]
    if parts[0] == 'experiments' and not re.fullmatch(r'experiments/[0-9a-f]{32}\.json', name):
        raise GitError('experiments/ is reserved for generated run records; put source/config elsewhere: ' + name)
    if any(s in leaf.lower() for s in ('credential', 'secret')):
        raise GitError('Credential container is not exportable: ' + name)
    if name != '.gitignore' and PurePosixPath(name).suffix.lower() not in SOURCE_SUFFIXES and not leaf.startswith(('LICENSE', 'NOTICE')):
        raise GitError('Only source and text configuration may enter Git: ' + name)


def safe_bytes(name, data):
    safe_path(name)
    if len(data) > MAX_FILE_BYTES or b'\0' in data:
        raise GitError('Oversized or binary source file: ' + name)
    try:
        text = data.decode('utf-8')
    except UnicodeError:
        raise GitError('Source file is not UTF-8: ' + name) from None
    if SECRET.search(text):
        raise GitError('Credential-like value in source/history: ' + name)
    if name.startswith('experiments/'):
        try:
            record = json.loads(text)
            required = {'run_id', 'task', 'state', 'branch', 'code_commit', 'source_tree_sha256',
                        'runtime_files', 'command', 'seed', 'started_at', 'returncode', 'prediction',
                        'finished_at', 'public_validation', 'notes'}
            valid = (isinstance(record, dict) and required <= record.keys()
                     and record.keys() <= required | {'error'}
                     and name == f"experiments/{record['run_id']}.json"
                     and record['state'] in ('completed', 'failed')
                     and record['task'] in PREDICTIONS
                     and isinstance(record['command'], list)
                     and isinstance(record['runtime_files'], dict)
                     and isinstance(record['notes'], str)
                     and isinstance(record['public_validation'], str))
        except (ValueError, KeyError, TypeError):
            valid = False
        if not valid:
            raise GitError('experiments/ only accepts generated result receipt JSON: ' + name)


def repo_path(workspace):
    workspace = Path(workspace).resolve()
    repo = workspace / 'baseline'
    if repo.is_symlink() or not repo.is_dir():
        raise GitError('Expected a regular baseline/ directory')
    marker = repo / '.git'
    if marker.is_symlink() or not marker.is_dir():
        raise GitError('Initialize baseline/.git with baseline_git.py init; external Git links are not accepted')
    if Path(git(repo, 'rev-parse', '--show-toplevel').decode().strip()).resolve() != repo:
        raise GitError('Git repository must belong exactly to baseline/, not the outer checkout')
    return repo


def working_files(repo):
    names = git(repo, 'ls-files', '--cached', '--others', '--exclude-standard', '-z').split(b'\0')
    for raw in sorted(set(filter(None, names))):
        name = raw.decode('utf-8')
        safe_path(name)
        path = repo / name
        if any(p.is_symlink() for p in (path, *path.parents) if p == repo or repo in p.parents):
            raise GitError('Source symlinks are not accepted: ' + name)
        if path.exists():
            if not path.is_file():
                raise GitError('Nested repositories/directories are not source files')
            if path.stat().st_size > MAX_FILE_BYTES:
                raise GitError('Source file exceeds limit: ' + name)
            safe_bytes(name, path.read_bytes())


def init_repo(workspace):
    workspace = Path(workspace).resolve()
    repo = workspace / 'baseline'
    if repo.is_symlink() or not repo.is_dir():
        raise GitError('Expected a regular baseline/ directory')
    if (repo / '.git').exists() or (repo / '.git').is_symlink():
        return repo_state(workspace)  # Never reset or auto-commit an existing repository.
    git(repo, 'init', '--initial-branch=baseline')
    git(repo, 'config', '--local', 'user.name', 'AutoRSE Research Agent')
    git(repo, 'config', '--local', 'user.email', 'autorse-agent@example.invalid')
    git(repo, 'config', '--local', 'core.autocrlf', 'false')
    ignore = repo / '.gitignore'
    old = ignore.read_text() if ignore.exists() else ''
    ignore.write_text(old.rstrip() + '\n' + IGNORE)
    working_files(repo)
    git(repo, 'add', '--all', '--', '.')
    git(repo, 'commit', '-m', 'Initial AutoRSE baseline source snapshot')
    git(repo, 'tag', INITIAL_TAG)
    return repo_state(workspace)


def repo_state(workspace):
    repo = repo_path(workspace)
    initial = git(repo, 'rev-parse', f'refs/tags/{INITIAL_TAG}^{{commit}}').decode().strip()
    baseline = git(repo, 'rev-parse', 'refs/heads/baseline').decode().strip()
    if baseline != initial:
        raise GitError('The initial baseline branch is protected; use exp/<name> branches')
    branch = git(repo, 'branch', '--show-current').decode().strip()
    if branch != 'baseline' and not branch.startswith('exp/'):
        raise GitError('Select baseline or an exp/<name> branch; detached HEAD is not a final selection')
    branches = {}
    for line in git(repo, 'for-each-ref', '--format=%(refname:short) %(objectname)', 'refs/heads').decode().splitlines():
        name, commit = line.rsplit(' ', 1); branches[name] = commit
    return {'branch': branch, 'commit': git(repo, 'rev-parse', 'HEAD').decode().strip(),
            'initial_commit': initial, 'branches': branches}


def check_repo(workspace):
    state = repo_state(workspace); repo = repo_path(workspace)
    working_files(repo)
    if git(repo, 'status', '--porcelain', '--untracked-files=all').strip():
        raise GitError('Baseline has uncommitted source; create an experiment branch and commit before training/export')
    return state


def new_branch(workspace, name, source):
    check_repo(workspace); repo = repo_path(workspace)
    if not re.fullmatch(r'exp/[A-Za-z0-9][A-Za-z0-9._/-]{0,100}', name):
        raise GitError('Use a new branch name such as exp/001-learning-rate')
    git(repo, 'check-ref-format', '--branch', name)
    git(repo, 'switch', '-c', name, source)
    return check_repo(workspace)


def commit_source(workspace, message):
    state = repo_state(workspace); repo = repo_path(workspace)
    if state['branch'] == 'baseline':
        raise GitError('Do not commit modifications on baseline; start an exp/<name> branch')
    if not message.strip() or SECRET.search(message):
        raise GitError('Provide a non-secret experiment commit message')
    working_files(repo)
    git(repo, 'add', '--all', '--', '.')
    if git(repo, 'diff', '--cached', '--name-only').strip():
        git(repo, 'commit', '-m', message)
    return check_repo(workspace)


def tree_entries(repo, ref):
    result = []
    for raw in filter(None, git(repo, 'ls-tree', '-r', '-z', ref).split(b'\0')):
        meta, name = raw.split(b'\t', 1)
        mode, kind, oid = meta.decode().split()
        path = name.decode('utf-8')
        safe_path(path)
        if kind != 'blob' or mode not in ('100644', '100755'):
            raise GitError('Git submodules and symlinks are not exportable')
        result.append((path, mode, oid))
    return result


def source_hash(workspace, ref='HEAD'):
    repo = repo_path(workspace)
    entries = tree_entries(repo, ref)
    for path, _, oid in entries:
        if path.startswith('experiments/'):
            safe_bytes(path, git(repo, 'cat-file', 'blob', oid))
    text = '\n'.join(f'{p}\0{mode}\0{oid}' for p, mode, oid in entries if not p.startswith('experiments/'))
    return hashlib.sha256(text.encode()).hexdigest()


def runtime_hashes(workspace):
    workspace = Path(workspace).resolve(); result = {}
    for name in RUNTIME_FILES:
        path = workspace / name
        if path.exists():
            if path.is_symlink() or not path.is_file():
                raise GitError('Runtime files must be regular text files')
            safe_bytes(name, path.read_bytes())
            result[name] = sha(path)
    return result


def run_experiment(workspace, task, command):
    workspace = Path(workspace).resolve()
    init_repo(workspace); state = check_repo(workspace)
    if not command:
        raise GitError('Supply the training command after --')
    relative = PREDICTIONS[task]; prediction = workspace / relative
    before = prediction.stat().st_mtime_ns if prediction.is_file() else None
    code_hash = source_hash(workspace); runtime = runtime_hashes(workspace)
    store = workspace / '.autorse-runs'; store.mkdir(exist_ok=True)
    lock = store / 'ACTIVE'
    run_id = uuid.uuid4().hex
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise GitError('Another run or an interrupted run owns .autorse-runs/ACTIVE; inspect it before removing this lock') from None
    with os.fdopen(descriptor, 'w') as f: json.dump({'run_id': run_id, 'pid': os.getpid()}, f)
    receipt = {'run_id': run_id, 'task': task, 'state': 'running', 'branch': state['branch'],
               'code_commit': state['commit'], 'source_tree_sha256': code_hash,
               'runtime_files': runtime, 'command': command,
               'seed': os.environ.get('RESEARCH_SEED', '2024'),
               'started_at': datetime.now(timezone.utc).isoformat(), 'returncode': None, 'prediction': None}
    process = None; interrupted = []
    def stop(signum, frame):
        interrupted.append(signum)
        if process is not None:
            try: os.killpg(process.pid, signum)
            except ProcessLookupError: pass
    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        atomic_json(store / 'latest.json', receipt)
        process = subprocess.Popen(command, cwd=workspace, start_new_session=True)
        if interrupted:
            try: os.killpg(process.pid, interrupted[-1])
            except ProcessLookupError: pass
        rc = process.wait()
        receipt['returncode'] = rc
        if (rc == 0 and not interrupted and prediction.is_file() and not prediction.is_symlink()
                and prediction.resolve().is_relative_to(workspace)
                and prediction.stat().st_mtime_ns != before):
            receipt['prediction'] = {'path': relative, 'sha256': sha(prediction), 'size': prediction.stat().st_size}
        receipt['state'] = 'completed' if receipt['prediction'] is not None else 'failed'
    except Exception:
        receipt['state'] = 'failed'
        receipt['error'] = 'Training command did not complete'
        raise
    finally:
        for sig, handler in previous.items(): signal.signal(sig, handler)
        receipt['finished_at'] = datetime.now(timezone.utc).isoformat()
        try:
            atomic_json(store / f'run-{run_id}.json', receipt); atomic_json(store / 'latest.json', receipt)
        finally:
            lock.unlink(missing_ok=True)
    return receipt


def record_result(workspace, validation, notes):
    workspace = Path(workspace).resolve(); state = check_repo(workspace)
    if state['branch'] == 'baseline':
        raise GitError('Keep baseline immutable; use exp/000-baseline to record a reference experiment')
    receipt = json.loads((workspace / '.autorse-runs/latest.json').read_text())
    if receipt['state'] == 'running' or receipt['branch'] != state['branch'] or receipt['source_tree_sha256'] != source_hash(workspace):
        raise GitError('Current branch/source does not match the finished run')
    text = json.dumps({**receipt, 'public_validation': validation, 'notes': notes}, indent=2) + '\n'
    safe_bytes('experiments/' + receipt['run_id'] + '.json', text.encode())
    target = workspace / 'baseline/experiments' / (receipt['run_id'] + '.json')
    target.parent.mkdir(exist_ok=True); target.write_text(text)
    return commit_source(workspace, 'Record experiment ' + receipt['run_id'])


def matching_run(workspace):
    workspace = Path(workspace).resolve(); state = check_repo(workspace)
    code_hash = source_hash(workspace); runtime = runtime_hashes(workspace)
    for path in sorted((workspace / '.autorse-runs').glob('run-*.json'), key=lambda p: p.stat().st_mtime_ns, reverse=True):
        row = json.loads(path.read_text())
        pred = row.get('prediction')
        if row.get('returncode') != 0 or row.get('state') != 'completed' or not pred:
            continue
        if row['branch'] != state['branch'] or row['source_tree_sha256'] != code_hash or row['runtime_files'] != runtime:
            continue
        prediction = workspace / pred['path']
        if prediction.is_symlink() or not prediction.is_file() or not prediction.resolve().is_relative_to(workspace):
            continue
        if prediction.stat().st_size != pred['size'] or sha(prediction) != pred['sha256']:
            continue
        try:
            git(repo_path(workspace), 'merge-base', '--is-ancestor', row['code_commit'], state['commit'])
        except GitError:
            continue
        return row
    raise GitError('No successful recorded run matches selected source/runtime and prediction bytes; restore the saved best prediction or run the selected branch')


def audit_history(workspace):
    repo = repo_path(workspace)
    # Bundles also contain annotated tags and ref names. Inspect them as well as
    # commit trees; refs directly targeting blobs/trees are not source branches.
    tags = []
    for line in git(repo, 'for-each-ref', '--format=%(objecttype) %(objectname) %(refname)').decode().splitlines():
        kind, oid, ref = line.split(' ', 2)
        safe_bytes('history-ref.txt', ref.encode())
        if kind == 'tag': tags.append(oid)
        elif kind != 'commit': raise GitError('Git refs must target commits or annotated tags')
    seen_tags = set()
    while tags:
        oid = tags.pop()
        if oid in seen_tags: continue
        seen_tags.add(oid)
        data = git(repo, 'cat-file', 'tag', oid)
        safe_bytes('history-tag.txt', data)
        lines = data.decode().splitlines()
        target, kind = lines[0].removeprefix('object '), lines[1].removeprefix('type ')
        if kind == 'tag': tags.append(target)
        elif kind != 'commit': raise GitError('Annotated tags must point to commit history')
    blobs = {}
    for commit in git(repo, 'rev-list', '--all').decode().splitlines():
        safe_bytes('history-commit.txt', git(repo, 'cat-file', 'commit', commit))
        for name, mode, oid in tree_entries(repo, commit):
            blobs.setdefault(oid, name)
    ids = list(blobs)
    query = ('\n'.join(ids) + '\n').encode()
    summaries = git(repo, 'cat-file', '--batch-check', input=query).decode().splitlines()
    sizes = {}
    for line in summaries:
        oid, kind, size = line.split(); size = int(size)
        if kind != 'blob' or size > MAX_FILE_BYTES:
            raise GitError('Historical source file exceeds size limit')
        sizes[oid] = size
    if sum(sizes.values()) > MAX_HISTORY_BYTES:
        raise GitError('Reachable source history exceeds the export limit')
    payload = git(repo, 'cat-file', '--batch', input=query)
    offset = 0
    for oid in ids:
        end = payload.index(b'\n', offset)
        header = payload[offset:end].decode().split(); size = int(header[2]); offset = end + 1
        safe_bytes(blobs[oid], payload[offset:offset + size]); offset += size + 1


def export_git(workspace, output):
    state = check_repo(workspace); audit_history(workspace)
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    bundle = output / 'baseline.git.bundle'
    git(repo_path(workspace), 'bundle', 'create', str(bundle.resolve()), '--all')
    if bundle.stat().st_size > MAX_BUNDLE_BYTES:
        bundle.unlink(); raise GitError('Git bundle exceeds the export limit')
    git(repo_path(workspace), 'bundle', 'verify', str(bundle.resolve()))
    provenance = {**state, 'source_tree_sha256': source_hash(workspace),
                  'bundle_sha256': sha(bundle), 'history': 'all reachable refs; source and experiment records only',
                  'git_config_hooks_reflogs_exported': False, 'working_tree_clean': True}
    atomic_json(output / 'GIT_PROVENANCE.json', provenance)
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    for action in ('init', 'new', 'commit', 'check', 'run', 'record'):
        p = sub.add_parser(action); p.add_argument('--workspace', type=Path, default=Path.cwd())
        if action == 'new': p.add_argument('name'); p.add_argument('--from', dest='source', default='baseline')
        if action == 'commit': p.add_argument('-m', '--message', required=True)
        if action == 'run': p.add_argument('--task', choices=PREDICTIONS, required=True); p.add_argument('command', nargs=argparse.REMAINDER)
        if action == 'record': p.add_argument('--validation', required=True); p.add_argument('--notes', default='')
    a = parser.parse_args()
    try:
        if a.action == 'init': result = init_repo(a.workspace)
        elif a.action == 'new': result = new_branch(a.workspace, a.name, a.source)
        elif a.action == 'commit': result = commit_source(a.workspace, a.message)
        elif a.action == 'check': result = check_repo(a.workspace)
        elif a.action == 'record': result = record_result(a.workspace, a.validation, a.notes)
        else:
            command = a.command[1:] if a.command[:1] == ['--'] else a.command
            result = run_experiment(a.workspace, a.task, command)
            print(json.dumps(result, indent=2))
            rc = result['returncode']
            return (rc if rc >= 0 else 128 - rc) if rc else (0 if result['state'] == 'completed' else 2)
        print(json.dumps(result, indent=2)); return 0
    except (GitError, OSError, ValueError, KeyError) as error:
        print('ERROR: ' + str(error), file=sys.stderr); return 2


if __name__ == '__main__':
    raise SystemExit(main())
