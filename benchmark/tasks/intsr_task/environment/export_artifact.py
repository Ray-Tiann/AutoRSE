#!/usr/bin/env python3
"""Export selected committed source, recoverable branches and prediction provenance."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from baseline_git import (RUNTIME_FILES, GitError, check_repo,
                          export_git, matching_run, repo_path, safe_bytes, tree_entries)


def checked_text(workspace, path):
    if not path.is_absolute(): path = workspace / path
    try: relative = path.resolve().relative_to(workspace)
    except ValueError: raise GitError('Extra configuration/helper must be inside the workspace') from None
    if path.is_symlink() or any(p.is_symlink() for p in path.parents if p == workspace or workspace in p.parents):
        raise GitError('Source symlinks are not exportable')
    if not path.is_file(): raise GitError('Required source file is missing')
    safe_bytes(relative.as_posix(), path.read_bytes())
    return path, relative


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=Path.cwd())
    parser.add_argument('--command', required=True, help='Exact reproduction commands, recorded as text')
    parser.add_argument('--config', type=Path, action='append', default=[])
    parser.add_argument('--helper', type=Path, action='append', default=[])
    parser.add_argument('--validation', default='', help='Public-validation selection rationale')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    workspace = args.workspace.resolve(); output = workspace / 'artifact'
    try:
        check_repo(workspace)
        run = matching_run(workspace)
        safe_bytes('README.md', (args.command + '\n' + args.validation).encode())
        if output.is_symlink() or (output.exists() and not args.overwrite):
            raise GitError('artifact already exists; use --overwrite to replace the previous selected artifact')
        with tempfile.TemporaryDirectory(prefix='.artifact-', dir=workspace) as temporary:
            stage = Path(temporary) / 'selected'; stage.mkdir()
            provenance = export_git(workspace, stage)
            repo = repo_path(workspace)
            entries = tree_entries(repo, 'HEAD')
            tracked = {entry[0] for entry in entries}
            for relative, mode, oid in entries:
                source = repo / relative
                target = stage / 'source/baseline' / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
                target.chmod(0o755 if mode == '100755' else 0o644)
            external = {}
            for name in RUNTIME_FILES:
                path = workspace / name
                if path.exists():
                    path, relative = checked_text(workspace, path)
                    target = stage / 'source' / relative
                    target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(path, target)
                    external[relative.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
            for path, kind in [(p, 'config') for p in args.config] + [(p, 'source') for p in args.helper]:
                path, relative = checked_text(workspace, path)
                if relative.parts[0] != 'baseline' or relative.relative_to('baseline').as_posix() not in tracked:
                    raise GitError('Put experimental helpers/config inside baseline and commit them before training')
                target = stage / kind / relative
                target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(path, target)
            provenance.update(evaluated_code_commit=run['code_commit'], selected_commit=provenance['commit'],
                              run=run, prediction=run['prediction'],
                              workspace_files_outside_baseline_git=external,
                              reproduction_command=args.command, public_validation=args.validation)
            manifest_text = json.dumps(provenance, indent=2) + '\n'
            safe_bytes('GIT_PROVENANCE.json', manifest_text.encode())
            (stage / 'GIT_PROVENANCE.json').write_text(manifest_text)
            (stage / 'README.md').write_text(
                '# Selected source artifact\n\n'
                f"Selected branch: `{provenance['branch']}`\n\n"
                f"Selected commit (including result records): `{provenance['commit']}`\n\n"
                f"Evaluated code commit: `{run['code_commit']}`\n\n"
                f"Prediction SHA256: `{run['prediction']['sha256']}`\n\n"
                'The source snapshot matches the selected clean commit. The Git bundle retains all '
                'reachable source branches and experiment records, without local Git configuration, '
                'hooks or reflogs. Weights, datasets and caches are excluded.\n\n'
                'Restore the selected repository (the bundle has no original remote credentials):\n\n'
                '```bash\ngit clone baseline.git.bundle restored-baseline\n'
                f"git -C restored-baseline switch {provenance['branch']}\n```\n\n"
                'Commands (run from source/, after installing the public data and restoring any '
                'additional config/ files):\n\n'
                f'```bash\n{args.command}\n```\n\n'
                f'Public-validation selection: {args.validation or "See committed experiment record."}\n\n'
                'GIT_PROVENANCE.json binds the evaluated code, actual recorded training command, '
                'runtime-file hashes and prediction bytes. Required static runtime files outside '
                'baseline Git are listed separately by SHA; keep experimental helpers/config inside baseline. '
                'Prediction files are collected separately and are the only files uploaded to the '
                'scoring Space. No hidden-test score is claimed by this artifact.\n')
            sums = [hashlib.sha256(p.read_bytes()).hexdigest() + '  ' + p.relative_to(stage).as_posix()
                    for p in sorted(stage.rglob('*')) if p.is_file()]
            (stage / 'SHA256SUMS').write_text('\n'.join(sums) + '\n')
            if output.exists(): shutil.rmtree(output)
            shutil.move(str(stage), output)
        print(f'ARTIFACT={output}')
    except (GitError, OSError, ValueError, KeyError) as error:
        parser.exit(2, 'ERROR: ' + str(error) + '\n')


if __name__ == '__main__': main()
