"""Exercise baseline research provenance without Docker, GPUs, or ML training."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / 'tools/baseline_git.py'
EXPORTER = ROOT / 'tools/export_artifact.py'
PREDICTION = Path('datasets/beauty/working/submission.npz')


class BaselineGitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location('baseline_git_under_test', HELPER)
        cls.helper = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cls.helper
        spec.loader.exec_module(cls.helper)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.workspace = Path(self.temporary.name) / 'workspace'
        self.workspace.mkdir()
        self.baseline = self.workspace / 'baseline'
        self.baseline.mkdir()
        (self.baseline / 'train.py').write_text('print("initial baseline")\n')
        (self.workspace / 'requirements.txt').write_text('numpy==1.26.4\n')
        (self.workspace / 'run_baseline.sh').write_text('bash train_baseline.sh "$@"\n')
        # Copying the source bytes makes the fake prediction observably depend
        # on the selected code, without claiming to create an ML-valid NPZ.
        (self.workspace / 'train_baseline.sh').write_text(
            '#!/usr/bin/env bash\nset -eu\n'
            'mkdir -p datasets/beauty/working\n'
            'cat baseline/train.py > datasets/beauty/working/submission.npz\n')
        self.environment = {key: value for key, value in os.environ.items()
                            if not key.startswith('GIT_')}
        self.environment.update({'GIT_CONFIG_NOSYSTEM': '1',
                                 'GIT_CONFIG_GLOBAL': os.devnull,
                                 'GIT_TERMINAL_PROMPT': '0'})

    def git(self, *arguments, cwd=None, check=True):
        return subprocess.run(
            ['git', '-C', str(cwd or self.baseline), *arguments],
            env=self.environment, capture_output=True, text=True, check=check)

    def cli(self, command, *arguments, check=True):
        return subprocess.run(
            [sys.executable, str(HELPER), command,
             '--workspace', str(self.workspace), *arguments],
            cwd=self.workspace, env=self.environment,
            capture_output=True, text=True, check=check)

    def initialize(self):
        self.cli('init')
        return self.git('rev-parse', 'HEAD').stdout.strip()

    def new_experiment(self, name='exp/001', source='print("experiment one")\n'):
        self.cli('new', name, '--from', 'baseline')
        (self.baseline / 'train.py').write_text(source)
        self.cli('commit', '-m', f'Implement {name}')
        return self.git('rev-parse', 'HEAD').stdout.strip()

    def run_experiment(self):
        self.cli('run', '--task', 'hstu', '--',
                 'bash', 'train_baseline.sh', 'beauty')
        return json.loads((self.workspace / '.autorse-runs/latest.json').read_text())

    def record_experiment(self):
        self.cli('record', '--validation', 'public validation NDCG@10=0.1',
                 '--notes', 'Selected using public validation only.')

    def successful_experiment(self):
        self.initialize()
        code_commit = self.new_experiment()
        run = self.run_experiment()
        self.record_experiment()
        return code_commit, run

    def export_cli(self, check=True, extra=()):
        return subprocess.run(
            [sys.executable, str(EXPORTER), '--workspace', str(self.workspace),
             '--command', 'bash run_baseline.sh beauty',
             '--validation', 'public validation NDCG@10=0.1', *extra],
            cwd=self.workspace, env=self.environment,
            capture_output=True, text=True, check=check)

    def test_initialization_is_idempotent_and_does_not_change_outer_repository(self):
        self.git('init', cwd=self.workspace)
        self.git('config', 'user.name', 'Outer fixture', cwd=self.workspace)
        self.git('config', 'user.email', 'outer@example.invalid', cwd=self.workspace)
        self.git('add', '.', cwd=self.workspace)
        self.git('commit', '-m', 'Outer source snapshot', cwd=self.workspace)
        outer_head = self.git('rev-parse', 'HEAD', cwd=self.workspace).stdout
        outer_files = self.git('ls-files', '-s', cwd=self.workspace).stdout

        first = self.initialize()
        self.cli('init')
        self.assertEqual(first, self.git('rev-parse', 'HEAD').stdout.strip())
        self.assertEqual('baseline', self.git('branch', '--show-current').stdout.strip())
        self.assertEqual(self.baseline.resolve(),
                         Path(self.git('rev-parse', '--show-toplevel').stdout.strip()).resolve())
        self.assertEqual(outer_head, self.git('rev-parse', 'HEAD', cwd=self.workspace).stdout)
        self.assertEqual(outer_files, self.git('ls-files', '-s', cwd=self.workspace).stdout)
        state = self.helper.repo_state(self.workspace)
        self.assertEqual(state['initial_commit'], first)
        self.assertEqual(state['commit'], first)
        self.assertEqual(state['branch'], 'baseline')
        self.assertIn('baseline', state['branches'])
        (self.baseline / 'train.py').write_text('print("edited after initialization")\n')
        self.cli('init')
        self.assertEqual(first, self.git('rev-parse', 'HEAD').stdout.strip())
        self.assertIn('train.py', self.git('diff', '--name-only').stdout)

    def test_each_experiment_records_the_committed_code_and_prediction(self):
        initial = self.initialize()
        code_commit = self.new_experiment()
        self.assertNotEqual(initial, code_commit)
        run = self.run_experiment()
        prediction = self.workspace / PREDICTION
        self.assertEqual(run['branch'], 'exp/001')
        self.assertEqual(run['code_commit'], code_commit)
        self.assertEqual(run['returncode'], 0)
        self.assertEqual(run['prediction']['path'], PREDICTION.as_posix())
        self.assertEqual(run['prediction']['sha256'],
                         hashlib.sha256(prediction.read_bytes()).hexdigest())
        self.assertEqual(run['prediction']['size'], prediction.stat().st_size)
        self.record_experiment()
        result_commit = self.git('rev-parse', 'HEAD').stdout.strip()
        self.assertNotEqual(result_commit, code_commit)
        self.git('merge-base', '--is-ancestor', code_commit, result_commit)
        self.assertEqual('', self.git('status', '--porcelain').stdout)
        self.cli('check')

    def test_experimental_commits_are_refused_on_the_original_baseline_branch(self):
        initial = self.initialize()
        (self.baseline / 'train.py').write_text('print("must use an experiment branch")\n')
        rejected = self.cli('commit', '-m', 'Improper baseline modification', check=False)
        self.assertNotEqual(rejected.returncode, 0)
        self.assertEqual(initial, self.git('rev-parse', 'HEAD').stdout.strip())

    def test_training_requires_committed_tracked_and_untracked_source(self):
        self.initialize()
        self.new_experiment()
        for name in ('train.py', 'new_helper.py'):
            with self.subTest(path=name):
                path = self.baseline / name
                previous = path.read_bytes() if path.exists() else None
                path.write_text('print("uncommitted change")\n')
                rejected = self.cli('run', '--task', 'hstu', '--',
                                    'bash', 'train_baseline.sh', 'beauty', check=False)
                self.assertNotEqual(rejected.returncode, 0)
                self.assertFalse((self.workspace / PREDICTION).exists())
                if previous is None:
                    path.unlink()
                else:
                    path.write_bytes(previous)

    def test_weights_are_ignored_and_are_not_added_by_experiment_commits(self):
        self.initialize()
        (self.baseline / 'checkpoint.pt').write_bytes(b'not source')
        (self.baseline / 'features.npz').write_bytes(b'not source')
        self.new_experiment()
        tracked = self.git('ls-files').stdout.splitlines()
        self.assertNotIn('checkpoint.pt', tracked)
        self.assertNotIn('features.npz', tracked)
        self.assertEqual('', self.git('status', '--porcelain').stdout)

    def test_export_restores_all_experiment_branches_and_the_selected_head(self):
        first_code, _ = self.successful_experiment()
        second_code = self.new_experiment('exp/002', 'print("experiment two")\n')
        self.run_experiment()
        self.record_experiment()
        selected = self.git('rev-parse', 'HEAD').stdout.strip()
        self.git('remote', 'add', 'private-origin',
                 'https://never-export-this-token@example.invalid/private.git')
        output = self.workspace / 'git-export'
        output.mkdir()
        self.helper.export_git(self.workspace, output)
        bundle = output / 'baseline.git.bundle'
        self.assertTrue(bundle.is_file())
        manifest = json.loads((output / 'GIT_PROVENANCE.json').read_text())
        serialized = json.dumps(manifest)
        for value in ('baseline', 'exp/001', 'exp/002', selected):
            self.assertIn(value, serialized)

        restored = Path(self.temporary.name) / 'restored'
        self.git('clone', str(bundle), str(restored), cwd=self.workspace)
        self.assertEqual(selected, self.git('rev-parse', 'HEAD', cwd=restored).stdout.strip())
        self.assertEqual('exp/002', self.git('branch', '--show-current', cwd=restored).stdout.strip())
        self.assertEqual((self.baseline / 'train.py').read_bytes(),
                         (restored / 'train.py').read_bytes())
        for branch in ('baseline', 'exp/001', 'exp/002'):
            self.git('show-ref', '--verify', f'refs/remotes/origin/{branch}', cwd=restored)
        for commit in (first_code, second_code):
            self.git('cat-file', '-e', f'{commit}^{{commit}}', cwd=restored)
        self.assertNotIn('never-export-this-token', (restored / '.git/config').read_text())
        self.git('fsck', '--full', cwd=restored)

    def test_export_rejects_unsafe_files_even_when_deleted_from_current_source(self):
        for index, (filename, content) in enumerate((
                ('old_weights.pt', b'old checkpoint'),
                ('credentials.json', b'{"token":"private"}'),
                ('old_access.py', b'TOKEN = "sk-' + b'A' * 32 + b'"\n'))):
            with self.subTest(path=filename):
                # Independent histories ensure a prior rejected artifact can
                # never mask the other file's history audit.
                self.workspace = Path(self.temporary.name) / f'unsafe-case-{index}'
                self.baseline = self.workspace / 'baseline'
                self.baseline.mkdir(parents=True)
                (self.baseline / 'train.py').write_text('print("safe source")\n')
                self.initialize()
                self.cli('new', 'exp/unsafe', '--from', 'baseline')
                path = self.baseline / filename
                path.write_bytes(content)
                # Deliberately bypass the helper to model an agent using raw
                # Git. Looking only at the current tree would miss this file.
                self.git('add', '-f', '--', filename)
                self.git('commit', '-m', 'Accidentally record excluded artifact')
                self.git('rm', '--', filename)
                self.git('commit', '-m', 'Remove excluded artifact')
                output = self.workspace / 'unsafe-export'
                with self.assertRaises((ValueError, RuntimeError)) as rejected:
                    self.helper.export_git(self.workspace, output)
                self.assertIn(filename, str(rejected.exception))
                self.assertFalse((output / 'baseline.git.bundle').exists())

    def test_export_bounds_the_size_of_history_removed_from_the_current_tree(self):
        self.initialize()
        self.new_experiment(source='print("historical source")\n' * 1024)
        (self.baseline / 'train.py').write_text('print("small selected source")\n')
        self.cli('commit', '-m', 'Replace large experiment source')
        output = self.workspace / 'bounded-export'
        with patch.object(self.helper, 'MAX_HISTORY_BYTES', 4096):
            with self.assertRaisesRegex((ValueError, RuntimeError), 'history'):
                self.helper.export_git(self.workspace, output)
        self.assertFalse((output / 'baseline.git.bundle').exists())

    def test_export_rejects_refs_pointing_directly_to_uninspected_blobs(self):
        self.initialize()
        payload = Path(self.temporary.name) / 'untracked-object.txt'
        payload.write_text('sk-' + 'A' * 32)
        oid = self.git('hash-object', '-w', str(payload)).stdout.strip()
        self.git('update-ref', 'refs/tags/direct-blob', oid)
        output = self.workspace / 'blob-ref-export'
        with self.assertRaises((ValueError, RuntimeError)):
            self.helper.export_git(self.workspace, output)
        self.assertFalse((output / 'baseline.git.bundle').exists())

    def test_export_rejects_credentials_in_annotated_tag_messages(self):
        self.initialize()
        self.git('tag', '-a', 'external-note', '-m', 'sk-' + 'A' * 32)
        output = self.workspace / 'tag-export'
        with self.assertRaises((ValueError, RuntimeError)):
            self.helper.export_git(self.workspace, output)
        self.assertFalse((output / 'baseline.git.bundle').exists())

    def test_experiment_records_cannot_hide_changed_training_configuration(self):
        self.successful_experiment()
        config = self.baseline / 'experiments/config.yaml'
        config.write_text('learning_rate: 10.0\n')
        # Raw Git must not bypass the reserved receipt directory's policy or
        # hide an untrained configuration from the source hash.
        self.git('add', '--', 'experiments/config.yaml')
        self.git('commit', '-m', 'Add untrained experiment configuration')
        self.assertNotEqual(self.cli('check', check=False).returncode, 0)
        self.assertNotEqual(self.export_cli(check=False).returncode, 0)
        self.assertFalse((self.workspace / 'artifact').exists())

    def test_reserved_receipt_filenames_require_generated_receipt_content(self):
        self.successful_experiment()
        relative = 'experiments/' + 'f' * 32 + '.json'
        (self.baseline / relative).write_text('{"learning_rate": 10.0}\n')
        self.git('add', '--', relative)
        self.git('commit', '-m', 'Add configuration disguised as result metadata')
        self.assertNotEqual(self.cli('check', check=False).returncode, 0)
        self.assertNotEqual(self.export_cli(check=False).returncode, 0)
        self.assertFalse((self.workspace / 'artifact').exists())

    def test_failed_preflight_does_not_leave_an_active_run_lock(self):
        self.initialize()
        self.new_experiment()
        script = self.workspace / 'train_baseline.sh'
        original = script.read_text()
        script.write_text('TOKEN="sk-' + 'A' * 32 + '"\n')
        rejected = self.cli('run', '--task', 'hstu', '--',
                            'bash', 'train_baseline.sh', 'beauty', check=False)
        self.assertNotEqual(rejected.returncode, 0)
        self.assertFalse((self.workspace / '.autorse-runs/ACTIVE').exists())
        script.write_text(original)
        self.assertEqual(self.run_experiment()['returncode'], 0)

    def test_dirty_and_untracked_source_cannot_be_exported(self):
        self.successful_experiment()
        for name in ('train.py', 'new_source.py'):
            with self.subTest(path=name):
                path = self.baseline / name
                previous = path.read_bytes() if path.exists() else None
                path.write_text('print("source missing from selected commit")\n')
                self.assertNotEqual(self.cli('check', check=False).returncode, 0)
                self.assertNotEqual(self.export_cli(check=False).returncode, 0)
                self.assertFalse((self.workspace / 'artifact').exists())
                if previous is None:
                    path.unlink()
                else:
                    path.write_bytes(previous)

    def test_a_failed_run_with_changed_runtime_cannot_export_a_stale_prediction(self):
        self.successful_experiment()
        stale = (self.workspace / PREDICTION).read_bytes()
        # The old successful receipt no longer describes the runtime files.
        # A stale prediction must not make this changed, failed runtime valid.
        (self.workspace / 'train_baseline.sh').write_text('#!/usr/bin/env bash\nexit 17\n')
        failed = self.cli('run', '--task', 'hstu', '--',
                          'bash', 'train_baseline.sh', 'beauty', check=False)
        self.assertNotEqual(failed.returncode, 0)
        latest = json.loads((self.workspace / '.autorse-runs/latest.json').read_text())
        self.assertEqual(latest['returncode'], 17)
        self.assertEqual(stale, (self.workspace / PREDICTION).read_bytes())
        self.assertNotEqual(self.export_cli(check=False).returncode, 0)
        self.assertFalse((self.workspace / 'artifact').exists())

    def test_earlier_best_branch_and_saved_predictions_can_be_restored_after_failure(self):
        best_code, best_run = self.successful_experiment()
        best_prediction = (self.workspace / PREDICTION).read_bytes()
        self.new_experiment('exp/002', 'print("later non-selected experiment")\n')
        self.run_experiment()
        self.record_experiment()
        self.assertNotEqual(best_prediction, (self.workspace / PREDICTION).read_bytes())
        failed = self.cli('run', '--task', 'hstu', '--',
                          'bash', '-c', 'exit 19', check=False)
        self.assertNotEqual(failed.returncode, 0)
        latest = json.loads((self.workspace / '.autorse-runs/latest.json').read_text())
        self.assertEqual(latest['returncode'], 19)

        self.git('switch', 'exp/001')
        (self.workspace / PREDICTION).write_bytes(best_prediction)
        self.export_cli()
        manifest = json.loads((self.workspace / 'artifact/GIT_PROVENANCE.json').read_text())
        serialized = json.dumps(manifest)
        for value in (best_code, best_run['run_id'], best_run['prediction']['sha256']):
            self.assertIn(value, serialized)

    def test_a_new_branch_requires_its_own_matching_run(self):
        self.successful_experiment()
        self.cli('new', 'exp/never-run', '--from', 'exp/001')
        self.assertNotEqual(self.export_cli(check=False).returncode, 0)
        self.assertFalse((self.workspace / 'artifact').exists())

    def test_unversioned_helpers_outside_baseline_are_rejected_after_a_valid_run(self):
        self.successful_experiment()
        (self.workspace / 'unversioned_helper.py').write_text('print("untracked helper")\n')
        rejected = self.export_cli(check=False, extra=('--helper', 'unversioned_helper.py'))
        self.assertNotEqual(rejected.returncode, 0)
        self.assertFalse((self.workspace / 'artifact').exists())

    def test_ignored_configuration_cannot_be_exported_with_an_old_run(self):
        self.initialize()
        self.new_experiment()
        ignore = self.baseline / '.gitignore'
        ignore.write_text(ignore.read_text() + '\nignored_config.yaml\n')
        self.cli('commit', '-m', 'Ignore generated configuration')
        config = self.baseline / 'ignored_config.yaml'
        config.write_text('learning_rate: 0.1\n')
        (self.workspace / 'train_baseline.sh').write_text(
            '#!/usr/bin/env bash\nset -eu\n'
            'mkdir -p datasets/beauty/working\n'
            'cat baseline/ignored_config.yaml > datasets/beauty/working/submission.npz\n')
        self.run_experiment()
        self.record_experiment()
        config.write_text('learning_rate: 10.0\n')
        self.assertEqual('', self.git('status', '--porcelain').stdout)
        rejected = self.export_cli(check=False, extra=('--config', 'baseline/ignored_config.yaml'))
        self.assertNotEqual(rejected.returncode, 0)
        self.assertFalse((self.workspace / 'artifact').exists())

    def test_selected_run_commands_are_scanned_before_exporting_provenance(self):
        self.initialize()
        self.new_experiment()
        command = ('mkdir -p datasets/beauty/working\n'
                   'printf prediction > datasets/beauty/working/submission.npz\n'
                   '# sk-' + 'A' * 32)
        self.cli('run', '--task', 'hstu', '--', 'bash', '-c', command)
        # No result record is committed, so history scans cannot detect this
        # receipt's credential; the final provenance itself must be checked.
        rejected = self.export_cli(check=False)
        self.assertNotEqual(rejected.returncode, 0)
        self.assertFalse((self.workspace / 'artifact').exists())

    def test_export_rejects_predictions_changed_after_the_recorded_run(self):
        self.successful_experiment()
        (self.workspace / PREDICTION).write_bytes(b'prediction from another run')
        self.assertNotEqual(self.export_cli(check=False).returncode, 0)
        self.assertFalse((self.workspace / 'artifact').exists())

    def test_export_rejects_source_committed_after_the_recorded_run(self):
        self.successful_experiment()
        (self.baseline / 'train.py').write_text('print("never trained")\n')
        self.cli('commit', '-m', 'Change selected code without running it')
        self.assertNotEqual(self.export_cli(check=False).returncode, 0)
        self.assertFalse((self.workspace / 'artifact').exists())

    def test_final_export_binds_bundle_source_code_commit_and_prediction_hash(self):
        code_commit, run = self.successful_experiment()
        selected_commit = self.git('rev-parse', 'HEAD').stdout.strip()
        self.export_cli()
        artifact = self.workspace / 'artifact'
        self.assertEqual((self.baseline / 'train.py').read_bytes(),
                         (artifact / 'source/baseline/train.py').read_bytes())
        manifest = json.loads((artifact / 'GIT_PROVENANCE.json').read_text())
        serialized = json.dumps(manifest)
        for value in (code_commit, selected_commit, run['prediction']['sha256'], 'exp/001'):
            self.assertIn(value, serialized)
        self.assertTrue((artifact / 'baseline.git.bundle').is_file())
        for line in (artifact / 'SHA256SUMS').read_text().splitlines():
            digest, name = line.split('  ', 1)
            self.assertEqual(digest, hashlib.sha256((artifact / name).read_bytes()).hexdigest())
        prediction = self.workspace / PREDICTION
        self.assertEqual(run['prediction']['sha256'],
                         hashlib.sha256(prediction.read_bytes()).hexdigest())


if __name__ == '__main__':
    unittest.main()
