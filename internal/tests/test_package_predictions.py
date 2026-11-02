"""Six-task packaging tests with opaque synthetic predictions, never model data."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import zipfile


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('autorse_package_predictions_test_target',
                                            ROOT / 'tools/package_predictions.py')
packager = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(packager)


class PackagePredictionsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='autorse-package-test-')
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name).resolve()
        self.runs = self.directory / 'runs'
        self.label = 'my-agent-v1'
        self.seed = 2024
        self.output = self.directory / 'submission.zip'
        self.sources = {}
        self.workspaces = {}
        self.contents = {}
        for task, spec in packager.TASKS.items():
            workspace = self.job(task) / 'trial-1/artifacts/workspace'
            source = workspace / 'datasets' / spec['dataset'] / 'working' / spec['file']
            source.parent.mkdir(parents=True)
            content = task.encode() + b'\x00opaque prediction bytes\xff\n'
            source.write_bytes(content)
            (workspace / 'artifact').mkdir()
            (workspace / 'artifact/README.md').write_text('Evidence stays outside the scoring ZIP.\n')
            self.sources[task] = source
            self.workspaces[task] = workspace
            self.contents[f"{task}/{spec['file']}"] = content
        validation = mock.patch.object(packager, 'validate', return_value={'format_valid': True})
        self.validator = validation.start()
        self.addCleanup(validation.stop)

    def job(self, task):
        return self.runs / f'{self.label}-{task}-s{self.seed}'

    def package(self, **kwargs):
        return packager.package_predictions(self.runs, self.label, self.seed,
                                             kwargs.pop('output', self.output), **kwargs)

    def assert_no_temporary_outputs(self):
        self.assertEqual(list(self.directory.glob(f'.{self.output.name}.*.tmp')), [])

    def assert_zip_contents(self, output):
        with zipfile.ZipFile(output) as archive:
            self.assertEqual(set(archive.namelist()), set(self.contents))
            self.assertEqual(len(archive.infolist()), 6)
            self.assertFalse(any(member.is_dir() for member in archive.infolist()))
            for member, content in self.contents.items():
                self.assertEqual(archive.read(member), content)

    def test_all_six_are_validated_before_zip_creation_and_bytes_are_unchanged(self):
        # A different configuration's collected output must not be selected.
        other = self.runs / f'other-agent-hstu-s{self.seed}/trial/artifacts/workspace'
        other.mkdir(parents=True)
        validated = []

        def validate(task, source):
            self.assertEqual(source, self.sources[task])
            self.assertFalse(self.output.exists())
            validated.append(task)

        mkstemp = tempfile.mkstemp

        def create_temporary(*args, **kwargs):
            self.assertEqual(validated, list(packager.TASKS))
            return mkstemp(*args, **kwargs)

        self.validator.side_effect = validate
        with mock.patch.object(packager.tempfile, 'mkstemp', side_effect=create_temporary):
            output = self.package()
        self.assertEqual(output, self.output.absolute())
        self.assert_zip_contents(output)
        for task, source in self.sources.items():
            self.assertEqual(source.read_bytes(), self.contents[f"{task}/{packager.TASKS[task]['file']}"])
        self.assert_no_temporary_outputs()

    def test_default_output_uses_the_run_label_and_seed_in_the_current_directory(self):
        previous = Path.cwd()
        try:
            os.chdir(self.directory)
            output = packager.package_predictions(self.runs, self.label, self.seed)
        finally:
            os.chdir(previous)
        self.assertEqual(output, self.directory / f'{self.label}-s{self.seed}.zip')
        self.assert_zip_contents(output)

    def test_missing_or_ambiguous_collected_trials_are_not_silently_selected(self):
        workspace = self.workspaces['hstu']
        saved = self.directory / 'saved-workspace'
        workspace.rename(saved)
        try:
            with self.assertRaisesRegex(ValueError, 'exactly one collected trial'):
                self.package()
        finally:
            saved.rename(workspace)
        extra = self.job('hstu') / 'trial-2/artifacts/workspace'
        extra.mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, 'exactly one collected trial'):
            self.package()
        self.validator.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_missing_prediction_is_detected_before_any_format_validation(self):
        self.sources[list(packager.TASKS)[-1]].unlink()
        with self.assertRaises(ValueError):
            self.package()
        self.validator.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_symlinks_in_prediction_trial_and_workspace_paths_are_rejected(self):
        source = self.sources['hstu']
        workspace = self.workspaces['hstu']
        for index, path in enumerate((source, source.parent, workspace, workspace.parent.parent)):
            with self.subTest(path=path.relative_to(self.runs)):
                saved = self.directory / f'external-{index}'
                is_directory = path.is_dir()
                path.rename(saved)
                path.symlink_to(saved, target_is_directory=is_directory)
                try:
                    with self.assertRaises(ValueError):
                        self.package()
                finally:
                    path.unlink()
                    saved.rename(path)
        alias = self.directory / 'runs-link'
        alias.symlink_to(self.runs, target_is_directory=True)
        with self.assertRaises(ValueError):
            packager.package_predictions(alias, self.label, self.seed, self.output)
        self.validator.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_run_labels_cannot_contain_paths_globs_or_shell_characters(self):
        for label in ('', '../other', 'a/b', 'a\\b', 'agent*', 'agent?', '[agent]',
                      'agent name', 'agent.v1', '-agent', '$HOME'):
            with self.subTest(label=label):
                with self.assertRaises(ValueError):
                    packager.package_predictions(self.runs, label, self.seed, self.output)
        self.validator.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_output_requires_a_regular_zip_path_and_existing_parent(self):
        directory_output = self.directory / 'directory.zip'
        directory_output.mkdir()
        existing = self.directory / 'existing.zip'
        existing.write_bytes(b'preserved external file')
        link = self.directory / 'link.zip'
        link.symlink_to(existing)
        for output in (self.directory / 'wrong.txt', directory_output,
                       self.directory / 'missing/parent.zip', link):
            with self.subTest(output=output.name):
                with self.assertRaises(ValueError):
                    self.package(output=output, overwrite=True)
        self.assertEqual(existing.read_bytes(), b'preserved external file')
        self.assertTrue(link.is_symlink())
        self.validator.assert_not_called()

    def test_output_cannot_be_a_hard_link_to_a_prediction_source(self):
        source = self.sources['hstu']
        before = source.read_bytes()
        os.link(source, self.output)
        with self.assertRaises(ValueError):
            self.package(overwrite=True)
        self.assertEqual(source.read_bytes(), before)
        self.assertEqual(self.output.read_bytes(), before)
        self.validator.assert_not_called()

    def test_existing_output_requires_overwrite_and_replacement_has_no_stale_members(self):
        with zipfile.ZipFile(self.output, 'w') as archive:
            archive.writestr('stale/source.py', b'old source must not be included')
        before = self.output.read_bytes()
        with self.assertRaises(ValueError):
            self.package()
        self.assertEqual(self.output.read_bytes(), before)
        self.validator.assert_not_called()
        self.package(overwrite=True)
        self.assert_zip_contents(self.output)
        self.assert_no_temporary_outputs()

    def test_format_failure_preserves_previous_output_without_creating_a_zip(self):
        self.output.write_bytes(b'previous accepted ZIP')
        last = list(packager.TASKS)[-1]

        def validate(task, source):
            if task == last:
                raise ValueError('invalid final task prediction')

        self.validator.side_effect = validate
        with mock.patch.object(packager.tempfile, 'mkstemp') as create_temporary:
            with self.assertRaisesRegex(ValueError, 'invalid final task'):
                self.package(overwrite=True)
        self.assertEqual(self.validator.call_count, 6)
        create_temporary.assert_not_called()
        self.assertEqual(self.output.read_bytes(), b'previous accepted ZIP')
        self.assert_no_temporary_outputs()

    def test_zip_write_failure_preserves_previous_output_and_removes_staging_file(self):
        self.output.write_bytes(b'previous accepted ZIP')
        write = zipfile.ZipFile.write
        calls = []

        def fail_after_first(archive, source, arcname=None, **kwargs):
            calls.append(arcname)
            if len(calls) == 2:
                raise OSError('simulated full disk')
            return write(archive, source, arcname=arcname, **kwargs)

        with mock.patch.object(packager.zipfile.ZipFile, 'write', new=fail_after_first):
            with self.assertRaisesRegex(OSError, 'simulated full disk'):
                self.package(overwrite=True)
        self.assertEqual(len(calls), 2)
        self.assertEqual(self.validator.call_count, 6)
        self.assertEqual(self.output.read_bytes(), b'previous accepted ZIP')
        self.assert_no_temporary_outputs()

    def test_cli_defaults_and_required_arguments(self):
        with mock.patch.object(packager, 'package_predictions', return_value=self.output) as package:
            arguments = ['package_predictions.py', '--run-label', self.label, '--seed', str(self.seed)]
            with mock.patch.object(sys, 'argv', arguments), contextlib.redirect_stdout(io.StringIO()) as stdout:
                packager.main()
            package.assert_called_once_with(ROOT / 'runs', self.label, self.seed, None, False)
            self.assertEqual(stdout.getvalue().strip(), str(self.output))
        for arguments in (['--seed', '2024'], ['--run-label', self.label],
                          ['--run-label', self.label, '--seed', 'not-an-integer']):
            with self.subTest(arguments=arguments), mock.patch.object(sys, 'argv', ['package_predictions.py', *arguments]):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as rejected:
                    packager.main()
                self.assertEqual(rejected.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
