"""Security and installation-contract tests using small synthetic archives only."""

import contextlib
import hashlib
import io
import json
from pathlib import Path
import stat
import tempfile
import unittest
from unittest import mock
import warnings
import zipfile

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import install_public_data as installer


class InstallerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="autorse-installer-test-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.root = self.directory / "checkout"
        self.archive = self.directory / "public.zip"
        self.data = {f"benchmark/tasks/{task}/environment/datasets/example/input.bin": task.encode()
                     for task in installer.TASKS}

    def create_archive(self, transform=None, extra=None):
        entries = [{"path": name, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                   for name, data in self.data.items()]
        manifest = {"schema_version": 1, "tasks": list(installer.TASKS), "files": entries,
                    "file_count": len(entries), "total_bytes": sum(entry["size"] for entry in entries)}
        if transform:
            transform(manifest)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with zipfile.ZipFile(self.archive, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for name, data in self.data.items():
                    archive.writestr(name, data)
                archive.writestr(installer.MANIFEST_NAME, json.dumps(manifest))
                if extra:
                    extra(archive)
        self.pins = mock.patch.multiple(installer,
                                       BUNDLE_BYTES=self.archive.stat().st_size,
                                       BUNDLE_SHA256=installer.sha256_file(self.archive))
        self.pins.start()
        self.addCleanup(self.pins.stop)

    def run_install(self, tasks=None):
        with contextlib.redirect_stdout(io.StringIO()):
            return installer.install(self.archive, self.root, set(tasks or installer.TASKS))

    def test_selective_install_and_idempotence(self):
        self.create_archive()
        self.assertEqual(self.run_install({"hstu_task"}), (1, 0))
        self.assertEqual(self.run_install({"hstu_task"}), (0, 1))
        files = list(self.root.rglob("*.bin"))
        self.assertEqual(len(files), 1)
        self.assertEqual(files[0].read_bytes(), b"hstu_task")

    def test_all_tasks_install(self):
        self.create_archive()
        self.assertEqual(self.run_install(), (6, 0))

    def test_conflicts_refused_before_other_writes(self):
        self.create_archive()
        last = self.root / list(self.data)[-1]
        last.parent.mkdir(parents=True)
        last.write_bytes(b"owner data")
        with self.assertRaisesRegex(installer.DataError, "existing file differs"):
            self.run_install()
        self.assertEqual(last.read_bytes(), b"owner data")
        self.assertEqual(list(self.root.rglob("*.bin")), [last])

    def test_archive_pin_required(self):
        self.create_archive()
        with mock.patch.object(installer, "BUNDLE_SHA256", "0" * 64):
            with self.assertRaisesRegex(installer.DataError, "SHA-256"):
                self.run_install()
        self.assertFalse(self.root.exists())

    def test_corrupt_file_hash_never_published(self):
        self.create_archive(transform=lambda manifest: manifest["files"][0].update(sha256="0" * 64))
        with self.assertRaisesRegex(installer.DataError, "SHA-256/size mismatch"):
            self.run_install()
        self.assertFalse(any(self.root.rglob("*.partial")))
        self.assertFalse(any(self.root.rglob("*.bin")))

    def test_duplicate_zip_name_rejected(self):
        self.create_archive(extra=lambda archive: archive.writestr(next(iter(self.data)), b"duplicate"))
        with self.assertRaisesRegex(installer.DataError, "duplicate ZIP"):
            self.run_install()
        self.assertFalse(self.root.exists())

    def test_unlisted_member_rejected(self):
        self.create_archive(extra=lambda archive: archive.writestr("benchmark/tasks/hstu_task/tests/private/labels", b"never install"))
        with self.assertRaisesRegex(installer.DataError, "inventories differ"):
            self.run_install()
        self.assertFalse(self.root.exists())

    def test_zip_symlink_rejected(self):
        def add(archive):
            info = zipfile.ZipInfo("symlink")
            info.create_system = 3
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(info, "/tmp/unsafe")
        self.create_archive(extra=add)
        with self.assertRaisesRegex(installer.DataError, "symlink"):
            self.run_install()
        self.assertFalse(self.root.exists())

    def test_destination_parent_symlink_refused(self):
        self.create_archive()
        self.root.mkdir()
        external = self.directory / "external"
        external.mkdir()
        (self.root / "benchmark").symlink_to(external, target_is_directory=True)
        with self.assertRaisesRegex(installer.DataError, "symlink"):
            self.run_install()
        self.assertEqual(list(external.iterdir()), [])

    def test_root_symlink_refused(self):
        self.create_archive()
        external = self.directory / "external"
        external.mkdir()
        self.root.symlink_to(external, target_is_directory=True)
        with self.assertRaisesRegex(installer.DataError, "root must not be a symlink"):
            self.run_install()

    def test_destination_file_symlink_refused(self):
        self.create_archive()
        target = self.root / next(iter(self.data))
        target.parent.mkdir(parents=True)
        external = self.directory / "external"
        external.write_bytes(b"keep me")
        target.symlink_to(external)
        with self.assertRaisesRegex(installer.DataError, "symlink"):
            self.run_install()
        self.assertEqual(external.read_bytes(), b"keep me")

    def test_unsafe_paths_rejected(self):
        for path in ["/absolute", "benchmark/tasks/hstu_task/environment/datasets/../bad",
                     "benchmark/tasks/hstu_task/environment/datasets//bad",
                     "benchmark/tasks/hstu_task/environment/datasets/C:bad",
                     "benchmark/tasks/hstu_task/environment/datasets/working/output",
                     "benchmark/tasks/hstu_task/tests/private/labels",
                     "benchmark/tasks/new_task/environment/datasets/input.bin"]:
            with self.subTest(path=path), self.assertRaises(installer.DataError):
                installer.public_path(path)

    def test_archive_size_limit(self):
        self.create_archive()
        with mock.patch.object(installer, "MAX_TOTAL_BYTES", 1):
            with self.assertRaisesRegex(installer.DataError, "total size"):
                self.run_install()

    def test_manifest_size_mismatch(self):
        self.create_archive(transform=lambda manifest: manifest["files"][0].update(size=999))
        with self.assertRaisesRegex(installer.DataError, "total size"):
            self.run_install()

    def test_duplicate_json_keys(self):
        with self.assertRaisesRegex(installer.DataError, "duplicate JSON key"):
            json.loads('{"schema_version":1,"schema_version":2}', object_pairs_hook=installer.unique_json_object)

    def test_download_input_validation(self):
        for repo in ["../oops", "owner/name/extra", "https://example.com/file", "name"]:
            with self.subTest(repo=repo), self.assertRaises(installer.DataError):
                installer.download_bundle(repo, "main", self.directory / "unused.zip")

    def test_default_dataset_and_explicit_archive(self):
        self.assertEqual(installer.parse_args([]).repo_id, "RayTian/AutoRSE")
        args = installer.parse_args(["--archive", "local.zip"])
        self.assertEqual(args.archive, Path("local.zip"))
        self.assertIsNone(args.repo_id)


if __name__ == "__main__":
    unittest.main()
