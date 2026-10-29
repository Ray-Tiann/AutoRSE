"""Private dataset deployment gates, exercised with synthetic opaque files only."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from space import service
from autorse.release import PUBLIC_BUNDLE_SHA256


@unittest.skipUnless(importlib.util.find_spec("huggingface_hub"), "Optional Space dependency is not installed")
class PrivateProvisioningTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name) / "synthetic-hub"
        self.destination = Path(self.temp.name) / "private-cache"
        self.source.mkdir()
        rows = []
        for task, paths in service.PRIVATE_FILES.items():
            for relative in paths:
                name = f"{task}/{relative}"
                path = self.source / name
                path.parent.mkdir(parents=True, exist_ok=True)
                data = b"synthetic opaque test fixture"
                path.write_bytes(data)
                rows.append({"path": name, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()})
        self.manifest = {"files": rows, "public_bundle_sha256": PUBLIC_BUNDLE_SHA256}
        self.env = {"AUTORSE_PRIVATE_REPO": "synthetic/private", "AUTORSE_PRIVATE_REVISION": "a" * 40,
                    "HF_TOKEN": "synthetic-owner-token"}

    def run_provision(self, private=True, corrupt=False):
        payload = json.dumps(self.manifest).encode()
        (self.source / "private-manifest.json").write_bytes(payload)
        self.env["AUTORSE_PRIVATE_MANIFEST_SHA256"] = hashlib.sha256(payload).hexdigest()
        if corrupt:
            (self.source / self.manifest["files"][0]["path"]).write_bytes(b"corrupt")
        with patch.dict("os.environ", self.env, clear=True), patch.object(service, "PRIVATE_ROOT", self.destination), \
                patch("huggingface_hub.HfApi") as api, patch("huggingface_hub.hf_hub_download") as fetch:
            api.return_value.dataset_info.return_value = SimpleNamespace(private=private, sha="a" * 40)
            fetch.side_effect = lambda repo, filename, **kwargs: str(self.source / filename)
            service.provision_private_data()

    def test_private_revision_and_file_hashes_accept_matching_opaque_files(self):
        self.run_provision()
        for row in self.manifest["files"]:
            target = self.destination / row["path"]
            self.assertTrue(target.is_file())
            self.assertEqual(target.stat().st_mode & 0o777, 0o600)

    def test_public_label_repository_is_refused(self):
        with self.assertRaises(service.EvaluatorUnavailable):
            self.run_provision(private=False)
        self.assertFalse(self.destination.exists())

    def test_corrupt_private_file_is_refused(self):
        with self.assertRaises(service.EvaluatorUnavailable) as caught:
            self.run_provision(corrupt=True)
        self.assertNotIn("synthetic-owner-token", str(caught.exception))

    def test_wrong_public_revision_is_refused(self):
        self.manifest["public_bundle_sha256"] = "0" * 64
        with self.assertRaises(service.EvaluatorUnavailable):
            self.run_provision()


if __name__ == "__main__":
    unittest.main()
