"""Submission-to-ranking integration with temporary synthetic evidence only."""
import copy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import test_validation
from autorse_leaderboard import cli


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {"GITHUB_SHA": "f" * 40})
        environment.start()
        self.addCleanup(environment.stop)
        self.fixture = test_validation.ValidationTests()
        self.fixture.setUp()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.submissions = self.root / "submissions"
        self.submissions.mkdir()
        self.path = self.submissions / "synthetic-agent.json"
        cli.write_json(self.path, self.fixture.submission)
        self.keys = self.root / "keys.json"
        cli.write_json(self.keys, self.fixture.keys)
        self.ledger = {"schema_version": 1, "repository": "Ray-Tiann/AutoRSE", "main_sha": "f" * 40,
                       "approvals": [{"submission_path": "leaderboard/submissions/synthetic-agent.json",
                                      "submission_sha256": hashlib.sha256(self.path.read_bytes()).hexdigest(),
                                      "pr_number": 10, "author": "synthetic-author", "head_sha": "e" * 40,
                                      "approved_reviews": [{"reviewer": "synthetic-maintainer", "permission": "write",
                                                            "commit_id": "e" * 40}]}]}
        self.ledger_path = self.root / "approvals.json"
        cli.write_json(self.ledger_path, self.ledger)
        self.args = SimpleNamespace(keys=str(self.keys), approval_ledger=str(self.ledger_path),
                                    submissions_dir=str(self.submissions), output_dir=str(self.root / "generated"),
                                    repo_id="Ray0311/AutoRSE-Leaderboard", dry_run=True)

    def reader(self, ref):
        return self.fixture.blobs[ref["path"]]

    def test_approved_evidence_to_six_task_rank_snapshot(self):
        with patch("autorse_leaderboard.artifacts.PublicArtifactReader", return_value=self.reader):
            cli.publish(self.args)
        data = cli.read_json(self.root / "generated/community-leaderboard.json")
        self.assertEqual(data["rows"][0]["mean_task_rank"], 1)
        self.assertEqual(data["rows"][0]["trial_counts"], [3] * 6)
        self.assertNotIn("relative_gain_percent", data["rows"][0])
        self.assertTrue((self.root / "generated/submissions/synthetic-agent.json").is_file())

    def test_no_approval_tamper_and_self_approval_are_rejected(self):
        for transform in (
            lambda x: x.update(approvals=[]),
            lambda x: x["approvals"][0].update(submission_sha256="0" * 64),
            lambda x: x["approvals"][0]["approved_reviews"][0].update(reviewer="synthetic-author"),
            lambda x: x["approvals"][0].update(head_sha=""),
        ):
            ledger = copy.deepcopy(self.ledger)
            transform(ledger)
            with self.assertRaises(ValueError):
                cli.check_approval(self.path, self.fixture.submission, ledger)

    def test_missing_evidence_and_test_kind_cannot_publish(self):
        for case in ("evidence", "test"):
            with self.subTest(case=case):
                submission = copy.deepcopy(self.fixture.submission)
                if case == "test":
                    submission["kind"] = "test"
                cli.write_json(self.path, submission)
                ledger = copy.deepcopy(self.ledger)
                ledger["approvals"][0]["submission_sha256"] = hashlib.sha256(self.path.read_bytes()).hexdigest()
                cli.write_json(self.ledger_path, ledger)
                reader = self.reader if case == "test" else lambda ref: b"changed artifact"
                with patch("autorse_leaderboard.artifacts.PublicArtifactReader", return_value=reader):
                    with self.assertRaises(ValueError):
                        cli.publish(self.args)
        self.assertFalse((self.root / "generated/community-leaderboard.json").exists())

    def test_stale_and_ambiguous_json_are_rejected(self):
        with patch.dict("os.environ", {"GITHUB_SHA": "a" * 40}):
            with self.assertRaises(ValueError):
                cli.check_approval(self.path, self.fixture.submission, self.ledger)
        self.path.write_text('{"id":"one","id":"two"}')
        with self.assertRaises(ValueError):
            cli.read_json(self.path)

    def test_template_has_eighteen_runs_and_false_acknowledgements(self):
        value = cli.template()
        self.assertEqual(len(value["runs"]), 18)
        self.assertFalse(value["acknowledgements"]["public_evidence"])


if __name__ == "__main__":
    unittest.main()
