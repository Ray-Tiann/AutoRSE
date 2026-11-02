"""Synthetic protocol tests: no private labels, remote requests or participant code."""
import base64
import copy
import hashlib
import json
import unittest
from unittest.mock import patch

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization

from autorse_leaderboard import receipts
from autorse_leaderboard.validation import (
    validate_submission, TASK_ORDER, DATA_REVISION, PUBLIC_BUNDLE_SHA256, PRIMARY_METRICS,
)


class ValidationTests(unittest.TestCase):
    def setUp(self):
        key = Ed25519PrivateKey.generate()
        self.private_key = base64.b64encode(key.private_bytes(
            serialization.Encoding.Raw, serialization.PrivateFormat.Raw,
            serialization.NoEncryption())).decode("ascii")
        self.keys = {"unit-test": {"algorithm": "Ed25519", "public_key": base64.b64encode(
            key.public_key().public_bytes(serialization.Encoding.Raw,
                                         serialization.PublicFormat.Raw)).decode("ascii")}}
        self.blobs = {}
        self.submission = {
            "schema_version": 1, "id": "synthetic-agent", "kind": "benchmark",
            "configuration": {
                "name": "Synthetic agent", "agent": "test-agent", "harness": "test-harness",
                "agent_version": "1.0", "model": "synthetic-model", "model_version": "1.0",
                "framework_commit": "a" * 40, "code_url": "https://example.org/research/code",
            },
            "protocol": {"data_revision": DATA_REVISION, "public_bundle_sha256": PUBLIC_BUNDLE_SHA256,
                         "research_budget_seconds": 43200, "gpus_per_run": 1},
            "receipts": [], "runs": [],
            "acknowledgements": {"public_evidence": True, "protocol_followed": True},
        }
        for trial in range(1, 4):
            tasks = {}
            for task_index, task in enumerate(TASK_ORDER):
                prediction = hashlib.sha256(f"{task}:{trial}".encode()).hexdigest()
                score = 0.30 + task_index * 0.05 + trial * 0.01
                metric = PRIMARY_METRICS[task]
                tasks[task] = {"task": task, "data_revision": DATA_REVISION,
                               "public_bundle_sha256": PUBLIC_BUNDLE_SHA256,
                               "submission_sha256": prediction, "metric": metric,
                               "metrics": {metric: score, "reward": score}}
                run = {
                    "task": task, "trial_index": trial, "run_id": f"{task}-trial-{trial}",
                    "seed": 2000 + trial, "receipt_evaluation_id": f"evaluation-{trial}",
                    "prediction_sha256": prediction, "code_commit": "b" * 40,
                    "agent_started_at": "2026-10-29T00:00:00+00:00",
                    "agent_finished_at": "2026-10-29T12:03:00+00:00", "gpus": 1,
                    "artifacts": {},
                }
                self.submission["runs"].append(run)
                config = {"research_budget_seconds": 43200, "minimum_runtime_sec": 43200,
                          "finalization_grace_sec": 300, "gpus": 1, "seed": run["seed"],
                          "harness": "test-harness", "model": "synthetic-model",
                          "command": ["harbor", "run", "--n-concurrent", "1", "--n-attempts", "1"]}
                provenance = {"evaluated_code_commit": run["code_commit"],
                              "run": {"code_commit": run["code_commit"], "seed": run["seed"],
                                      "prediction": {"sha256": prediction}}}
                for kind, value in (
                    ("code", b"Opaque source archive. It must never be executed."),
                    ("report", b"# Experiment report\nSynthetic changes and validation observations.\n"),
                    ("trajectory", {"messages": [{"role": "assistant", "content": "Synthetic event"}]}),
                    ("config", config), ("provenance", provenance),
                ):
                    self.replace_artifact(len(self.submission["runs"]) - 1, kind, value)
            payload = {
                "schema_version": 1, "status": "complete", "complete": True,
                "evaluation_id": f"evaluation-{trial}", "evaluated_at": "2026-10-30T00:00:00Z",
                "task_count": 6, "task_order": list(TASK_ORDER), "data_revision": DATA_REVISION,
                "public_bundle_sha256": PUBLIC_BUNDLE_SHA256, "batch_sha256": "c" * 64,
                "evaluator_code_sha256": "d" * 64, "participant": "synthetic-researcher",
                "usage": "benchmark", "tasks": tasks,
            }
            self.submission["receipts"].append(receipts.sign_receipt(payload, self.private_key, "unit-test"))

    def replace_artifact(self, run_index, kind, value):
        run = self.submission["runs"][run_index]
        data = value if type(value) is bytes else json.dumps(value, sort_keys=True, allow_nan=False).encode("utf-8")
        path = f"{run['run_id']}/{kind}.dat"
        self.blobs[path] = data
        run["artifacts"][kind] = {
            "repo_id": "research/public-evidence", "revision": "e" * 40, "path": path,
            "sha256": hashlib.sha256(data).hexdigest(), "size": len(data),
        }

    def read_artifact(self, run_index, kind):
        return json.loads(self.blobs[self.submission["runs"][run_index]["artifacts"][kind]["path"]])

    def resign(self, index):
        payload = dict(self.submission["receipts"][index])
        payload.pop("signature", None)
        self.submission["receipts"][index] = receipts.sign_receipt(payload, self.private_key, "unit-test")

    def validate(self, **kwargs):
        return validate_submission(self.submission, self.keys,
                                   artifact_reader=lambda ref: self.blobs[ref["path"]], **kwargs)

    @staticmethod
    def failed_codes(report):
        return [item["code"] for item in report["checks"] if not item["passed"]]

    def test_complete_signed_three_trial_submission(self):
        report = self.validate()
        self.assertTrue(report["valid"], self.failed_codes(report))
        self.assertTrue(report["publishable"])
        self.assertTrue(report["summary"]["requires_maintainer_review"])
        self.assertEqual(report["summary"]["task_order"], list(TASK_ORDER))
        self.assertEqual(report["summary"]["trial_counts"], [3] * 6)
        self.assertAlmostEqual(report["summary"]["scores"][0], 0.32)
        self.assertAlmostEqual(report["summary"]["sd"][0], 0.01)

    def test_metadata_only_cannot_publish(self):
        report = validate_submission(self.submission, self.keys)
        self.assertFalse(report["valid"])
        self.assertTrue(report["metadata_valid"])
        self.assertFalse(report["summary"]["publishable"])
        self.assertIn("artifacts.pending", self.failed_codes(report))

    def test_test_kind_requires_opt_in_and_never_publishes(self):
        self.submission["kind"] = "test"
        self.assertFalse(self.validate()["valid"])
        report = self.validate(allow_test=True)
        self.assertTrue(report["valid"])
        self.assertFalse(report["publishable"])
        self.assertFalse(report["summary"]["publishable"])

    def test_all_signatures_checked_before_receipt_fields(self):
        self.submission["receipts"][0]["signature"]["value"] = "invalid"
        self.submission["receipts"][1]["participant"] = "tampered"
        with patch.object(receipts, "verify_receipt", wraps=receipts.verify_receipt) as verify:
            report = self.validate()
        self.assertFalse(report["valid"])
        self.assertEqual(verify.call_count, 3)
        self.assertNotIn("receipts.protocol.1", [c["code"] for c in report["checks"]])

    def test_unsigned_receipt_rejected(self):
        del self.submission["receipts"][0]["signature"]
        self.assertIn("receipts.signature.1", self.failed_codes(self.validate()))

    def test_signed_demo_receipt_rejected_even_in_test_mode(self):
        self.submission["kind"] = "test"
        self.submission["receipts"][0]["usage"] = "demo"
        self.resign(0)
        self.assertFalse(self.validate(allow_test=True)["valid"])

    def test_different_receipt_participants_rejected(self):
        self.submission["receipts"][1]["participant"] = "another-participant"
        self.resign(1)
        self.assertIn("receipts.participant", self.failed_codes(self.validate()))

    def test_duplicate_evaluation_id_rejected(self):
        self.submission["receipts"][1]["evaluation_id"] = "evaluation-1"
        self.resign(1)
        self.assertIn("receipts.unique", self.failed_codes(self.validate()))

    def test_receipt_list_order_does_not_change_trial_binding(self):
        self.submission["receipts"].reverse()
        report = self.validate()
        self.assertTrue(report["valid"], self.failed_codes(report))
        self.assertAlmostEqual(report["summary"]["task_scores"]["hstu"][0], 0.31)

    def test_wrong_receipt_prediction_digest_rejected(self):
        self.submission["runs"][0]["prediction_sha256"] = "f" * 64
        self.assertIn("runs.receipt_binding.1", self.failed_codes(self.validate()))

    def test_receipt_task_set_malformed_is_safe_failure(self):
        self.submission["receipts"][0]["tasks"] = []
        self.resign(0)
        self.assertFalse(self.validate()["valid"])

    def test_duplicate_run_and_task_trial_pairs_rejected(self):
        self.submission["runs"][1] = copy.deepcopy(self.submission["runs"][0])
        self.assertIn("runs.unique.2", self.failed_codes(self.validate()))

    def test_seed_must_differ_across_trials_for_each_task(self):
        self.submission["runs"][6]["seed"] = self.submission["runs"][0]["seed"]
        self.assertIn("runs.seeds", self.failed_codes(self.validate()))

    def test_each_trialset_uses_one_complete_receipt(self):
        self.submission["runs"][0]["receipt_evaluation_id"] = "evaluation-2"
        self.assertIn("runs.trialsets", self.failed_codes(self.validate()))

    def test_incomplete_short_or_extended_declared_budget_rejected(self):
        for end in ("2026-10-29T11:59:59Z", "2026-10-29T12:05:01Z", "2026-10-29T12:00:00"):
            with self.subTest(end=end):
                self.submission["runs"][0]["agent_finished_at"] = end
                self.assertIn("runs.declared_budget.1", self.failed_codes(self.validate()))

    def test_declared_agent_finish_cannot_follow_signed_evaluation_time(self):
        self.submission["runs"][0]["agent_started_at"] = "2026-10-30T00:00:00Z"
        self.submission["runs"][0]["agent_finished_at"] = "2026-10-30T12:00:00Z"
        self.assertIn("runs.evaluation_order.1", self.failed_codes(self.validate()))

    def test_timezones_are_compared_as_instants_including_equal_finish(self):
        self.submission["receipts"][0]["evaluated_at"] = "2026-10-29T20:03:00+08:00"
        self.resign(0)
        report = self.validate()
        self.assertTrue(report["valid"], self.failed_codes(report))
        self.submission["receipts"][0]["evaluated_at"] = "2026-10-29T20:02:59+08:00"
        self.resign(0)
        self.assertIn("runs.evaluation_order.1", self.failed_codes(self.validate()))

    def test_evaluation_timestamp_requires_timezone(self):
        self.submission["receipts"][0]["evaluated_at"] = "2026-10-30T00:00:00"
        self.resign(0)
        self.assertIn("receipts.signature.1", self.failed_codes(self.validate()))

    def test_boolean_numeric_fields_are_rejected(self):
        for target, field in ((self.submission["runs"][0], "seed"),
                              (self.submission["runs"][0], "gpus"),
                              (self.submission["protocol"], "gpus_per_run")):
            original = target[field]
            target[field] = True
            with self.subTest(field=field):
                self.assertFalse(self.validate()["valid"])
            target[field] = original

    def test_nan_in_json_is_rejected_before_signature_verification(self):
        self.submission["extra"] = float("nan")
        with patch.object(receipts, "verify_receipt") as verify:
            report = self.validate()
        self.assertFalse(report["valid"])
        verify.assert_not_called()

    def test_huge_or_boolean_score_is_safe_failure(self):
        for score in (True, 10 ** 1000):
            with self.subTest(score_type=type(score).__name__):
                metrics = self.submission["receipts"][0]["tasks"]["hstu"]["metrics"]
                metrics.update(ndcg_at_10=score, reward=score)
                self.resign(0)
                self.assertIn("receipts.scores.1", self.failed_codes(self.validate()))

    def test_data_revision_pin_is_required(self):
        self.submission["protocol"]["data_revision"] = "old-data"
        self.assertIn("protocol.pins", self.failed_codes(self.validate()))

    def test_unsafe_artifact_paths_and_unpinned_refs_rejected(self):
        ref = self.submission["runs"][0]["artifacts"]["code"]
        for field, value in (("path", "../private/file"), ("path", "/absolute"),
                             ("path", "data\\..\\private"), ("path", "%2e%2e/file"),
                             ("repo_id", "https://example.org/repo"), ("revision", "main"),
                             ("size", True), ("size", 160 * 1024 * 1024 + 1)):
            original = ref[field]
            ref[field] = value
            with self.subTest(field=field, value=value):
                self.assertIn("runs.artifact_refs.1", self.failed_codes(self.validate()))
            ref[field] = original

    def test_missing_artifact_rejected(self):
        del self.submission["runs"][0]["artifacts"]["trajectory"]
        self.assertIn("runs.artifact_refs.1", self.failed_codes(self.validate()))

    def test_artifact_hash_and_size_checked(self):
        path = self.submission["runs"][0]["artifacts"]["code"]["path"]
        original = self.blobs[path]
        self.blobs[path] = original[:-1] + bytes([original[-1] ^ 1])
        self.assertIn("artifacts.integrity.1.code", self.failed_codes(self.validate()))
        self.blobs[path] = original + b"extra"
        self.assertIn("artifacts.integrity.1.code", self.failed_codes(self.validate()))

    def test_reader_error_does_not_leak_exception_content(self):
        def failing_reader(_):
            raise OSError("PRIVATE_TOKEN_MUST_NOT_APPEAR")
        report = validate_submission(self.submission, self.keys, artifact_reader=failing_reader)
        self.assertFalse(report["valid"])
        self.assertNotIn("PRIVATE_TOKEN_MUST_NOT_APPEAR", json.dumps(report))

    def test_config_budget_resource_and_command_overrides_rejected(self):
        original = self.read_artifact(0, "config")
        variants = [dict(original, minimum_runtime_sec=60), dict(original, gpus=2),
                    dict(original, overrides={"gpus": 1}), dict(original, timeout_multiplier=2),
                    dict(original, command=["harbor", "run", "--timeout-multiplier", "2"]),
                    dict(original, command=["harbor", "run", "--agent-kwarg", "minimum_runtime_sec=5"]),
                    dict(original, command=["harbor", "run", "--no-persistence"])]
        for index, config in enumerate(variants):
            with self.subTest(variant=index):
                self.replace_artifact(0, "config", config)
                self.assertIn("artifacts.format.1.config", self.failed_codes(self.validate()))

    def test_task_timeout_is_not_mistaken_for_agent_wall_time(self):
        config = self.read_artifact(0, "config")
        config["task"] = {"timeout_sec": 900, "setup_timeout_sec": 120}
        self.replace_artifact(0, "config", config)
        self.assertTrue(self.validate()["valid"])

    def test_provenance_binds_evaluated_code_and_prediction(self):
        original = self.read_artifact(0, "provenance")
        for field in ("commit", "prediction"):
            value = copy.deepcopy(original)
            if field == "commit":
                value["evaluated_code_commit"] = "f" * 40
            else:
                value["run"]["prediction"]["sha256"] = "f" * 64
            self.replace_artifact(0, "provenance", value)
            with self.subTest(field=field):
                self.assertIn("artifacts.format.1.provenance", self.failed_codes(self.validate()))

    def test_trajectory_is_structured_and_nonempty(self):
        self.replace_artifact(0, "trajectory", {"messages": []})
        self.assertIn("artifacts.format.1.trajectory", self.failed_codes(self.validate()))

    def test_duplicate_artifact_json_keys_rejected(self):
        self.replace_artifact(0, "trajectory", b'{"messages":[],"messages":[{"content":"event"}]}')
        self.assertIn("artifacts.format.1.trajectory", self.failed_codes(self.validate()))

    def test_code_is_verified_as_opaque_bytes(self):
        self.replace_artifact(0, "code", b"raise RuntimeError('USER CODE MUST NEVER EXECUTE')\n")
        self.assertTrue(self.validate()["valid"])


if __name__ == "__main__":
    unittest.main()
