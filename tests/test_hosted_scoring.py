"""Synthetic labels only; no benchmark private data is needed for these tests."""
import csv
import gzip
import importlib.util
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from space.service import score_submission, SubmissionRejected, EvaluatorUnavailable


class HostedScoringTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.private = self.root / "private"

    def call(self, task, path):
        # Fixtures intentionally have fewer rows than the real benchmark.
        with patch("space.service.validate", return_value={"format_valid": True}):
            return score_submission(task, path, private_root=self.private)

    def test_hstu_and_tiger_known_ranks(self):
        submission = self.root / "submission.npz"
        np.savez(submission, user_ids=np.array([1, 2]), top_items=np.array([range(1, 11), range(1, 11)]))
        for task in ("hstu", "tiger"):
            label = self.private / task / "beauty"
            label.mkdir(parents=True)
            np.save(label / "expected_user_ids.npy", [1, 2])
            np.save(label / "catalog.npy", np.arange(1, 12))
            np.save(label / "test_targets.npy", [1, 11])
            self.assertEqual(self.call(task, submission)["metrics"]["ndcg_at_10"], .5)
        np.savez(submission, user_ids=np.array([2, 1]), top_items=np.array([range(1, 11), range(1, 11)]))
        with self.assertRaises(SubmissionRejected) as e:
            self.call("hstu", submission)
        self.assertNotIn(str(label), str(e.exception))

    def test_ctr_tasks_perfect_auc(self):
        ids = np.arange(4, dtype=np.int64)
        labels = np.array([0, 1, 0, 1])
        scores = [.1, .9, .2, .8]
        rank = self.private / "rankmixer/criteo5m"
        rank.mkdir(parents=True)
        np.save(rank / "expected_sample_ids.npy", ids)
        np.save(rank / "test_labels.npy", labels)
        file = self.root / "submission.npz"
        np.savez(file, sample_ids=ids, scores=scores)
        self.assertEqual(self.call("rankmixer", file)["metrics"]["auc"], 1)
        wu = self.private / "wukong/movielenslatest"
        wu.mkdir(parents=True)
        with (wu / "test_data_labeled.csv").open("w") as f:
            f.write("sample_id,label\n" + "".join(f"{i},{y}\n" for i, y in zip(ids, labels)))
        file = self.root / "submission.csv.gz"
        with gzip.open(file, "wt") as f:
            f.write("sample_id,score\n" + "".join(f"{i},{s}\n" for i, s in zip(ids, scores)))
        self.assertEqual(self.call("wukong", file)["metrics"]["auc"], 1)
        one = self.private / "onetrans/kuaisar_v2"
        one.mkdir(parents=True)
        np.savez(one / "labels.npz", sample_ids=ids, click=labels, user_id=[1, 1, 2, 2])
        file = self.root / "submission.csv"
        file.write_text("sample_id,prediction\n" + "".join(f"{i},{s}\n" for i, s in zip(ids, scores)))
        result = self.call("onetrans", file)
        self.assertEqual(result["metrics"]["auc"], 1)
        self.assertEqual(result["metrics"]["uauc"], 1)

    def test_intsr_ties_use_expected_discount(self):
        label = self.private / "intsr/Amazon"
        label.mkdir(parents=True)
        candidates = np.tile(np.arange(1, 101), (2, 1)).astype(np.int32)
        np.savez(label / "expected_submission.npz", sample_ids=[1, 2], task_ids=[0, 1], candidate_items=candidates)
        (label / "test_data_labeled.csv").write_text("sample_id,task,positive_item_id\n1,rec,1\n2,src,1\n")
        file = self.root / "submission.npz"
        np.savez(file, sample_ids=[1, 2], task_ids=[0, 1], candidate_items=candidates, scores=np.ones((2, 100)))
        result = self.call("intsr", file)
        expected = sum(1 / math.log2(i + 1) for i in range(1, 11)) / 100
        self.assertAlmostEqual(result["metrics"]["macro_ndcg_at_10"], expected)

    def test_missing_evaluator_has_no_fabricated_zero(self):
        file = self.root / "submission.npz"
        np.savez(file, sample_ids=[1], scores=[.5])
        with self.assertRaises(EvaluatorUnavailable):
            self.call("rankmixer", file)

    def test_auc_tie_convention_matches_scorers(self):
        for task in ("rankmixer", "wukong", "onetrans"):
            source = Path(__file__).resolve().parents[1] / "space/scorers" / f"{task}.py"
            spec = importlib.util.spec_from_file_location("scorer", source)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            auc = getattr(module, "auc_score", getattr(module, "binary_auc", None))
            self.assertEqual(auc(np.array([0, 1]), np.array([.5, .5])), .5)


if __name__ == "__main__":
    unittest.main()
