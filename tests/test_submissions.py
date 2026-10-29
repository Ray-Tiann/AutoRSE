import gzip
from pathlib import Path
import tempfile
import unittest
import zipfile
from unittest.mock import patch
import numpy as np
from autorse.submissions import validate


class SubmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_top10_valid_and_duplicate_rejected(self):
        spec = {"file": "submission.npz", "rows": 2, "metric": "ndcg_at_10",
                "arrays": {"user_ids": (2,), "top_items": (2, 10)}}
        path = self.root / "submission.npz"
        with patch.dict("autorse.submissions.TASKS", {"hstu": spec}):
            np.savez(path, user_ids=np.arange(2), top_items=np.tile(np.arange(1, 11), (2, 1)))
            self.assertFalse(validate("hstu", path)["hidden_test_scored"])
            np.savez(path, user_ids=np.arange(2), top_items=np.ones((2, 10), dtype=int))
            with self.assertRaises(ValueError):
                validate("hstu", path)

    def test_object_array_and_wrong_shape_rejected(self):
        spec = {"file": "submission.npz", "rows": 2, "metric": "auc",
                "arrays": {"sample_ids": (2,), "scores": (2,)}}
        path = self.root / "submission.npz"
        with patch.dict("autorse.submissions.TASKS", {"rankmixer": spec}):
            for scores in (np.array([object(), object()]), np.array([.3]), np.array([float("nan"), .5])):
                np.savez(path, sample_ids=np.arange(2), scores=scores)
                with self.assertRaises(ValueError):
                    validate("rankmixer", path)

    def test_duplicate_and_traversal_zip_members_rejected(self):
        path = self.root / "submission.npz"
        with zipfile.ZipFile(path, "w") as z:
            z.writestr("../../secret.npy", b"invalid")
        with self.assertRaises(ValueError):
            validate("hstu", path)

    def test_gzip_csv_bounded_and_finite(self):
        spec = {"file": "submission.csv.gz", "rows": 2, "metric": "auc", "columns": ["sample_id", "score"]}
        path = self.root / "submission.csv.gz"
        with patch.dict("autorse.submissions.TASKS", {"wukong": spec}):
            for content, valid in [("sample_id,score\n1,0.5\n2,0.7\n", True),
                                   ("sample_id,score\n1,nan\n2,0.2\n", False),
                                   ("sample_id,score\n1,0.5\n", False),
                                   ("sample_id,score\n" + "x" * 5000, False)]:
                with gzip.open(path, "wt") as f:
                    f.write(content)
                if valid:
                    self.assertTrue(validate("wukong", path)["format_valid"])
                else:
                    with self.assertRaises(ValueError):
                        validate("wukong", path)

    def test_symlink_rejected(self):
        target = self.root / "real.npz"
        target.write_bytes(b"not a submission")
        link = self.root / "submission.npz"
        link.symlink_to(target)
        with self.assertRaises(ValueError):
            validate("hstu", link)


if __name__ == "__main__":
    unittest.main()
