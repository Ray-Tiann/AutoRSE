import base64
from contextlib import contextmanager
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from autorse_leaderboard import github_review as review

REPO = "example/benchmark"
HEAD = "a" * 40
PATH = "leaderboard/submissions/example-agent.json"


def blob_sha(data):
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def pr_payload(head=HEAD):
    return {"number": 7, "changed_files": 1, "state": "closed", "merged_at": "2026-10-30T00:00:00Z",
            "merge_commit_sha": head, "head": {"sha": head},
            "base": {"ref": "main", "repo": {"full_name": REPO}}, "user": {"login": "author"}}


def approval(head=HEAD, state="APPROVED", identifier=1, login="maintainer", when="2026-10-30T00:00:00Z"):
    return {"id": identifier, "commit_id": head, "state": state, "submitted_at": when,
            "user": {"login": login, "type": "User"}}


class FakeAPI(review.GitHubAPI):
    def __init__(self, data, head=HEAD):
        self.data = data
        self.head = head
        self.pr = pr_payload(head)
        self.files = [{"filename": PATH, "status": "added", "sha": blob_sha(data)}]
        self.reviews = [approval(head)]
        self.permission = {"permission": "write", "role_name": "maintain"}
        self.tip = head
        self.calls = []
        self.candidates = [{"number": 7}]

    def get(self, path):
        self.calls.append(path)
        root = f"/repos/{REPO}"
        if path == root + "/git/ref/heads/main":
            return {"object": {"sha": self.tip}}
        if path == root + "/pulls/7":
            return copy.deepcopy(self.pr)
        if path.startswith(root + "/pulls/7/files?"):
            return copy.deepcopy(self.files)
        if path.startswith(root + "/pulls/7/reviews?"):
            return copy.deepcopy(self.reviews)
        if path.startswith(root + "/collaborators/"):
            return copy.deepcopy(self.permission)
        if path.startswith(root + "/commits/") and "/pulls?" in path:
            return self.candidates
        if path == root + "/git/blobs/" + blob_sha(self.data):
            return {"encoding": "base64", "size": len(self.data),
                    "content": base64.b64encode(self.data).decode()}
        raise AssertionError("Unexpected synthetic route: " + path)


@contextmanager
def repository(data=None):
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp).resolve()
        def git(*args):
            return subprocess.check_output(["git", *args], cwd=root, stderr=subprocess.DEVNULL).decode().strip()
        git("init", "-b", "main")
        git("config", "user.name", "Synthetic Test")
        git("config", "user.email", "synthetic@example.test")
        (root / "README.md").write_text("synthetic")
        source = root / "leaderboard/submissions"
        source.mkdir(parents=True)
        (source / ".gitkeep").write_text("")
        git("add", ".")
        git("commit", "-m", "initial")
        if data is not None:
            (root / PATH).write_bytes(data)
            git("add", PATH)
            git("commit", "-m", "add submission")
        old = Path.cwd()
        os.chdir(root)
        try:
            yield root, source, git("rev-parse", "HEAD"), git
        finally:
            os.chdir(old)


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.data = json.dumps({"id": "example-agent", "kind": "benchmark", "command": "$(do-not-execute)"}).encode()

    def test_fetch_copies_only_json_data_and_binds_sha(self):
        api = FakeAPI(self.data)
        with tempfile.TemporaryDirectory() as temp:
            result = review.fetch_pr(api, REPO, 7, HEAD, Path(temp) / "input")
            self.assertEqual(Path(result["local_path"]).read_bytes(), self.data)
            self.assertEqual(result["submission_path"], PATH)
            self.assertEqual(result["submission_sha256"], hashlib.sha256(self.data).hexdigest())

    def test_fetch_rejects_mixed_edit_rename_and_head_changes(self):
        cases = [
            [{"filename": PATH, "status": "modified", "sha": blob_sha(self.data)}],
            [{"filename": PATH, "status": "renamed", "sha": blob_sha(self.data), "previous_filename": "old.json"}],
            [{"filename": PATH, "status": "added", "sha": blob_sha(self.data)},
             {"filename": ".github/workflows/publish.yml", "status": "modified", "sha": HEAD}],
        ]
        for files in cases:
            api = FakeAPI(self.data); api.files = files; api.pr["changed_files"] = len(files)
            with tempfile.TemporaryDirectory() as temp, self.assertRaises(review.ReviewError):
                review.fetch_pr(api, REPO, 7, HEAD, Path(temp) / "input")
        api = FakeAPI(self.data); api.pr["head"]["sha"] = "b" * 40
        with tempfile.TemporaryDirectory() as temp, self.assertRaises(review.ReviewError):
            review.fetch_pr(api, REPO, 7, HEAD, Path(temp) / "input")

    def test_maintenance_has_no_submission_eligibility(self):
        api = FakeAPI(self.data)
        api.files = [{"filename": "README.md", "status": "modified", "sha": HEAD}]
        with tempfile.TemporaryDirectory() as temp:
            result = review.fetch_pr(api, REPO, 7, HEAD, Path(temp) / "input", allow_maintenance=True)
            self.assertFalse(result["is_submission"])
            self.assertFalse((Path(temp) / "input").exists())
            with self.assertRaises(review.ReviewError):
                review.fetch_pr(api, REPO, 7, HEAD, Path(temp) / "input")

    def test_id_must_match_filename_and_slug_is_bounded(self):
        for identifier in (None, "another-agent"):
            api = FakeAPI(json.dumps({"id": identifier, "kind": "benchmark"}).encode())
            with tempfile.TemporaryDirectory() as temp, self.assertRaisesRegex(
                    review.ReviewError, "submission_id_filename_mismatch"):
                review.fetch_pr(api, REPO, 7, HEAD, Path(temp) / "input")
        for name in ("example_agent", "example--agent", "example-agent-", "a" * 81, "../example-agent"):
            api = FakeAPI(self.data)
            api.files[0]["filename"] = "leaderboard/submissions/" + name + ".json"
            with tempfile.TemporaryDirectory() as temp, self.assertRaises(review.ReviewError):
                review.fetch_pr(api, REPO, 7, HEAD, Path(temp) / "input")

    def test_latest_approval_not_revoked_by_comment(self):
        api = FakeAPI(self.data)
        api.reviews.append(approval(state="COMMENTED", identifier=2, when="2026-10-30T00:01:00Z"))
        result = review.approved_reviews(api, REPO, api.pr)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["permission"], "maintain")

    def test_rejects_dismissed_requested_stale_self_bot_or_read_only(self):
        for mutation in ("dismissed", "requested", "stale", "self", "bot", "read"):
            api = FakeAPI(self.data)
            if mutation in ("dismissed", "requested"):
                api.reviews.append(approval(state="DISMISSED" if mutation == "dismissed" else "CHANGES_REQUESTED",
                                            identifier=2, when="2026-10-30T00:01:00Z"))
            if mutation == "stale": api.reviews[0]["commit_id"] = "b" * 40
            if mutation == "self": api.reviews[0]["user"]["login"] = "author"
            if mutation == "bot": api.reviews[0]["user"]["type"] = "Bot"
            if mutation == "read": api.permission = {"permission": "read", "role_name": "read"}
            with self.subTest(case=mutation), self.assertRaises(review.ReviewError):
                review.approved_reviews(api, REPO, api.pr)

    def test_api_pagination_fetches_all_review_pages(self):
        class Paged(FakeAPI):
            def get(self, path):
                if "/reviews?" in path:
                    if path.endswith("page=1"):
                        return [approval(state="COMMENTED", identifier=i + 1) for i in range(100)]
                    if path.endswith("page=2"):
                        return [approval(identifier=101)]
                return super().get(path)
        api = Paged(self.data)
        self.assertEqual(review.approved_reviews(api, REPO, api.pr)[0]["review_id"], 101)

    def test_collect_binds_approved_current_bytes_and_main_snapshot(self):
        with repository(self.data) as (root, source, head, git):
            api = FakeAPI(self.data, head)
            result = review.collect(api, REPO, head, source, root / "approved")
            entry = result["approvals"][0]
            self.assertEqual(entry["submission_sha256"], hashlib.sha256(self.data).hexdigest())
            self.assertEqual(entry["submission_path"], PATH)
            self.assertEqual(entry["author"], "author")
            self.assertEqual((root / "approved/example-agent.json").read_bytes(), self.data)
            self.assertEqual(result["main_sha"], head)

    def test_collect_refuses_test_kind_direct_push_or_unapproved(self):
        for mode in ("test", "direct", "unapproved", "stale-main"):
            data = json.dumps({"id": "example-agent", "kind": "test" if mode == "test" else "benchmark"}).encode()
            with repository(data) as (root, source, head, git):
                api = FakeAPI(data, head)
                if mode == "direct": api.candidates = []
                if mode == "unapproved": api.reviews = []
                if mode == "stale-main": api.tip = "b" * 40
                with self.subTest(mode=mode), self.assertRaises(review.ReviewError):
                    review.collect(api, REPO, head, source, root / "approved")
                self.assertFalse((root / "approved").exists())

    def test_collect_refuses_dirty_or_later_modified_submission(self):
        for committed in (False, True):
            with repository(self.data) as (root, source, original_head, git):
                api = FakeAPI(self.data, original_head)
                (root / PATH).write_text('{"id":"example-agent","kind":"benchmark","changed":true}')
                if committed:
                    git("add", PATH);git("commit", "-m", "unreviewed edit")
                    head = git("rev-parse", "HEAD");api.tip = head
                else:
                    head = original_head
                with self.assertRaises(review.ReviewError):
                    review.collect(api, REPO, head, source, root / "approved")

    def test_no_json_produces_empty_ledger_without_a_publication_claim(self):
        with repository() as (root, source, head, git):
            result = review.collect(FakeAPI(self.data, head), REPO, head, source, root / "approved")
            self.assertEqual(result["approvals"], [])
            self.assertEqual(list((root / "approved").iterdir()), [])

    def test_json_and_repository_input_bounds(self):
        for raw in (b"", b'{"kind":"benchmark","kind":"test"}', b'{"x":NaN}', b"[]"):
            with self.assertRaises(review.ReviewError):review.parse_submission(raw)
        for repo in ("evil//repo", "../bad", "https://example.com/repo"):
            with self.assertRaises(review.ReviewError):review._repo(repo)
        with self.assertRaises(review.ReviewError):review.parse_submission(b"x" * (review.MAX_SUBMISSION_BYTES + 1))


if __name__ == "__main__":
    unittest.main()
