"""Read GitHub review evidence; never execute code or trust approval fields from a PR."""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import HTTPRedirectHandler, Request, build_opener

MAX_SUBMISSION_BYTES = 2 * 1024**2
MAX_API_BYTES = 8 * 1024**2
PATH_RE = re.compile(r"leaderboard/submissions/(?=[a-z0-9-]{1,80}\.json\Z)[a-z0-9]+(?:-[a-z0-9]+)*\.json\Z")
REPO_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9-]{0,38}/[A-Za-z0-9_.-]{1,100}\Z")
SHA_RE = re.compile(r"[0-9a-f]{40}\Z")
PUBLISH_PERMISSIONS = {"write", "maintain", "admin"}


class ReviewError(ValueError):
    """A bounded error code safe to put in a CI log."""


def require(value, code):
    if not value:
        raise ReviewError(code)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def _unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate_json_key")
        result[key] = value
    return result


def parse_submission(data):
    require(0 < len(data) <= MAX_SUBMISSION_BYTES, "submission_size_limit")
    try:
        value = json.loads(data, object_pairs_hook=_unique,
                           parse_constant=lambda _: (_ for _ in ()).throw(ReviewError("invalid_json_number")))
    except (UnicodeError, json.JSONDecodeError):
        raise ReviewError("invalid_submission_json") from None
    require(isinstance(value, dict), "submission_must_be_object")
    return value


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class GitHubAPI:
    def __init__(self, token):
        require(isinstance(token, str) and bool(token), "github_token_required")
        self._token = token
        self._opener = build_opener(_NoRedirect)

    def get(self, endpoint):
        require(endpoint.startswith("/repos/") and not endpoint.startswith("//"), "invalid_api_endpoint")
        request = Request("https://api.github.com" + endpoint, headers={
            "Authorization": "Bearer " + self._token,
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2026-03-10",
            "User-Agent": "autorse-leaderboard-review",
        })
        try:
            with self._opener.open(request, timeout=30) as response:
                data = response.read(MAX_API_BYTES + 1)
            require(len(data) <= MAX_API_BYTES, "github_api_response_too_large")
            return json.loads(data)
        except HTTPError as error:
            raise ReviewError("github_api_http_" + str(error.code)) from None
        except (URLError, TimeoutError, json.JSONDecodeError):
            raise ReviewError("github_api_unavailable") from None

    def pages(self, endpoint):
        result = []
        separator = "&" if "?" in endpoint else "?"
        for page in range(1, 101):
            rows = self.get(endpoint + separator + f"per_page=100&page={page}")
            require(isinstance(rows, list) and len(rows) <= 100, "invalid_github_page")
            result.extend(rows)
            if len(rows) < 100:
                return result
        raise ReviewError("github_pagination_limit")


def _repo(value):
    require(isinstance(value, str) and REPO_RE.fullmatch(value), "invalid_repository")
    require(value.split("/")[1] not in (".", ".."), "invalid_repository")
    return value


def _sha(value):
    require(isinstance(value, str) and SHA_RE.fullmatch(value), "invalid_commit_sha")
    return value


def assert_main_tip(api, repository, main_sha):
    _repo(repository)
    _sha(main_sha)
    current = api.get(f"/repos/{repository}/git/ref/heads/main")
    require(current.get("object", {}).get("sha") == main_sha, "stale_main_snapshot")


def _pull(api, repository, number):
    require(type(number) is int and number > 0, "invalid_pr_number")
    result = api.get(f"/repos/{repository}/pulls/{number}")
    require(result.get("base", {}).get("repo", {}).get("full_name", "").lower() == repository.lower()
            and result.get("base", {}).get("ref") == "main", "pr_wrong_target")
    return result


def _files(api, repository, pr):
    count = pr.get("changed_files")
    require(type(count) is int and 0 < count <= 3000, "pr_file_count_limit")
    result = api.pages(f"/repos/{repository}/pulls/{pr['number']}/files")
    require(len(result) == count, "incomplete_pr_file_listing")
    return result


def _only_added_submission(files):
    require(len(files) == 1, "submission_pr_must_add_only_one_json")
    entry = files[0]
    path = entry.get("filename", "")
    require(entry.get("status") == "added" and PATH_RE.fullmatch(path)
            and not entry.get("previous_filename"), "submission_pr_must_add_only_one_json")
    _sha(entry.get("sha"))
    return path, entry["sha"]


def _blob(api, repository, blob_sha):
    payload = api.get(f"/repos/{repository}/git/blobs/{_sha(blob_sha)}")
    require(payload.get("encoding") == "base64" and type(payload.get("size")) is int
            and 0 < payload["size"] <= MAX_SUBMISSION_BYTES, "submission_size_limit")
    encoded = payload.get("content", "")
    require(isinstance(encoded, str) and len(encoded) <= MAX_SUBMISSION_BYTES * 2, "submission_size_limit")
    try:
        data = base64.b64decode("".join(encoded.split()), validate=True)
    except (ValueError, TypeError):
        raise ReviewError("invalid_blob_encoding") from None
    require(len(data) == payload["size"], "blob_size_mismatch")
    actual_blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
    require(actual_blob == blob_sha, "blob_hash_mismatch")
    parse_submission(data)
    return data


def fetch_pr(api, repository, number, head_sha, output_dir, *, allow_maintenance=False):
    """Copy only the new JSON blob, never the PR checkout, to a temporary folder."""
    repository, head_sha = _repo(repository), _sha(head_sha)
    pr = _pull(api, repository, number)
    require(pr.get("head", {}).get("sha") == head_sha, "pr_head_changed")
    files = _files(api, repository, pr)
    if not any(str(item.get("filename", "")).startswith("leaderboard/submissions/") for item in files):
        require(allow_maintenance, "not_a_submission_pr")
        return {"is_submission": False, "reason": "maintenance_pr", "head_sha": head_sha}
    path, blob_sha = _only_added_submission(files)
    try:
        data = _blob(api, repository, blob_sha)
    except ReviewError as error:
        # A fork's not-yet-merged blob can be outside the base repository's object view.
        if str(error) != "github_api_http_404":
            raise
        fork = (pr.get("head", {}).get("repo") or {})
        require(fork.get("private") is False, "fork_blob_not_public")
        data = _blob(api, _repo(fork.get("full_name")), blob_sha)
    output_dir = Path(output_dir)
    require(parse_submission(data).get("id") == Path(path).stem, "submission_id_filename_mismatch")
    require(_pull(api, repository, number).get("head", {}).get("sha") == head_sha, "pr_head_changed")
    require(not output_dir.is_symlink(), "unsafe_output_directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / Path(path).name
    require(not target.exists() and not target.is_symlink(), "output_already_exists")
    target.write_bytes(data)
    return {"is_submission": True, "submission_path": path, "local_path": str(target),
            "submission_sha256": sha256(data), "head_sha": head_sha, "pr_number": number}


def approved_reviews(api, repository, pr):
    """Latest submitted decision per reviewer; comments do not revoke a decision."""
    head_sha = _sha(pr.get("head", {}).get("sha"))
    author = str(pr.get("user", {}).get("login", "")).lower()
    decisions = {}
    for review in api.pages(f"/repos/{repository}/pulls/{pr['number']}/reviews"):
        if review.get("state") not in ("APPROVED", "CHANGES_REQUESTED", "DISMISSED"):
            continue
        login = str(review.get("user", {}).get("login", ""))
        if not login or not review.get("submitted_at"):
            continue
        previous = decisions.get(login.lower())
        order = (review["submitted_at"], review.get("id", 0))
        if previous is None or order > (previous["submitted_at"], previous.get("id", 0)):
            decisions[login.lower()] = review
    approved = []
    for login, review in sorted(decisions.items()):
        if (login == author or review["state"] != "APPROVED" or review.get("commit_id") != head_sha
                or review.get("user", {}).get("type") != "User"):
            continue
        name = review["user"]["login"]
        permission = api.get(f"/repos/{repository}/collaborators/{quote(name, safe='')}/permission")
        role = permission.get("role_name")
        level = permission.get("permission")
        if level not in PUBLISH_PERMISSIONS and role not in PUBLISH_PERMISSIONS:
            continue
        approved.append({"reviewer": name, "review_id": review["id"],
                         "submitted_at": review["submitted_at"], "commit_id": head_sha,
                         "permission": role if role in PUBLISH_PERMISSIONS else level})
    require(approved, "qualified_current_head_approval_required")
    return approved


def _git(*args, cwd):
    try:
        return subprocess.check_output(["git", *args], cwd=cwd, stderr=subprocess.DEVNULL,
                                       timeout=30).decode().strip()
    except (subprocess.SubprocessError, UnicodeError):
        raise ReviewError("git_history_unavailable") from None


def _ancestor(older, newer, cwd):
    try:
        subprocess.run(["git", "merge-base", "--is-ancestor", _sha(older), _sha(newer)], cwd=cwd,
                       check=True, timeout=30, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False
    except subprocess.SubprocessError:
        raise ReviewError("git_history_unavailable") from None


def _approve_file(api, repository, main_sha, repository_root, path, data):
    introduction = _git("log", "--diff-filter=A", "--format=%H", "-1", main_sha, "--", path,
                        cwd=repository_root)
    _sha(introduction)
    candidates = api.pages(f"/repos/{repository}/commits/{introduction}/pulls")
    approvals = []
    for candidate in candidates:
        try:
            pr = _pull(api, repository, candidate.get("number"))
        except ReviewError as error:
            if str(error) == "pr_wrong_target":
                continue
            raise
        if not pr.get("merged_at") or pr.get("state") != "closed":
            continue
        merge_sha = pr.get("merge_commit_sha")
        if not SHA_RE.fullmatch(str(merge_sha)) or not _ancestor(merge_sha, main_sha, repository_root):
            continue
        try:
            canonical, blob_sha = _only_added_submission(_files(api, repository, pr))
        except ReviewError as error:
            if str(error) == "submission_pr_must_add_only_one_json":
                continue
            raise
        if canonical != path:
            continue
        if _blob(api, repository, blob_sha) != data:
            continue
        try:
            reviews = approved_reviews(api, repository, pr)
        except ReviewError as error:
            if str(error) == "qualified_current_head_approval_required":
                continue
            raise
        approvals.append({"submission_path": path, "submission_sha256": sha256(data),
                          "pr_number": pr["number"], "merge_commit_sha": merge_sha,
                          "head_sha": pr["head"]["sha"], "author": pr["user"]["login"],
                          "approved_reviews": reviews})
    require(len(approvals) == 1, "unique_approved_merge_required")
    return approvals[0]


def collect(api, repository, main_sha, submissions_dir, approved_dir):
    """Fail closed on any main JSON lacking an approved, byte-identical merge."""
    repository, main_sha = _repo(repository), _sha(main_sha)
    assert_main_tip(api, repository, main_sha)
    repository_root = Path(_git("rev-parse", "--show-toplevel", cwd=Path.cwd())).resolve()
    require(_git("rev-parse", "HEAD", cwd=repository_root) == main_sha, "checkout_not_main_snapshot")
    source = Path(submissions_dir).resolve()
    require(source == repository_root / "leaderboard/submissions" and not Path(submissions_dir).is_symlink(),
            "submissions_directory_mismatch")
    require(source.is_dir(), "submissions_directory_missing")
    require(not any(path.is_symlink() for path in source.rglob("*")), "submission_symlink_refused")
    files = sorted(source.rglob("*.json"))
    records = []
    payloads = []
    for path in files:
        canonical = path.relative_to(repository_root).as_posix()
        require(PATH_RE.fullmatch(canonical) and path.is_file(), "invalid_submission_path")
        require(0 < path.stat().st_size <= MAX_SUBMISSION_BYTES, "submission_size_limit")
        data = path.read_bytes()
        submission = parse_submission(data)
        require(submission.get("id") == path.stem, "submission_id_filename_mismatch")
        require(submission.get("kind") == "benchmark", "test_or_nonbenchmark_cannot_publish")
        # The checkout must still be exactly its tracked main version, not a dirty working copy.
        tracked_size = _git("cat-file", "-s", f"{main_sha}:{canonical}", cwd=repository_root)
        require(tracked_size.isdigit() and int(tracked_size) <= MAX_SUBMISSION_BYTES, "submission_size_limit")
        tracked = subprocess.check_output(["git", "show", f"{main_sha}:{canonical}"],
                                          cwd=repository_root, stderr=subprocess.DEVNULL, timeout=30)
        require(tracked == data, "working_submission_differs_from_main")
        records.append(_approve_file(api, repository, main_sha, repository_root, canonical, data))
        payloads.append((path.name, data))
    assert_main_tip(api, repository, main_sha)
    approved_dir = Path(approved_dir)
    require(not approved_dir.exists() and not approved_dir.is_symlink(), "approved_directory_already_exists")
    approved_dir.mkdir(parents=True)
    for name, data in payloads:
        (approved_dir / name).write_bytes(data)
    return {"schema_version": 1, "repository": repository, "main_sha": main_sha,
            "generated_at": datetime.now(timezone.utc).isoformat(), "approvals": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="command", required=True)
    extract = modes.add_parser("fetch-pr")
    extract.add_argument("--repository", required=True)
    extract.add_argument("--pr", required=True, type=int)
    extract.add_argument("--head-sha", required=True)
    extract.add_argument("--output-dir", required=True, type=Path)
    extract.add_argument("--report", required=True, type=Path)
    extract.add_argument("--allow-maintenance", action="store_true")
    ledger = modes.add_parser("collect")
    ledger.add_argument("--repository", required=True)
    ledger.add_argument("--sha", required=True)
    ledger.add_argument("--submissions-dir", required=True, type=Path)
    ledger.add_argument("--approved-dir", required=True, type=Path)
    ledger.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        api = GitHubAPI(os.environ.get("GITHUB_TOKEN", ""))
        if args.command == "fetch-pr":
            result = fetch_pr(api, args.repository, args.pr, args.head_sha, args.output_dir,
                              allow_maintenance=args.allow_maintenance)
            output = args.report
        else:
            result = collect(api, args.repository, args.sha, args.submissions_dir, args.approved_dir)
            output = args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"ok": True, "is_submission": result.get("is_submission"),
                          "approved_count": len(result.get("approvals", []))}))
        return 0
    except Exception as error:
        code = str(error) if isinstance(error, ReviewError) else type(error).__name__
        print(json.dumps({"ok": False, "error_code": code}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
