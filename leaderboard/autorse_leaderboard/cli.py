"""Prepare, validate and publish reviewed AutoRSE submissions."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
from .protocol import TASK_ORDER, DATA_REVISION, PUBLIC_BUNDLE_SHA256


def read_json(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 2 * 1024 * 1024:
        raise ValueError("invalid_or_oversized_json")
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate_json_key")
            result[key] = value
        return result
    def constant(value):
        raise ValueError("nonfinite_json")
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs, parse_constant=constant)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def template():
    artifact = {"repo_id": "YOUR-HF-USERNAME/YOUR-PUBLIC-EVIDENCE-DATASET", "revision": "", "path": "",
                "sha256": "", "size": 0}
    return {"schema_version": 1, "id": "your-agent-model-version", "kind": "benchmark",
            "configuration": {"name": "", "agent": "", "harness": "", "agent_version": "",
                              "model": "", "model_version": "", "framework_commit": "", "code_url": ""},
            "protocol": {"data_revision": DATA_REVISION, "public_bundle_sha256": PUBLIC_BUNDLE_SHA256,
                         "research_budget_seconds": 43200, "gpus_per_run": 1},
            "receipts": [], "runs": [
                {"task": task, "trial_index": trial, "run_id": f"{task}-trial-{trial}", "seed": 2023 + trial,
                 "receipt_evaluation_id": "", "prediction_sha256": "", "code_commit": "",
                 "agent_started_at": "", "agent_finished_at": "", "gpus": 1,
                 "artifacts": {name: dict(artifact) for name in ("code", "report", "trajectory", "config", "provenance")}}
                for trial in range(1, 4) for task in TASK_ORDER],
            "acknowledgements": {"public_evidence": False, "protocol_followed": False}}


def check_approval(path, submission, ledger):
    if ledger.get("schema_version") != 1 or ledger.get("repository") != "Ray-Tiann/AutoRSE":
        raise ValueError("invalid_approval_ledger")
    main_sha = ledger.get("main_sha", "")
    if not re.fullmatch(r"[0-9a-f]{40}", main_sha):
        raise ValueError("approval_main_sha_missing")
    if os.environ.get("GITHUB_SHA") and os.environ["GITHUB_SHA"] != main_sha:
        raise ValueError("stale_approval_ledger")
    matches = [entry for entry in ledger.get("approvals", [])
               if entry.get("submission_path") == f"leaderboard/submissions/{submission['id']}.json"]
    if len(matches) != 1:
        raise ValueError("submission_requires_maintainer_approval")
    entry = matches[0]
    if not re.fullmatch(r"[0-9a-f]{40}", entry.get("head_sha", "")):
        raise ValueError("approved_head_missing")
    if entry.get("submission_sha256") != hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError("approved_submission_bytes_changed")
    if not entry.get("approved_reviews") or not all(
            review.get("permission") in {"write", "maintain", "admin"}
            and isinstance(review.get("reviewer"), str) and review["reviewer"]
            and review["reviewer"] != entry.get("author")
            and review.get("commit_id") == entry.get("head_sha") for review in entry["approved_reviews"]):
        raise ValueError("qualified_review_missing")
    if type(entry.get("pr_number")) is not int or entry["pr_number"] <= 0:
        raise ValueError("approved_pr_missing")
    return entry


def assert_current_main(repository, commit):
    import httpx
    if repository != "Ray-Tiann/AutoRSE" or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("invalid_publication_source")
    headers = {"Accept": "application/vnd.github+json"}
    github_token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if github_token:
        headers["Authorization"] = "Bearer " + github_token
    response = httpx.get(f"https://api.github.com/repos/{repository}/git/ref/heads/main", headers=headers, timeout=30)
    response.raise_for_status()
    if response.json()["object"]["sha"] != commit:
        raise ValueError("stale_source_snapshot_refused")


def check_publisher(repo_id, commit):
    from datetime import datetime, timezone
    from huggingface_hub import HfApi, CommitOperationAdd
    if repo_id != "Ray0311/AutoRSE-Leaderboard":
        raise ValueError("unexpected_publish_destination")
    assert_current_main("Ray-Tiann/AutoRSE", commit)
    api = HfApi()
    info = api.space_info(repo_id)
    record = {"schema_version": 1, "repository": "Ray-Tiann/AutoRSE", "source_commit": commit,
              "checked_at": datetime.now(timezone.utc).isoformat(), "status": "publisher_access_verified",
              "note": "Write-access health check only; no benchmark results were added."}
    operation = CommitOperationAdd(path_in_repo="publisher-status.json", path_or_fileobj=(
        json.dumps(record, indent=2) + "\n").encode())
    result = api.create_commit(repo_id=repo_id, repo_type="space", parent_commit=info.sha, operations=[operation],
                               commit_message="Verify main-branch leaderboard publisher access")
    print(json.dumps({"publisher_access_verified": True, "hf_commit": result.oid, "results_added": 0}))


def publish(args):
    from .validation import validate_submission
    from .ranking import build_leaderboard
    from .artifacts import PublicArtifactReader
    keys, ledger = read_json(args.keys), read_json(args.approval_ledger)
    folder, output = Path(args.submissions_dir), Path(args.output_dir)
    submissions = sorted(folder.glob("*.json"))
    if not submissions:
        raise ValueError("no_approved_submissions_no_publish")
    summaries, approved = [], []
    for path in submissions:
        submission = read_json(path)
        entry = check_approval(path, submission, ledger)
        report = validate_submission(submission, keys, artifact_reader=PublicArtifactReader())
        if not report.get("valid") or not report.get("publishable") or submission.get("kind") != "benchmark":
            raise ValueError("unverified_submission_refused")
        summary = report["summary"]
        summary["source_url"] = f"https://github.com/{ledger['repository']}/blob/{ledger['main_sha']}/leaderboard/submissions/{submission['id']}.json"
        summaries.append(summary)
        approved.append((path, entry))
    data = build_leaderboard(summaries)
    # Bind a reproducible source snapshot, not a wall-clock-only freshness indicator.
    data.setdefault("metadata", {}).update(source_repository=ledger["repository"], source_commit=ledger["main_sha"])
    write_json(output / "community-leaderboard.json", data)
    write_json(output / "approval-ledger.json", ledger)
    for path, _entry in approved:
        write_json(output / "submissions" / path.name, read_json(path))
    if args.dry_run:
        print(json.dumps({"dry_run": True, "approved_entries": len(summaries)}))
        return
    if args.repo_id != "Ray0311/AutoRSE-Leaderboard":
        raise ValueError("unexpected_publish_destination")
    from huggingface_hub import HfApi, CommitOperationAdd
    assert_current_main(ledger["repository"], ledger["main_sha"])
    api = HfApi()
    info = api.space_info(args.repo_id)
    operations = [CommitOperationAdd(path_in_repo="community-leaderboard.json", path_or_fileobj=str(output / "community-leaderboard.json")),
                  CommitOperationAdd(path_in_repo="approval-ledger.json", path_or_fileobj=str(output / "approval-ledger.json"))]
    operations += [CommitOperationAdd(path_in_repo="submissions/" + path.name,
                                     path_or_fileobj=str(output / "submissions" / path.name)) for path, _ in approved]
    result = api.create_commit(repo_id=args.repo_id, repo_type="space", parent_commit=info.sha,
        operations=operations, commit_message="Publish reviewed AutoRSE community results from " + ledger["main_sha"][:12])
    print(json.dumps({"published_entries": len(summaries), "hf_commit": result.oid}))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("template")
    p.add_argument("--output", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--metadata", required=True)
    p.add_argument("--receipts", nargs=3, required=True)
    p.add_argument("--output", required=True)
    p = commands.add_parser("artifact-ref")
    p.add_argument("--file", required=True)
    p.add_argument("--repo-id", required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--path", required=True)
    p.add_argument("--output", required=True)
    p = commands.add_parser("validate")
    p.add_argument("--submission", required=True)
    p.add_argument("--keys", required=True)
    p.add_argument("--resolve-artifacts", action="store_true")
    p.add_argument("--allow-test", action="store_true")
    p.add_argument("--report", required=True)
    p = commands.add_parser("publish")
    p.add_argument("--submissions-dir", required=True)
    p.add_argument("--keys", required=True)
    p.add_argument("--approval-ledger", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--repo-id", default="Ray0311/AutoRSE-Leaderboard")
    p.add_argument("--dry-run", action="store_true")
    p = commands.add_parser("check-publisher")
    p.add_argument("--repo-id", required=True)
    p.add_argument("--sha", required=True)
    args = parser.parse_args(argv)
    if args.command == "template":
        write_json(args.output, template())
    elif args.command == "prepare":
        metadata = read_json(args.metadata)
        metadata["receipts"] = [read_json(path) for path in args.receipts]
        for run in metadata.get("runs", []):
            receipt = metadata["receipts"][run["trial_index"] - 1]
            expected = {"receipt_evaluation_id": receipt["evaluation_id"],
                        "prediction_sha256": receipt["tasks"][run["task"]]["submission_sha256"]}
            for field, value in expected.items():
                if run.get(field) and run[field] != value:
                    raise ValueError("metadata_receipt_binding_conflict")
                run[field] = value
        write_json(args.output, metadata)
    elif args.command == "artifact-ref":
        from .artifacts import MAX_FILE_BYTES
        from .validation import _artifact_ref_problem
        path = Path(args.file)
        if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= MAX_FILE_BYTES:
            raise ValueError("invalid_artifact_file")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        ref = {"repo_id": args.repo_id, "revision": args.revision, "path": args.path,
               "sha256": digest, "size": path.stat().st_size}
        if _artifact_ref_problem(ref):
            raise ValueError("invalid_pinned_artifact_reference")
        write_json(args.output, ref)
    elif args.command == "validate":
        from .validation import validate_submission
        from .artifacts import PublicArtifactReader
        report = validate_submission(read_json(args.submission), read_json(args.keys),
            artifact_reader=PublicArtifactReader() if args.resolve_artifacts else None, allow_test=args.allow_test)
        write_json(args.report, report)
        print(json.dumps({key: report.get(key) for key in ("valid", "publishable")}))
        return 0 if report.get("valid") else 1
    elif args.command == "check-publisher":
        check_publisher(args.repo_id, args.sha)
    else:
        publish(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        # No API response bodies, artifact content, local secrets or credential values in CI logs.
        print(json.dumps({"error_type": type(error).__name__, "message": "Operation failed; inspect validation checks or maintainer setup."}))
        raise SystemExit(1)
