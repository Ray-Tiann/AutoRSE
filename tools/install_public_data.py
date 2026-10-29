#!/usr/bin/env python3
"""Download or install the pinned public AutoRSE benchmark data without execution.

Only Python's standard library is required. ZIP members are copied as inert
bytes; pickle/NumPy/Parquet content is never deserialized by this installer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import tempfile
from urllib.parse import quote
from urllib.request import Request, urlopen
import zipfile


# Updated only with a reviewed public-data release, never from remote metadata.
BUNDLE_NAME = "recagent-data-public-v2.zip"
BUNDLE_SHA256 = "4185fa03339cd37df9591906ef1e2e309beec6b3676e724d2c60916b13679414"
BUNDLE_BYTES = 941_569_295
DEFAULT_REPO_ID = "RayTian/AutoRSE"
MANIFEST_NAME = "recagent-data-manifest.json"
TASKS = ("hstu_task", "tiger_task", "intsr_task", "wukong_task", "rankmixer_task", "onetrans_task")
MAX_MANIFEST_BYTES = 1_048_576
MAX_FILE_BYTES = 2_147_483_648
MAX_TOTAL_BYTES = 4_294_967_296
CHUNK_BYTES = 8 * 1024 * 1024
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
REPO_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}\Z")


class DataError(Exception):
    """An archive, destination or download failed validation."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_archive_pin(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise DataError("archive must be a regular file, not a symlink")
    if path.stat().st_size != BUNDLE_BYTES:
        raise DataError(f"archive size differs from pinned release ({BUNDLE_BYTES} bytes)")
    if sha256_file(path) != BUNDLE_SHA256:
        raise DataError("archive SHA-256 differs from pinned public release")


def download_bundle(repo_id: str, revision: str, destination: Path) -> None:
    if not REPO_RE.fullmatch(repo_id) or any(part in (".", "..") for part in repo_id.split("/")):
        raise DataError("--repo-id must be a Hugging Face namespace/dataset name")
    if not revision or any(ord(char) < 32 for char in revision):
        raise DataError("--revision must be nonempty and contain no control characters")
    url = f"https://huggingface.co/datasets/{repo_id}/resolve/{quote(revision, safe='')}/{BUNDLE_NAME}"
    request = Request(url, headers={"User-Agent": "AutoRSE-public-data-installer/2"})
    digest = hashlib.sha256()
    total = 0
    print(f"Downloading {BUNDLE_NAME} from {repo_id} at {revision}", flush=True)
    with urlopen(request, timeout=60) as response, destination.open("xb") as output:
        if not response.geturl().startswith("https://"):
            raise DataError("refusing a download redirect without HTTPS")
        length = response.headers.get("Content-Length")
        if length is not None and int(length) != BUNDLE_BYTES:
            raise DataError("download Content-Length differs from pinned release")
        while True:
            chunk = response.read(min(CHUNK_BYTES, BUNDLE_BYTES - total + 1))
            if not chunk:
                break
            total += len(chunk)
            if total > BUNDLE_BYTES:
                raise DataError("download exceeds pinned release size")
            digest.update(chunk)
            output.write(chunk)
    if total != BUNDLE_BYTES or digest.hexdigest() != BUNDLE_SHA256:
        raise DataError("download does not match the pinned size and SHA-256")


def unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise DataError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def public_path(name: str) -> str:
    if not isinstance(name, str) or "\\" in name or ":" in name or "\x00" in name:
        raise DataError("invalid archive member path")
    parts = name.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise DataError("noncanonical archive member path")
    if (len(parts) < 6 or parts[:2] != ["benchmark", "tasks"] or parts[2] not in TASKS
            or parts[3:5] != ["environment", "datasets"]):
        raise DataError(f"member outside the public dataset namespace: {name}")
    lowered = {part.lower() for part in parts}
    if lowered & {"private", "tests", "solution", "working", "experiments", "logs", "jobs", "harbor_runs", ".recagent"}:
        raise DataError(f"forbidden dataset path: {name}")
    if any(part.lower() in {".env", "credentials.json", "secrets.json", "config.yaml"}
           or part.lower().endswith(".env") for part in parts):
        raise DataError(f"credential-like dataset path: {name}")
    return parts[2]


def read_manifest(archive: zipfile.ZipFile) -> dict:
    infos = archive.infolist()
    if len(infos) > 1024:
        raise DataError("too many ZIP entries")
    names = [info.filename for info in infos]
    if len(names) != len(set(names)):
        raise DataError("duplicate ZIP member names")
    if names.count(MANIFEST_NAME) != 1:
        raise DataError("ZIP must contain exactly one public-data manifest")
    for info in infos:
        mode = info.external_attr >> 16
        if info.is_dir() or stat.S_IFMT(mode) not in (0, stat.S_IFREG):
            raise DataError(f"ZIP contains a directory, symlink or special file: {info.filename}")
        if info.flag_bits & 1 or info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
            raise DataError("encrypted or unsupported ZIP member")
        if info.file_size < 0 or info.file_size > MAX_FILE_BYTES:
            raise DataError("ZIP member exceeds size limit")
    if archive.getinfo(MANIFEST_NAME).file_size > MAX_MANIFEST_BYTES:
        raise DataError("public-data manifest exceeds size limit")
    manifest = json.loads(archive.read(MANIFEST_NAME), object_pairs_hook=unique_json_object)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise DataError("unsupported manifest schema")
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise DataError("manifest has no file list")
    expected = {}
    total = 0
    present_tasks = set()
    for item in entries:
        if not isinstance(item, dict):
            raise DataError("invalid file entry")
        name, size, checksum = item.get("path"), item.get("size"), item.get("sha256")
        present_tasks.add(public_path(name))
        if name in expected:
            raise DataError("duplicate manifest path")
        if type(size) is not int or not 0 <= size <= MAX_FILE_BYTES:
            raise DataError("invalid manifest file size")
        if not isinstance(checksum, str) or not SHA256_RE.fullmatch(checksum):
            raise DataError("invalid manifest SHA-256")
        expected[name] = item
        total += size
    if total > MAX_TOTAL_BYTES or manifest.get("total_bytes") != total:
        raise DataError("invalid manifest total size")
    if manifest.get("file_count") != len(entries):
        raise DataError("invalid manifest file count")
    if set(manifest.get("tasks", [])) != present_tasks or present_tasks != set(TASKS):
        raise DataError("manifest task inventory differs from six-task release")
    if set(names) != set(expected) | {MANIFEST_NAME}:
        raise DataError("ZIP and manifest inventories differ")
    for name, item in expected.items():
        if archive.getinfo(name).file_size != item["size"]:
            raise DataError(f"ZIP size differs from manifest: {name}")
    return manifest


def checked_target(root: Path, name: str, create: bool = False) -> Path:
    public_path(name)
    current = root
    for part in PurePosixPath(name).parts[:-1]:
        current = current / part
        if current.is_symlink():
            raise DataError(f"refusing destination symlink: {current}")
        if current.exists() and not current.is_dir():
            raise DataError(f"destination parent is not a directory: {current}")
        if create:
            current.mkdir(exist_ok=True)
    target = current / PurePosixPath(name).name
    if target.is_symlink():
        raise DataError(f"refusing destination symlink: {target}")
    if target.exists() and not target.is_file():
        raise DataError(f"destination is not a regular file: {target}")
    if not target.resolve().is_relative_to(root):
        raise DataError("destination escaped repository root")
    return target


def matches_file(target: Path, item: dict) -> bool:
    return target.is_file() and target.stat().st_size == item["size"] and sha256_file(target) == item["sha256"]


def install(archive_path: Path, root: Path, tasks: set[str]) -> tuple[int, int]:
    check_archive_pin(archive_path)
    root = root.expanduser().absolute()
    if root.is_symlink():
        raise DataError("repository root must not be a symlink")
    root = root.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        manifest = read_manifest(archive)
        entries = [item for item in manifest["files"] if public_path(item["path"]) in tasks]
        if root.exists() and not root.is_dir():
            raise DataError("repository root is not a directory")
        # Resolve every conflict before writing any dataset file.
        unchanged = set()
        for item in entries:
            target = checked_target(root, item["path"])
            if target.exists():
                if not matches_file(target, item):
                    raise DataError(f"existing file differs; remove or move it explicitly: {target}")
                unchanged.add(item["path"])
        root.mkdir(parents=True, exist_ok=True)
        installed = 0
        for item in entries:
            if item["path"] in unchanged:
                continue
            target = checked_target(root, item["path"], create=True)
            descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".partial", dir=target.parent)
            temporary = Path(temporary_name)
            try:
                digest = hashlib.sha256()
                total = 0
                with os.fdopen(descriptor, "wb") as output, archive.open(item["path"]) as source:
                    for chunk in iter(lambda: source.read(CHUNK_BYTES), b""):
                        total += len(chunk)
                        if total > item["size"]:
                            raise DataError("extracted member exceeds manifest size")
                        digest.update(chunk)
                        output.write(chunk)
                    output.flush()
                    os.fsync(output.fileno())
                if total != item["size"] or digest.hexdigest() != item["sha256"]:
                    raise DataError(f"extracted member SHA-256/size mismatch: {item['path']}")
                checked_target(root, item["path"])
                try:
                    # link() publishes the verified file atomically without replacing
                    # any destination created since preflight, on Unix and Windows.
                    os.link(temporary, target)
                except FileExistsError:
                    checked_target(root, item["path"])
                    if not matches_file(target, item):
                        raise DataError(f"destination appeared during install: {target}")
                    unchanged.add(item["path"])
                else:
                    installed += 1
            finally:
                temporary.unlink(missing_ok=True)
        print(f"Installed {installed}; unchanged {len(unchanged)} files under {root}")
        return installed, len(unchanged)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--archive", type=Path, help="local pinned public-data ZIP")
    source.add_argument("--repo-id", help=f"public Hugging Face dataset (default when no archive is given: {DEFAULT_REPO_ID})")
    parser.add_argument("--revision", default="main", help="Hugging Face revision (commit recommended; default: main)")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent,
                        help="benchmark repository root (default: script's grandparent directory)")
    parser.add_argument("--task", action="append", choices=[*TASKS, *(name.removesuffix('_task') for name in TASKS)],
                        help="install only this task; repeat for multiple tasks; default: all six")
    args = parser.parse_args(argv)
    if args.archive is None and args.repo_id is None:
        args.repo_id = DEFAULT_REPO_ID
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    tasks = {name if name.endswith("_task") else name + "_task" for name in args.task} if args.task else set(TASKS)
    try:
        if args.archive:
            install(args.archive.expanduser().absolute(), args.root, tasks)
        else:
            with tempfile.TemporaryDirectory(prefix="autorse-public-data-") as temporary_dir:
                archive_path = Path(temporary_dir) / BUNDLE_NAME
                download_bundle(args.repo_id, args.revision, archive_path)
                install(archive_path, args.root, tasks)
        return 0
    except (DataError, OSError, ValueError, TypeError, KeyError, zipfile.BadZipFile, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
