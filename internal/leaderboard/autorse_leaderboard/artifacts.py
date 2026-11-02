"""Read pinned, public HF evidence without running submitted code or extracting it."""
import hashlib
import re
from pathlib import PurePosixPath

MAX_FILE_BYTES = 160 * 1024 * 1024
MAX_TOTAL_BYTES = 2 * 1024 * 1024 * 1024


class PublicArtifactReader:
    def __init__(self):
        self.total = 0
        self.repositories = set()
        self.cache = {}

    def __call__(self, ref):
        from huggingface_hub import HfApi, hf_hub_url, get_hf_file_metadata
        import httpx
        repo, revision, path = ref["repo_id"], ref["revision"], ref["path"]
        parts = PurePosixPath(path).parts if isinstance(path, str) else ()
        if (not isinstance(repo, str) or not re.fullmatch(r"[A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*", repo)
                or not re.fullmatch(r"[0-9a-f]{40}", revision) or not parts
                or PurePosixPath(path).is_absolute() or ".." in parts or "\\" in path
                or any(ord(c) < 32 for c in path)):
            raise ValueError("unsafe_public_artifact_reference")
        size = ref["size"]
        if type(size) is not int or not 0 < size <= MAX_FILE_BYTES:
            raise ValueError("artifact_size_limit")
        cache_key = (repo, revision, path, ref["sha256"], size)
        if cache_key in self.cache:
            return self.cache[cache_key]
        if self.total + size > MAX_TOTAL_BYTES:
            raise ValueError("submission_artifact_size_limit")
        if (repo, revision) not in self.repositories:
            info = HfApi(token=False).dataset_info(repo, revision=revision)
            if info.private is not False or info.sha != revision or info.gated:
                raise ValueError("evidence_must_be_pinned_and_public")
            self.repositories.add((repo, revision))
        url = hf_hub_url(repo, filename=path, repo_type="dataset", revision=revision)
        metadata = get_hf_file_metadata(url, token=False)
        if metadata.size != size or metadata.commit_hash != revision:
            raise ValueError("artifact_metadata_mismatch")
        chunks, count, digest = [], 0, hashlib.sha256()
        with httpx.stream("GET", url, follow_redirects=True, timeout=60) as response:
            response.raise_for_status()
            for chunk in response.iter_bytes(1024 * 1024):
                count += len(chunk)
                if count > size:
                    raise ValueError("artifact_size_mismatch")
                digest.update(chunk)
                chunks.append(chunk)
        if count != size or digest.hexdigest() != ref["sha256"]:
            raise ValueError("artifact_hash_mismatch")
        data = b"".join(chunks)
        self.total += count
        self.cache[cache_key] = data
        return data
