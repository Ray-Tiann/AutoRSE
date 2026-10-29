#!/usr/bin/env bash
set -Eeuo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
docker build -t autorse-agent:latest \
  --build-arg "CODEX_VERSION=${CODEX_VERSION:-latest}" \
  --build-arg "CLAUDE_CODE_VERSION=${CLAUDE_CODE_VERSION:-latest}" \
  "$ROOT_DIR/benchmark/base" "$@"
