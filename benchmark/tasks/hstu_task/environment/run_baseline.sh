#!/usr/bin/env bash
set -Eeuo pipefail
WORKSPACE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$WORKSPACE/baseline_git.py" run --workspace "$WORKSPACE" --task hstu -- bash "$WORKSPACE/train_baseline.sh" "$@"
