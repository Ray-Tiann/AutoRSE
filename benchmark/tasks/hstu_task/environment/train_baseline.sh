#!/usr/bin/env bash
set -Eeuo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
[[ "${1:-beauty}" == beauty ]] || exit 2
if [[ $# -gt 0 ]]; then shift; fi
python3 baseline/train.py --seed "${RESEARCH_SEED:-2024}" "$@"
python3 baseline/submit.py
