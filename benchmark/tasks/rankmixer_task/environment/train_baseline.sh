#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATASET="${1:-criteo5m}"
[[ "$DATASET" == "criteo5m" ]] || { echo "Only criteo5m is available" >&2; exit 2; }
if [[ $# -gt 0 ]]; then
  shift
fi
EXPERIMENT_NAME="${EXPERIMENT_NAME:-rankmixer_baseline_seed${RESEARCH_SEED:-2024}}"
RESULT_DIR="$ROOT_DIR/datasets/$DATASET/experiments/$EXPERIMENT_NAME"
WORKING_DIR="$ROOT_DIR/datasets/$DATASET/working"
GPU_COUNT="${GPU_COUNT:-1}"
MASTER_PORT="${MASTER_PORT:-29500}"

[[ "$GPU_COUNT" == "1" ]] || { echo "Authorized single-GPU study" >&2; exit 2; }
mkdir -p "$RESULT_DIR" "$WORKING_DIR"
cd "$ROOT_DIR"

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc-per-node="$GPU_COUNT" \
  baseline/train.py \
  --config baseline/config.yaml \
  --results_dir "$RESULT_DIR" \
  --seed "${RESEARCH_SEED:-2024}" \
  "$@" \
  2>&1 | tee "$RESULT_DIR/launcher.log"

python3 baseline/predict.py \
  --checkpoint "$RESULT_DIR/checkpoint.pt" \
  --data-dir "$ROOT_DIR/datasets/$DATASET/input" \
  --output "$RESULT_DIR/submission.npz"

cp "$RESULT_DIR/submission.npz" "$WORKING_DIR/submission.npz"
echo "SUBMISSION=$WORKING_DIR/submission.npz"
