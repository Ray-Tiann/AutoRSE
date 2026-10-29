#!/usr/bin/env bash
set -Eeuo pipefail

DATASET="${1:-movielenslatest}"
shift || true
if [[ "$DATASET" != "movielenslatest" ]]; then
    echo "This Harbor task contains only the movielenslatest dataset." >&2
    exit 2
fi

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
GPU_ID="${GPU_ID:-0}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-wukong_baseline}"
RESULT_DIR="$ROOT_DIR/datasets/$DATASET/experiments/$EXPERIMENT_NAME"
WORKING_DIR="$ROOT_DIR/datasets/$DATASET/working"

mkdir -p "$RESULT_DIR" "$WORKING_DIR"
export PYTHONPATH="$ROOT_DIR/baseline/fuxictr:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU_ID}"

"$PYTHON_BIN" -u "$ROOT_DIR/baseline/train.py" \
    --dataset "$DATASET" \
    --run-name "$EXPERIMENT_NAME" \
    --gpu 0 \
    --seed "${RESEARCH_SEED:-2024}" \
    --epochs 100 \
    --batch-size 10000 \
    --learning-rate 0.001 \
    --embedding-dim 128 \
    --num-layers 4 \
    --compressed-features 16 \
    --rank-k 24 \
    --dropout 0.0 \
    "$@" \
    2>&1 | tee "$RESULT_DIR/train.log"

cp "$RESULT_DIR/prediction.csv.gz" "$WORKING_DIR/submission.csv.gz"
echo "SUBMISSION=$WORKING_DIR/submission.csv.gz"
