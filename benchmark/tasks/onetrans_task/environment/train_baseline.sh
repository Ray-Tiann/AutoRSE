#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATASET="${1:-kuaisar_v2}"
[[ "$DATASET" == "kuaisar_v2" ]] || { echo "Only kuaisar_v2 is available" >&2; exit 2; }
if [[ $# -gt 0 ]]; then
  shift
fi

GPU_COUNT="${GPU_COUNT:-1}"
[[ "$GPU_COUNT" == "1" ]] || { echo "Single-GPU study" >&2; exit 2; }

SEED="${RESEARCH_SEED:-2024}"
SEQ_LEN="${SEQ_LEN:-128}"
PER_GPU_BATCH="${PER_GPU_BATCH:-512}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
MAX_STEPS="${MAX_STEPS:-1000}"
VALIDATE_EVERY="${VALIDATE_EVERY:-250}"
SAVE_EVERY="${SAVE_EVERY:-500}"
NUM_WORKERS="${NUM_WORKERS:-2}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-onetrans_fast_seed${SEED}}"
RESULT_DIR="$ROOT_DIR/datasets/$DATASET/experiments/$EXPERIMENT_NAME"
WORKING_DIR="$ROOT_DIR/datasets/$DATASET/working"
PROCESSED_DIR="$ROOT_DIR/datasets/$DATASET/input/processed"
MASTER_PORT="${MASTER_PORT:-29500}"

mkdir -p "$RESULT_DIR" "$WORKING_DIR"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/baseline${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

checkpoint_args=()
if [[ "${ACTIVATION_CHECKPOINT:-0}" == "1" ]]; then
  checkpoint_args+=(--activation-checkpoint)
fi

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc-per-node="$GPU_COUNT" \
  --master-port="$MASTER_PORT" \
  baseline/scripts/train_kuaisar.py \
  --processed-dir "$PROCESSED_DIR" \
  --output-dir "$RESULT_DIR" \
  --model onetrans \
  --optimizer hybrid \
  --device cuda:0 \
  --amp-dtype bf16 \
  --seq-len "$SEQ_LEN" \
  --ns-len 12 \
  --d-model 256 \
  --num-heads 4 \
  --ffn-hidden 1856 \
  --num-layers 6 \
  --pyramid-align 32 \
  --mask-type paper_causal \
  --batch-size "$PER_GPU_BATCH" \
  --grad-accum "$GRAD_ACCUM" \
  --lr 0.001 \
  --sparse-lr 0.1 \
  --rmsprop-alpha 0.99999 \
  --weight-decay 0 \
  --click-only \
  --skip-test-eval \
  --grad-clip-dense 90 \
  --grad-clip-sparse 120 \
  --num-workers "$NUM_WORKERS" \
  --seed "$SEED" \
  --epochs 10 \
  --max-global-steps "$MAX_STEPS" \
  --validate-every "$VALIDATE_EVERY" \
  --save-every "$SAVE_EVERY" \
  --log-every 20 \
  --early-stop-after "$MAX_STEPS" \
  --early-stop-patience 0 \
  --distributed \
  "${checkpoint_args[@]}" \
  "$@" \
  2>&1 | tee "$RESULT_DIR/launcher.log"

[[ -f "$RESULT_DIR/best.pt" ]] || { echo "best.pt was not produced" >&2; exit 3; }
python3 -u baseline/scripts/predict_kuaisar.py \
  --processed-dir "$PROCESSED_DIR" \
  --checkpoint "$RESULT_DIR/best.pt" \
  --output "$RESULT_DIR/submission.csv" \
  --split test \
  --batch-size "${PREDICT_BATCH:-1024}" \
  --num-workers "$NUM_WORKERS" \
  --amp-dtype bf16 \
  2>&1 | tee "$RESULT_DIR/predict.log"

cp "$RESULT_DIR/submission.csv" "$WORKING_DIR/submission.csv"
echo "SUBMISSION=$WORKING_DIR/submission.csv"
