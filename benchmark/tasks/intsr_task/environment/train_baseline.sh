#!/usr/bin/env bash
set -Eeuo pipefail

DATASET_NAME="${1:-Amazon}"
shift || true
if [[ "$DATASET_NAME" != "Amazon" ]]; then
    echo "Unknown dataset: $DATASET_NAME" >&2
    exit 2
fi

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASELINE_DIR="$ROOT_DIR/baseline"
DATASET_DIR="$ROOT_DIR/datasets/$DATASET_NAME"
TRAIN_DATA_DIR="$DATASET_DIR/input/train_data"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-intsr_baseline}"
RESULT_DIR="$DATASET_DIR/experiments/$EXPERIMENT_NAME"
PYTHON_BIN="${PYTHON_BIN:-python3}"
GPU_ID="${GPU_ID:-0}"
NUM_WORKERS="${NUM_WORKERS:-2}"
PREDICT_MAX_EVAL_BATCHES="${PREDICT_MAX_EVAL_BATCHES:-0}"

mkdir -p "$RESULT_DIR/checkpoints" "$DATASET_DIR/working"
for required in \
    "$DATASET_DIR/input/test_data_unlabeled.pkl" \
    "$TRAIN_DATA_DIR/features/qwen3_0.6b_last_fp16.npz" \
    "$TRAIN_DATA_DIR/vocab/src_session_pos_items.npy"; do
    if [[ ! -f "$required" ]]; then
        echo "Missing required public artifact: $required" >&2
        exit 2
    fi
done

QUERY_ARGS=(
    --query_backend precomputed_qwen
    --query_feature_path "$TRAIN_DATA_DIR/features/qwen3_0.6b_last_fp16.npz"
    --session_pos_items_path "$TRAIN_DATA_DIR/vocab/src_session_pos_items.npy"
    --query_augmentation_beta 0.0
    --separate_task_embedding 1
    --prediction_mode ranking_mlp
    --historical_query_include_item 0
    --candidate_scorer_layers 2
)

cd "$BASELINE_DIR"
"$PYTHON_BIN" main.py \
    --model IntSR \
    --data "$DATASET_NAME" \
    --data_root "$TRAIN_DATA_DIR" \
    --model_path "$RESULT_DIR/checkpoints" \
    --log_path "$RESULT_DIR/train.log" \
    --metadata_path "$RESULT_DIR/metrics.json" \
    --time "$EXPERIMENT_NAME" \
    --gpu "$GPU_ID" \
    --num_workers "$NUM_WORKERS" \
    "${QUERY_ARGS[@]}" \
    --random_seed "${RESEARCH_SEED:-2024}" \
    --epoch 6 \
    --batch_size 32 \
    --eval_batch_size 32 \
    --early_stop 0 \
    "$@"

cp "$RESULT_DIR/checkpoints/best.pt" "$RESULT_DIR/checkpoint.pt"
CHECKPOINT_ARGS=(--checkpoint "$RESULT_DIR/checkpoint.pt")
if [[ -f "$RESULT_DIR/checkpoints/rec_pretrain_best.pt" ]]; then
    cp "$RESULT_DIR/checkpoints/rec_pretrain_best.pt" "$RESULT_DIR/rec_checkpoint.pt"
    CHECKPOINT_ARGS=(
        --rec-checkpoint "$RESULT_DIR/rec_checkpoint.pt"
        --src-checkpoint "$RESULT_DIR/checkpoint.pt"
    )
fi

"$PYTHON_BIN" predict.py \
    --data "$DATASET_NAME" \
    --train-data "$TRAIN_DATA_DIR" \
    --unlabeled "$DATASET_DIR/input/test_data_unlabeled.pkl" \
    "${CHECKPOINT_ARGS[@]}" \
    --output "$RESULT_DIR/prediction.npz" \
    --gpu "$GPU_ID" \
    --num_workers "$NUM_WORKERS" \
    --eval_batch_size 32 \
    --max_eval_batches "$PREDICT_MAX_EVAL_BATCHES" \
    "${QUERY_ARGS[@]}"

cp "$RESULT_DIR/prediction.npz" "$DATASET_DIR/working/submission.npz"
echo "SUBMISSION=$DATASET_DIR/working/submission.npz"
