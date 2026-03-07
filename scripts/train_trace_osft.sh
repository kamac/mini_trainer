#!/bin/bash
# train_trace_osft.sh — Phase 4 & 5: sequential OSFT training + per-task evaluation
#
# For each of the 8 TRACE tasks in order:
#   1. Resume from the previous task's checkpoint (or the base model for task 1)
#   2. Skip training if this task's checkpoint already exists (resume support)
#   3. Run TRACE task evaluation on all tasks seen so far
#   4. Run MMLU evaluation
#
# Artifacts per task N:
#   $CKPT_ROOT/task_N_<TASK>/                         — fine-tuned checkpoint
#   $RESULTS_DIR/<TASK>_after_task<N>.json            — TRACE task accuracy
#   $RESULTS_DIR/mmlu/osft_after_task_<N>/results.json — MMLU score
#
# Usage:
#   bash scripts/train_trace_osft.sh [options]
#
# Options:
#   --model MODEL                  Base model (default: meta-llama/Llama-2-7b-chat-hf)
#   --data-root DIR                Tokenized TRACE data root (default: /data/TRACE_tokenized)
#   --trace-raw-dir DIR            Raw TRACE data for evaluation (default: /data/TRACE)
#   --ckpt-root DIR                Checkpoint output root (default: /checkpoints/trace_osft)
#   --results-dir DIR              Results output root (default: results)
#   --n-gpus N                     Number of GPUs (default: 1)
#   --max-tokens-per-gpu N         Tokens per GPU per step (default: 2048)
#   --batch-size N                 Sequences per global batch (default: 128)
#   --unfreeze-rank-ratio FLOAT    OSFT rank ratio (default: 0.25)
#   --start-task N                 Start from task N (1-indexed) instead of auto-detecting

set -euo pipefail

# ── defaults ──────────────────────────────────────────────────────────────────
MODEL="meta-llama/Llama-2-7b-chat-hf"
DATA_ROOT="/data/TRACE_tokenized"
TRACE_RAW_DIR="/data/TRACE"
CKPT_ROOT="/checkpoints/trace_osft"
RESULTS_DIR="results"
N_GPUS=1
MAX_TOKENS_PER_GPU=2048
BATCH_SIZE=128
UNFREEZE_RANK_RATIO="0.25"
START_TASK=""   # empty = auto-detect from existing checkpoints

# ── argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)                MODEL="$2";               shift 2 ;;
        --data-root)            DATA_ROOT="$2";           shift 2 ;;
        --trace-raw-dir)        TRACE_RAW_DIR="$2";       shift 2 ;;
        --ckpt-root)            CKPT_ROOT="$2";           shift 2 ;;
        --results-dir)          RESULTS_DIR="$2";         shift 2 ;;
        --n-gpus)               N_GPUS="$2";              shift 2 ;;
        --max-tokens-per-gpu)   MAX_TOKENS_PER_GPU="$2";  shift 2 ;;
        --batch-size)           BATCH_SIZE="$2";          shift 2 ;;
        --unfreeze-rank-ratio)  UNFREEZE_RANK_RATIO="$2"; shift 2 ;;
        --start-task)           START_TASK="$2";          shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

TASKS=(
    "C-STANCE"
    "FOMC"
    "MeetingBank"
    "Py150"
    "ScienceQA"
    "NumGLUE-cm"
    "NumGLUE-ds"
    "20Minuten"
)

mkdir -p "$CKPT_ROOT" "$RESULTS_DIR/mmlu"

# ── resume detection ──────────────────────────────────────────────────────────
# Find the last completed task checkpoint so we can skip already-done tasks.
# A checkpoint is considered complete when its directory contains config.json
# (written by save_pretrained at the end of training).

find_last_completed_task() {
    # Return the last task number in an unbroken sequence of completed checkpoints.
    # Stops at the first gap so that a partially-saved or deleted checkpoint causes
    # re-training from that point, not a silent skip that corrupts the model chain.
    local last=0
    for i in "${!TASKS[@]}"; do
        local task_num=$((i + 1))
        local task="${TASKS[$i]}"
        local ckpt="$CKPT_ROOT/task_${task_num}_${task}"
        if [[ -f "$ckpt/config.json" ]]; then
            last=$task_num
        else
            break
        fi
    done
    echo "$last"
}

if [[ -n "$START_TASK" ]]; then
    LAST_COMPLETED=$((START_TASK - 1))
    echo "Starting from task $START_TASK (forced via --start-task)"
else
    LAST_COMPLETED=$(find_last_completed_task)
    if [[ $LAST_COMPLETED -gt 0 ]]; then
        echo "Resuming: tasks 1–$LAST_COMPLETED already have checkpoints, starting from task $((LAST_COMPLETED + 1))"
    else
        echo "No existing checkpoints found, starting from task 1"
    fi
fi

# ── resolve the starting model (base or last checkpoint) ─────────────────────
if [[ $LAST_COMPLETED -eq 0 ]]; then
    CURRENT_MODEL="$MODEL"
else
    PREV_TASK="${TASKS[$((LAST_COMPLETED - 1))]}"
    CURRENT_MODEL="$CKPT_ROOT/task_${LAST_COMPLETED}_${PREV_TASK}"
    echo "Starting model: $CURRENT_MODEL"
fi

echo "================================================================"
echo " TRACE OSFT training"
echo "  base model         : $MODEL"
echo "  starting model     : $CURRENT_MODEL"
echo "  data root          : $DATA_ROOT"
echo "  checkpoint root    : $CKPT_ROOT"
echo "  results dir        : $RESULTS_DIR"
echo "  GPUs               : $N_GPUS"
echo "  max tokens/GPU     : $MAX_TOKENS_PER_GPU"
echo "  batch size         : $BATCH_SIZE"
echo "  unfreeze rank ratio: $UNFREEZE_RANK_RATIO"
echo "================================================================"

# ── main loop ─────────────────────────────────────────────────────────────────
for i in "${!TASKS[@]}"; do
    TASK="${TASKS[$i]}"
    TASK_NUM=$((i + 1))
    OUTPUT_DIR="$CKPT_ROOT/task_${TASK_NUM}_${TASK}"

    # Skip tasks already completed before this run started
    if [[ $TASK_NUM -le $LAST_COMPLETED ]]; then
        echo "── Skipping task $TASK_NUM ($TASK): checkpoint exists ──"
        CURRENT_MODEL="$OUTPUT_DIR"
        continue
    fi

    # ── 4. Train on this task ─────────────────────────────────────────────────
    echo ""
    echo "════ Task $TASK_NUM / ${#TASKS[@]}: $TASK ════"
    echo "    input model : $CURRENT_MODEL"
    echo "    output      : $OUTPUT_DIR"

    torchrun \
        --nnodes=1 \
        --nproc-per-node="$N_GPUS" \
        -m mini_trainer.train \
        --model-name-or-path "$CURRENT_MODEL" \
        --data-path "$DATA_ROOT/$TASK/train.jsonl" \
        --output-dir "$OUTPUT_DIR" \
        --batch-size "$BATCH_SIZE" \
        --max-tokens-per-gpu "$MAX_TOKENS_PER_GPU" \
        --learning-rate 2e-4 \
        --num-warmup-steps 100 \
        --lr-scheduler cosine \
        --beta1 0.9 \
        --beta2 0.95 \
        --train-dtype bfloat16 \
        --max-epochs 3 \
        --training-mode epoch \
        --use-liger-kernels \
        --osft \
        --osft-unfreeze-rank-ratio "$UNFREEZE_RANK_RATIO" \
        --save-final-checkpoint \
        --checkpoint-at-epoch

    echo "── Training complete: $OUTPUT_DIR ──"

    # ── 5. TRACE task evaluation on all tasks seen so far ────────────────────
    echo ""
    echo "── Evaluating TRACE tasks 1–$TASK_NUM after task $TASK_NUM ──"
    for j in $(seq 0 $((TASK_NUM - 1))); do
        EVAL_TASK="${TASKS[$j]}"
        EVAL_OUT="$RESULTS_DIR/${EVAL_TASK}_after_task${TASK_NUM}.json"
        if [[ -f "$EVAL_OUT" ]]; then
            echo "   Skipping $EVAL_TASK (result already exists)"
            continue
        fi
        echo "   Evaluating $EVAL_TASK ..."
        python /opt/TRACE/metrics.py \
            --model "$OUTPUT_DIR" \
            --test_file "$TRACE_RAW_DIR/$EVAL_TASK/test.json" \
            --task "$EVAL_TASK" \
            --output_file "$EVAL_OUT"
    done

    # ── MMLU evaluation ───────────────────────────────────────────────────────
    MMLU_OUT="$RESULTS_DIR/mmlu/osft_after_task_${TASK_NUM}"
    if [[ -f "$MMLU_OUT/results.json" ]]; then
        echo "── Skipping MMLU for task $TASK_NUM (already exists) ──"
    else
        echo ""
        echo "── MMLU evaluation after task $TASK_NUM ──"
        mkdir -p "$MMLU_OUT"
        lm_eval \
            --model hf \
            --model_args "pretrained=$OUTPUT_DIR,dtype=bfloat16" \
            --tasks mmlu \
            --num_fewshot 5 \
            --batch_size auto \
            --output_path "$MMLU_OUT/results.json"
        echo "── MMLU complete: $MMLU_OUT/results.json ──"
    fi

    CURRENT_MODEL="$OUTPUT_DIR"
done

echo ""
echo "All tasks complete. Next: bash scripts/report.sh"
