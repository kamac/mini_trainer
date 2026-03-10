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
#   --max-epochs N                 Epochs per task (default: 3)
#   --no-liger-kernels             Disable Liger kernels (useful when cuda extras are not installed)
#   --skip-trace-eval              Skip TRACE task evaluation (useful for testing)
#   --skip-mmlu-eval               Skip MMLU evaluation (useful for testing)

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
UNFREEZE_RANK_RATIO="0.10"
START_TASK=""   # empty = auto-detect from existing checkpoints
MAX_EPOCHS=3
USE_LIGER_KERNELS=true
SKIP_TRACE_EVAL=false
SKIP_MMLU_EVAL=false
SPECTRAL_DIR=""

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
        --max-epochs)           MAX_EPOCHS="$2";          shift 2 ;;
        --no-liger-kernels)     USE_LIGER_KERNELS=false;  shift ;;
        --skip-trace-eval)      SKIP_TRACE_EVAL=true;     shift ;;
        --skip-mmlu-eval)       SKIP_MMLU_EVAL=true;      shift ;;
        --spectral-dir)         SPECTRAL_DIR="$2";        shift 2 ;;
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

# ── checkpoint path helpers ───────────────────────────────────────────────────
# mini_trainer saves HF checkpoints under:
#   $output_dir/hf_format/samples_<N>/
# This function finds the latest (highest-samples) checkpoint inside an output dir.

find_hf_checkpoint() {
    local output_dir="$1"
    local hf_dir="$output_dir/hf_format"
    if [[ ! -d "$hf_dir" ]]; then
        echo ""
        return
    fi
    # Sort numerically by the samples count after the underscore
    local latest
    latest=$(ls -d "$hf_dir"/samples_* 2>/dev/null \
        | sort -t_ -k2 -g \
        | tail -1)
    echo "${latest:-}"
}

# ── resume detection ──────────────────────────────────────────────────────────
# Find the last completed task checkpoint so we can skip already-done tasks.
# A checkpoint is considered complete when its hf_format/samples_* dir contains config.json.

find_last_completed_task() {
    # Return the last task number in an unbroken sequence of completed checkpoints.
    # Stops at the first gap so that a partially-saved or deleted checkpoint causes
    # re-training from that point, not a silent skip that corrupts the model chain.
    local last=0
    for i in "${!TASKS[@]}"; do
        local task_num=$((i + 1))
        local task="${TASKS[$i]}"
        local ckpt="$CKPT_ROOT/task_${task_num}_${task}"
        local hf_ckpt
        hf_ckpt=$(find_hf_checkpoint "$ckpt")
        if [[ -n "$hf_ckpt" && -f "$hf_ckpt/config.json" ]]; then
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
    PREV_OUTPUT_DIR="$CKPT_ROOT/task_${LAST_COMPLETED}_${PREV_TASK}"
    CURRENT_MODEL=$(find_hf_checkpoint "$PREV_OUTPUT_DIR")
    if [[ -z "$CURRENT_MODEL" ]]; then
        echo "Error: could not find HF checkpoint in $PREV_OUTPUT_DIR" >&2
        exit 1
    fi
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
        CURRENT_MODEL=$(find_hf_checkpoint "$OUTPUT_DIR")
        continue
    fi

    # ── Delete the checkpoint two tasks back before training ─────────────────
    # Task N trains from task N-1 checkpoint, so we can safely delete task N-2.
    # (task N-2's MMLU already ran at the end of the previous iteration)
    if [[ $TASK_NUM -gt 2 ]]; then
        OLD_TASK_DIR="$CKPT_ROOT/task_$((TASK_NUM - 2))_${TASKS[$((TASK_NUM - 3))]}"
        if [[ -d "$OLD_TASK_DIR" ]]; then
            echo "── Removing old checkpoint $OLD_TASK_DIR to free disk ──"
            rm -rf "$OLD_TASK_DIR"
        fi
    fi

    # ── 4. Train on this task ─────────────────────────────────────────────────
    echo ""
    echo "════ Task $TASK_NUM / ${#TASKS[@]}: $TASK ════"
    echo "    input model : $CURRENT_MODEL"
    echo "    output      : $OUTPUT_DIR"

    LIGER_FLAG=""
    if [[ "$USE_LIGER_KERNELS" == "true" ]]; then
        LIGER_FLAG="--use-liger-kernels"
    fi

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
        --max-epochs "$MAX_EPOCHS" \
        --training-mode epoch \
        $LIGER_FLAG \
        --osft \
        --osft-unfreeze-rank-ratio "$UNFREEZE_RANK_RATIO" \
        --save-final-checkpoint

    # Resolve the actual HF checkpoint (hf_format/samples_N/)
    CURRENT_MODEL=$(find_hf_checkpoint "$OUTPUT_DIR")
    if [[ -z "$CURRENT_MODEL" ]]; then
        echo "Error: training completed but no HF checkpoint found in $OUTPUT_DIR" >&2
        exit 1
    fi
    echo "── Training complete: $CURRENT_MODEL ──"

    # ── 5. TRACE task evaluation on all tasks seen so far ────────────────────
    if [[ "$SKIP_TRACE_EVAL" == "true" ]]; then
        echo "── Skipping TRACE evaluation (--skip-trace-eval) ──"
    else
        echo ""
        echo "── Evaluating TRACE tasks 1–$TASK_NUM after task $TASK_NUM ──"
        SCRIPT_DIR_EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
        for j in $(seq 0 $((TASK_NUM - 1))); do
            EVAL_TASK="${TASKS[$j]}"
            EVAL_OUT="$RESULTS_DIR/trace/${EVAL_TASK}_after_task${TASK_NUM}.json"
            if [[ -f "$EVAL_OUT" ]]; then
                echo "   Skipping $EVAL_TASK (result already exists)"
                continue
            fi
            echo "   Evaluating $EVAL_TASK ..."
            python3 "$SCRIPT_DIR_EVAL/eval_trace_task.py" \
                --model "$CURRENT_MODEL" \
                --task "$EVAL_TASK" \
                --test-file "$TRACE_RAW_DIR/$EVAL_TASK/test.json" \
                --output-file "$EVAL_OUT"
        done
    fi

    # ── MMLU evaluation ───────────────────────────────────────────────────────
    if [[ "$SKIP_MMLU_EVAL" == "true" ]]; then
        echo "── Skipping MMLU evaluation (--skip-mmlu-eval) ──"
    else
        MMLU_OUT="$RESULTS_DIR/mmlu/osft_after_task_${TASK_NUM}"
        # lm_eval ≥0.4 writes results_<timestamp>.json into a model-named subdir
        if find "$MMLU_OUT" -name "results*.json" 2>/dev/null | grep -q .; then
            echo "── Skipping MMLU for task $TASK_NUM (already exists) ──"
        else
            echo ""
            echo "── MMLU evaluation after task $TASK_NUM ──"
            mkdir -p "$MMLU_OUT"
            lm_eval \
                --model hf \
                --model_args "pretrained=$CURRENT_MODEL,dtype=bfloat16" \
                --tasks mmlu \
                --num_fewshot 5 \
                --batch_size auto \
                --output_path "$MMLU_OUT"
            echo "── MMLU complete: $MMLU_OUT ──"
        fi
    fi

    # ── Spectral analysis ────────────────────────────────────────────────────
    if [[ -n "$SPECTRAL_DIR" ]]; then
        SPECTRAL_OUT="$SPECTRAL_DIR/task_${TASK_NUM}_${TASK}"
        if [[ -f "$SPECTRAL_OUT/summary.json" ]]; then
            echo "── Skipping spectral analysis for task $TASK_NUM (already exists) ──"
        else
            echo ""
            echo "── Spectral analysis after task $TASK_NUM ──"
            SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
            python3 "$SCRIPT_DIR/analyze_checkpoint_spectra.py" \
                --checkpoint "$CURRENT_MODEL" \
                --label "task_${TASK_NUM}_${TASK}" \
                --output-dir "$SPECTRAL_OUT" \
                --ratios 0.05 0.10 0.15 0.20 0.25 0.30
            echo "── Spectral analysis complete: $SPECTRAL_OUT ──"
        fi
    fi

done

echo ""
echo "All tasks complete. Next: bash scripts/report.sh"
