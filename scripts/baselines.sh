#!/bin/bash
# baselines.sh — Phase 3: MMLU baselines before any continual learning
#
# Produces:
#   $RESULTS_DIR/mmlu/original/results.json       — unmodified model MMLU score
#   $CKPT_ROOT/svd_truncated_baseline/            — weight-truncated model checkpoint
#   $RESULTS_DIR/mmlu/svd_truncated/results.json  — truncated model MMLU score
#
# Usage:
#   bash scripts/baselines.sh [--model MODEL] [--ckpt-root DIR] [--results-dir DIR]
#                             [--unfreeze-rank-ratio FLOAT] [--n-gpus N]

set -euo pipefail

# ── defaults ──────────────────────────────────────────────────────────────────
MODEL="meta-llama/Llama-2-7b-chat-hf"
CKPT_ROOT="/checkpoints/trace_osft"
RESULTS_DIR="results"
UNFREEZE_RANK_RATIO="0.25"
N_GPUS=1

# ── argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)                MODEL="$2";               shift 2 ;;
        --ckpt-root)            CKPT_ROOT="$2";           shift 2 ;;
        --results-dir)          RESULTS_DIR="$2";         shift 2 ;;
        --unfreeze-rank-ratio)  UNFREEZE_RANK_RATIO="$2"; shift 2 ;;
        --n-gpus)               N_GPUS="$2";              shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

SVD_CKPT="$CKPT_ROOT/svd_truncated_baseline"
MMLU_DIR="$RESULTS_DIR/mmlu"

echo "================================================================"
echo " TRACE baselines"
echo "  model              : $MODEL"
echo "  SVD checkpoint     : $SVD_CKPT"
echo "  results            : $MMLU_DIR"
echo "  unfreeze rank ratio: $UNFREEZE_RANK_RATIO"
echo "================================================================"

mkdir -p "$MMLU_DIR" "$CKPT_ROOT"

# ── 3a. MMLU on the original model ────────────────────────────────────────────
if [[ -f "$MMLU_DIR/original/results.json" ]]; then
    echo "── Skipping original MMLU (already exists at $MMLU_DIR/original/results.json) ──"
else
    echo ""
    echo "── 3a. Evaluating MMLU on original model ──"
    lm_eval \
        --model hf \
        --model_args "pretrained=$MODEL,dtype=bfloat16" \
        --tasks mmlu \
        --num_fewshot 5 \
        --batch_size auto \
        --output_path "$MMLU_DIR/original"
    echo "── 3a complete: $MMLU_DIR/original/results.json ──"
fi

# ── 3b. Build SVD-truncated model ─────────────────────────────────────────────
if [[ -f "$SVD_CKPT/config.json" ]]; then
    echo "── Skipping SVD truncation (checkpoint already exists at $SVD_CKPT) ──"
else
    echo ""
    echo "── 3b. Building SVD-truncated baseline model ──"
    python scripts/make_svd_truncated_model.py \
        --model-path "$MODEL" \
        --output-path "$SVD_CKPT" \
        --unfreeze-rank-ratio "$UNFREEZE_RANK_RATIO"
    echo "── 3b complete: $SVD_CKPT ──"
fi

# ── 3c. MMLU on the SVD-truncated model ───────────────────────────────────────
if [[ -f "$MMLU_DIR/svd_truncated/results.json" ]]; then
    echo "── Skipping SVD-truncated MMLU (already exists) ──"
else
    echo ""
    echo "── 3c. Evaluating MMLU on SVD-truncated model ──"
    lm_eval \
        --model hf \
        --model_args "pretrained=$SVD_CKPT,dtype=bfloat16" \
        --tasks mmlu \
        --num_fewshot 5 \
        --batch_size auto \
        --output_path "$MMLU_DIR/svd_truncated"
    echo "── 3c complete: $MMLU_DIR/svd_truncated/results.json ──"
fi

echo ""
echo "Baselines done. Next: bash scripts/train_trace_osft.sh"
