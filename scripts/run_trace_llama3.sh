#!/bin/bash
# run_trace_llama3.sh — Full TRACE OSFT experiment for Llama-3.2-3B-Instruct
#
# Phases:
#   1. Tokenize TRACE data
#   2. MMLU baseline on original model
#   3. Build SVD-truncated model + MMLU baseline
#   4. Sequential OSFT training (8 tasks) with MMLU after each task
#   5. Final report
#
# All output is tee'd to $LOG_FILE.
# Resume-safe: each phase checks whether its output already exists.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

MODEL="Qwen/Qwen2.5-3B-Instruct"
TRACE_RAW="/tmp/TRACE"
TRACE_TOK="/tmp/TRACE_tokenized_qwen25"
CKPT_ROOT="/tmp/checkpoints/trace_osft_qwen25"
RESULTS_DIR="$REPO_ROOT/results"
SVD_MODEL="$CKPT_ROOT/svd_truncated_baseline"
LOG_FILE="/tmp/trace_qwen25_run.log"

export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"  # set HF_TOKEN in your environment
# Use SDPA instead of flash-attn (not installed)
export TESTING=true

UNFREEZE_RANK_RATIO="0.10"
SPECTRAL_DIR="$RESULTS_DIR/spectral"

mkdir -p "$CKPT_ROOT" "$RESULTS_DIR/mmlu" "$SPECTRAL_DIR"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── Phase 1: Tokenize ─────────────────────────────────────────────────────────
if [[ -f "$TRACE_TOK/20Minuten/train.jsonl" ]]; then
    log "Phase 1: tokenized data already exists at $TRACE_TOK — skipping"
else
    log "Phase 1: tokenizing TRACE data with $MODEL ..."
    python3 "$REPO_ROOT/scripts/convert_trace_data.py" \
        --model "$MODEL" \
        --trace-dir "$TRACE_RAW" \
        --output-dir "$TRACE_TOK"
    log "Phase 1: done"
fi

# ── Phase 2: MMLU baseline on original model ─────────────────────────────────
if find "$RESULTS_DIR/mmlu/original" -name "results*.json" 2>/dev/null | grep -q .; then
    log "Phase 2: original MMLU already exists — skipping"
else
    log "Phase 2: MMLU baseline on original model ..."
    mkdir -p "$RESULTS_DIR/mmlu/original"
    lm_eval \
        --model hf \
        --model_args "pretrained=$MODEL,dtype=bfloat16" \
        --tasks mmlu \
        --num_fewshot 5 \
        --batch_size auto \
        --output_path "$RESULTS_DIR/mmlu/original"
    log "Phase 2: done"
fi

# ── Phase 3a: Build SVD-truncated model ───────────────────────────────────────
if [[ -f "$SVD_MODEL/config.json" ]]; then
    log "Phase 3a: SVD-truncated model already exists — skipping"
else
    log "Phase 3a: building SVD-truncated baseline ..."
    python3 "$REPO_ROOT/scripts/make_svd_truncated_model.py" \
        --model-path "$MODEL" \
        --output-path "$SVD_MODEL" \
        --unfreeze-rank-ratio "$UNFREEZE_RANK_RATIO"
    log "Phase 3a: done"
fi

# ── Phase 3b: MMLU baseline on SVD-truncated model ───────────────────────────
if find "$RESULTS_DIR/mmlu/svd_truncated" -name "results*.json" 2>/dev/null | grep -q .; then
    log "Phase 3b: SVD-truncated MMLU already exists — skipping"
else
    log "Phase 3b: MMLU baseline on SVD-truncated model ..."
    mkdir -p "$RESULTS_DIR/mmlu/svd_truncated"
    lm_eval \
        --model hf \
        --model_args "pretrained=$SVD_MODEL,dtype=bfloat16" \
        --tasks mmlu \
        --num_fewshot 5 \
        --batch_size auto \
        --output_path "$RESULTS_DIR/mmlu/svd_truncated"
    log "Phase 3b: done"
fi

# ── Phase 3c: Spectral analysis of original model ───────────────────────────
if [[ -f "$SPECTRAL_DIR/original/summary.json" ]]; then
    log "Phase 3c: original spectral analysis already exists — skipping"
else
    log "Phase 3c: spectral analysis of original model ..."
    python3 "$REPO_ROOT/scripts/analyze_checkpoint_spectra.py" \
        --checkpoint "$MODEL" \
        --label original \
        --output-dir "$SPECTRAL_DIR/original" \
        --ratios 0.05 0.10 0.15 0.20 0.25 0.30
    log "Phase 3c: done"
fi

# ── Phase 4 & 5: Sequential OSFT training + per-task MMLU ────────────────────
log "Phase 4+5: sequential OSFT training ..."
bash "$REPO_ROOT/scripts/train_trace_osft.sh" \
    --model "$MODEL" \
    --data-root "$TRACE_TOK" \
    --ckpt-root "$CKPT_ROOT" \
    --results-dir "$RESULTS_DIR" \
    --n-gpus 1 \
    --max-tokens-per-gpu 8192 \
    --batch-size 128 \
    --unfreeze-rank-ratio "$UNFREEZE_RANK_RATIO" \
    --max-epochs 3 \
    --no-liger-kernels \
    --skip-trace-eval \
    --spectral-dir "$SPECTRAL_DIR"
log "Phase 4+5: done"

# ── Phase 6: Report ───────────────────────────────────────────────────────────
log "Phase 6: collecting MMLU results ..."
python3 "$REPO_ROOT/scripts/collect_mmlu.py" --results-dir "$RESULTS_DIR"

# ── Phase 7: Aggregate spectral analysis ─────────────────────────────────────
log "Phase 7: aggregating spectral reports ..."
python3 "$REPO_ROOT/scripts/aggregate_spectral_reports.py" \
    --scan-dir "$SPECTRAL_DIR" \
    --output-dir "$SPECTRAL_DIR/summary"
log "All done. Full log at $LOG_FILE"
