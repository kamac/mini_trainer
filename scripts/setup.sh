#!/bin/bash
# setup.sh — Phase 1 & 2: environment setup and data preparation
#
# Usage:
#   bash scripts/setup.sh [--model MODEL] [--data-dir DIR] [--trace-dir DIR]
#
# Defaults:
#   MODEL      = meta-llama/Llama-2-7b-chat-hf
#   DATA_DIR   = /data/TRACE_tokenized
#   TRACE_DIR  = /data/TRACE
#
# The TRACE raw data must already be downloaded and extracted to TRACE_DIR.
# Download from the Google Drive link in https://github.com/BeyonderXX/TRACE

set -euo pipefail

# ── defaults ──────────────────────────────────────────────────────────────────
MODEL="meta-llama/Llama-2-7b-chat-hf"
TRACE_DIR="/data/TRACE"
DATA_DIR="/data/TRACE_tokenized"

# ── argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)       MODEL="$2";     shift 2 ;;
        --data-dir)    DATA_DIR="$2";  shift 2 ;;
        --trace-dir)   TRACE_DIR="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

echo "================================================================"
echo " TRACE benchmark setup"
echo "  model     : $MODEL"
echo "  raw data  : $TRACE_DIR"
echo "  tokenized : $DATA_DIR"
echo "================================================================"

# ── Pre-flight check ──────────────────────────────────────────────────────────
# Check for the raw TRACE data before spending time on pip installs.
if [[ ! -d "$TRACE_DIR" ]]; then
    echo "ERROR: TRACE raw data directory not found: $TRACE_DIR"
    echo "Download it from the Google Drive link in https://github.com/BeyonderXX/TRACE"
    echo "then re-run this script."
    exit 1
fi

# ── Phase 1: install dependencies ─────────────────────────────────────────────
echo ""
echo "── Phase 1: installing dependencies ──"

pip install -e ".[cuda]" --no-build-isolation

pip install \
    rouge-score \
    sacrebleu \
    transformers \
    datasets \
    accelerate \
    lm-eval

if [[ ! -d /opt/TRACE ]]; then
    git clone https://github.com/BeyonderXX/TRACE.git /opt/TRACE
fi
pip install -r /opt/TRACE/requirements.txt

echo "── Phase 1 complete ──"

# ── Phase 2: tokenize TRACE data ──────────────────────────────────────────────
echo ""
echo "── Phase 2: tokenizing TRACE data ──"

python scripts/convert_trace_data.py \
    --model "$MODEL" \
    --trace-dir "$TRACE_DIR" \
    --output-dir "$DATA_DIR"

echo "── Phase 2 complete ──"
echo ""
echo "Setup done. Next: bash scripts/baselines.sh"
