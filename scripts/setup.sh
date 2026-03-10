#!/bin/bash
# setup.sh — Phase 1, 2 & 3: environment setup, data download, and tokenization
#
# Usage:
#   bash scripts/setup.sh [--model MODEL] [--data-dir DIR] [--trace-dir DIR]
#
# Defaults:
#   MODEL      = meta-llama/Llama-2-7b-chat-hf
#   DATA_DIR   = /data/TRACE_tokenized
#   TRACE_DIR  = /data/TRACE
#
# If TRACE_DIR does not exist, the dataset is downloaded automatically from
# Google Drive (https://github.com/BeyonderXX/TRACE) using gdown.

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

# ── Phase 1: install dependencies ─────────────────────────────────────────────
echo ""
echo "── Phase 1: installing dependencies ──"

pip install -e ".[cuda]" --no-build-isolation

pip install \
    gdown \
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

# ── Phase 2: download TRACE dataset (if not already present) ──────────────────
echo ""
echo "── Phase 2: downloading TRACE dataset ──"

GDRIVE_FILE_ID="1S0SmU0WEw5okW_XvP2Ns0URflNzZq6sV"

if [[ -d "$TRACE_DIR" ]]; then
    echo "  TRACE data already present at $TRACE_DIR — skipping download."
else
    TRACE_ARCHIVE="$(mktemp /tmp/TRACE_XXXXXX)"
    echo "  Downloading from Google Drive …"
    gdown "https://drive.google.com/uc?id=$GDRIVE_FILE_ID" \
        -O "$TRACE_ARCHIVE" --fuzzy

    TRACE_TMP="$(mktemp -d /tmp/TRACE_extract_XXXXXX)"
    echo "  Extracting …"
    if python3 -c "import zipfile, sys; zipfile.ZipFile(sys.argv[1])" \
            "$TRACE_ARCHIVE" 2>/dev/null; then
        unzip -q "$TRACE_ARCHIVE" -d "$TRACE_TMP"
    else
        tar -xf "$TRACE_ARCHIVE" -C "$TRACE_TMP"
    fi
    rm "$TRACE_ARCHIVE"

    # If the archive extracted a single top-level subdirectory, use that.
    mapfile -t EXTRACTED < <(find "$TRACE_TMP" -mindepth 1 -maxdepth 1 -type d)
    if [[ ${#EXTRACTED[@]} -eq 1 ]]; then
        mv "${EXTRACTED[0]}" "$TRACE_DIR"
        rm -rf "$TRACE_TMP"
    else
        mv "$TRACE_TMP" "$TRACE_DIR"
    fi
    echo "  TRACE dataset extracted to $TRACE_DIR"
fi

echo "── Phase 2 complete ──"

# ── Phase 3: tokenize TRACE data ──────────────────────────────────────────────
echo ""
echo "── Phase 3: tokenizing TRACE data ──"

python scripts/convert_trace_data.py \
    --model "$MODEL" \
    --trace-dir "$TRACE_DIR" \
    --output-dir "$DATA_DIR"

echo "── Phase 3 complete ──"
echo ""
echo "Setup done. Next: bash scripts/baselines.sh"
