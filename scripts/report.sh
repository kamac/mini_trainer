#!/bin/bash
# report.sh — Phase 6: compute and print all metrics
#
# Runs both metric scripts and writes a combined report to $RESULTS_DIR/report.txt
#
# Usage:
#   bash scripts/report.sh [--results-dir DIR]

set -euo pipefail

RESULTS_DIR="results"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --results-dir) RESULTS_DIR="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

REPORT="$RESULTS_DIR/report.txt"
mkdir -p "$RESULTS_DIR"

{
    echo "================================================================"
    echo " TRACE + MMLU report  ($(date -u '+%Y-%m-%d %H:%M UTC'))"
    echo "================================================================"
    echo ""
    echo "── TRACE continual-learning metrics ────────────────────────────"
    python scripts/compute_metrics.py --results-dir "$RESULTS_DIR"
    echo ""
    echo "── MMLU across checkpoints ─────────────────────────────────────"
    python scripts/collect_mmlu.py --results-dir "$RESULTS_DIR"
} | tee "$REPORT"

echo ""
echo "Report written to $REPORT"
