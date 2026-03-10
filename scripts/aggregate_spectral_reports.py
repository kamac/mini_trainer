"""Aggregate per-checkpoint spectral analyses into a summary table and charts.

Reads summary.json files produced by analyze_checkpoint_spectra.py and generates:
  - spectral_summary.csv   — table of spectral mass at each ratio per checkpoint
  - ratio_selection.png     — spectral mass vs ratio, one curve per checkpoint
  - spectral_evolution.png  — mean singular value curves overlaid across checkpoints

Usage:
    python scripts/aggregate_spectral_reports.py \
        --report-dirs results/spectral/original \
                      results/spectral/task_1_C-STANCE \
                      results/spectral/task_2_FOMC \
        --output-dir results/spectral/summary

    # Or auto-discover all subdirectories:
    python scripts/aggregate_spectral_reports.py \
        --scan-dir results/spectral \
        --output-dir results/spectral/summary
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def load_summaries(report_dirs: list[Path]) -> list[dict]:
    """Load summary.json from each directory, skip missing."""
    summaries = []
    for d in report_dirs:
        p = d / "summary.json"
        if not p.exists():
            print(f"  Warning: {p} not found, skipping")
            continue
        summaries.append(json.loads(p.read_text()))
    return summaries


def print_and_save_table(summaries: list[dict], ratios: list[str],
                         output_dir: Path) -> None:
    """Print summary table to stdout and write CSV."""
    col_w = 12
    label_w = 25

    # Header
    print(f"\n{'Checkpoint':<{label_w}}", end="")
    for r in ratios:
        print(f"  {'mass@' + r:>{col_w}}", end="")
    print()
    print("-" * (label_w + (col_w + 2) * len(ratios)))

    # Rows
    csv_rows = []
    for s in summaries:
        row = {"Checkpoint": s["label"]}
        print(f"{s['label']:<{label_w}}", end="")
        for r in ratios:
            mass = s["ratios"].get(r, 0)
            pct = f"{mass * 100:.2f}%"
            row[f"mass@{r}"] = pct
            print(f"  {pct:>{col_w}}", end="")
        print()
        csv_rows.append(row)

    # Save CSV
    csv_path = output_dir / "spectral_summary.csv"
    fieldnames = ["Checkpoint"] + [f"mass@{r}" for r in ratios]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\nSaved: {csv_path}")


def plot_ratio_selection(summaries: list[dict], ratios: list[str],
                         output_dir: Path) -> None:
    """Plot spectral mass vs ratio — the key chart for choosing unfreeze_rank_ratio."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ratio_floats = [float(r) for r in ratios]
    cmap = plt.cm.tab10
    n = len(summaries)

    for i, s in enumerate(summaries):
        masses = [s["ratios"].get(r, 0) * 100 for r in ratios]
        color = cmap(i / max(n - 1, 1))
        ax.plot(ratio_floats, masses, "o-", label=s["label"],
                color=color, linewidth=2, markersize=5)

    ax.set_xlabel("Unfreeze rank ratio", fontsize=11)
    ax.set_ylabel("Spectral mass in trainable subspace (%)", fontsize=11)
    ax.set_title("How much model knowledge is exposed at each OSFT ratio?")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(ratio_floats)

    out = output_dir / "ratio_selection.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_spectral_evolution(summaries: list[dict], output_dir: Path) -> None:
    """Plot mean singular value curves — shows how the spectrum shifts per task."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    cmap = plt.cm.tab10
    n = len(summaries)
    grid = np.linspace(0, 1, 200)

    for i, s in enumerate(summaries):
        curve = np.array(s.get("mean_sv_curve", []))
        if len(curve) == 0:
            continue
        color = cmap(i / max(n - 1, 1))
        ax1.plot(grid, curve, label=s["label"], color=color, linewidth=1.5)
        ax2.plot(grid, curve, label=s["label"], color=color, linewidth=1.5)

    ax1.set_xlabel("Normalized rank (0=largest, 1=smallest)")
    ax1.set_ylabel("Singular value")
    ax1.set_title("Full Spectrum")
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)

    # Zoomed on the low-rank region (bottom 30%)
    ax2.set_xlim(0.70, 1.0)
    ax2.set_xlabel("Normalized rank")
    ax2.set_ylabel("Singular value")
    ax2.set_title("Low-rank subspace (bottom 30%)")
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)

    out = output_dir / "spectral_evolution.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--report-dirs", nargs="+",
                       help="Directories containing summary.json files, in order")
    group.add_argument("--scan-dir",
                       help="Parent directory to scan for subdirs with summary.json")
    parser.add_argument("--output-dir", required=True,
                        help="Directory for summary CSV and plots")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve report directories
    if args.report_dirs:
        report_dirs = [Path(d) for d in args.report_dirs]
    else:
        scan = Path(args.scan_dir)
        report_dirs = sorted(
            d for d in scan.iterdir()
            if d.is_dir() and (d / "summary.json").exists() and d.name != "summary"
        )
        if not report_dirs:
            print(f"No summary.json files found under {scan}")
            sys.exit(1)
        print(f"Found {len(report_dirs)} reports under {scan}")

    summaries = load_summaries(report_dirs)
    if not summaries:
        print("No summaries loaded.")
        sys.exit(1)

    # Get ratio keys from first summary
    ratios = sorted(summaries[0]["ratios"].keys())

    print_and_save_table(summaries, ratios, output_dir)

    if not HAS_MATPLOTLIB:
        print("\nInstall matplotlib for plots: pip install matplotlib")
        return

    plot_ratio_selection(summaries, ratios, output_dir)
    plot_spectral_evolution(summaries, output_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
