"""Collect MMLU accuracy scores across all checkpoints and print a comparison table.

Reads lm_eval results.json files from:
  $results_dir/mmlu/original/
  $results_dir/mmlu/svd_truncated/
  $results_dir/mmlu/osft_after_task_1/ .. osft_after_task_8/

Usage:
    python scripts/collect_mmlu.py [--results-dir results]
"""

import argparse
import json
from pathlib import Path

TASKS = [
    "C-STANCE", "FOMC", "MeetingBank", "Py150",
    "ScienceQA", "NumGLUE-cm", "NumGLUE-ds", "20Minuten",
]


def extract_mmlu_acc(results_dir: Path) -> float | None:
    path = results_dir / "results.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    results = data.get("results", {})

    # lm_eval ≥0.4 stores aggregate under "mmlu" key
    if "mmlu" in results:
        entry = results["mmlu"]
        val = entry.get("acc,none")
        return val if val is not None else entry.get("acc")

    # Fall back: average all mmlu_* subtask entries
    subtask_accs = [
        (v.get("acc,none") if v.get("acc,none") is not None else v.get("acc"))
        for k, v in results.items()
        if k.startswith("mmlu_") and isinstance(v, dict)
    ]
    if subtask_accs:
        return sum(subtask_accs) / len(subtask_accs)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    mmlu_root = Path(args.results_dir) / "mmlu"

    checkpoints: list[tuple[str, Path]] = [
        ("original",      mmlu_root / "original"),
        ("svd_truncated", mmlu_root / "svd_truncated"),
        *[(f"osft_task_{i}", mmlu_root / f"osft_after_task_{i}") for i in range(1, 9)],
    ]

    original_acc = extract_mmlu_acc(mmlu_root / "original")
    svd_acc      = extract_mmlu_acc(mmlu_root / "svd_truncated")

    col_label  = 30  # "osft_task_8 (20Minuten)" = 23 chars; extra headroom
    col_acc    = 10
    col_dorig  = 15
    col_dsvd   = 16

    header = (
        f"{'Checkpoint':<{col_label}}"
        f"{'MMLU acc':>{col_acc}}"
        f"{'Δ vs original':>{col_dorig}}"
        f"{'Δ vs SVD-trunc':>{col_dsvd}}"
    )
    print(header)
    print("-" * len(header))

    after_task_label = {i: TASKS[i - 1] for i in range(1, 9)}

    for label, results_dir in checkpoints:
        acc = extract_mmlu_acc(results_dir)

        if acc is None:
            acc_str  = "N/A"
            d_orig   = "N/A"
            d_svd    = "N/A"
        else:
            acc_str = f"{acc * 100:.2f}%"
            d_orig  = f"{(acc - original_acc) * 100:+.2f}%" if original_acc is not None else "N/A"
            d_svd   = f"{(acc - svd_acc) * 100:+.2f}%"     if svd_acc      is not None else "N/A"

        # Annotate task-checkpoint rows with the TRACE task name
        display = label
        if label.startswith("osft_task_"):
            task_idx = int(label.split("_")[-1])
            display = f"{label} ({after_task_label[task_idx]})"

        print(
            f"{display:<{col_label}}"
            f"{acc_str:>{col_acc}}"
            f"{d_orig:>{col_dorig}}"
            f"{d_svd:>{col_dsvd}}"
        )

    print()
    print("Δ vs original  : total MMLU change from continual learning")
    print("Δ vs SVD-trunc : effect of learning in the low-singular subspace vs. merely reserving it")


if __name__ == "__main__":
    main()
