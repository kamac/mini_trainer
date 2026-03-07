"""Compute TRACE continual-learning metrics from per-task evaluation results.

Reads results/<TASK>_after_task<N>.json files produced by eval_trace.sh and
computes Average Accuracy and Backward Transfer.

Usage:
    python scripts/compute_metrics.py [--results-dir results]
"""

import argparse
import json
from pathlib import Path

TASKS = [
    "C-STANCE", "FOMC", "MeetingBank", "Py150",
    "ScienceQA", "NumGLUE-cm", "NumGLUE-ds", "20Minuten",
]
T = len(TASKS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    results_dir = Path(args.results_dir)

    # A[i][j] = accuracy on task i after training through task j+1 (0-indexed, j >= i)
    A: dict[tuple[int, int], float] = {}
    missing = []

    for j in range(T):
        for i in range(j + 1):
            task = TASKS[i]
            path = results_dir / f"{task}_after_task{j + 1}.json"
            if not path.exists():
                missing.append(str(path))
                continue
            data = json.loads(path.read_text())
            A[(i, j)] = data["accuracy"]

    if missing:
        print(f"Warning: {len(missing)} result file(s) not found — metrics will be partial.")
        for p in missing:
            print(f"  missing: {p}")

    # Per-task accuracy after all tasks completed
    print(f"\n{'Task':<20} {'Acc@own':>9} {'Acc@final':>10} {'BWT contrib':>12}")
    print("-" * 56)
    for i, task in enumerate(TASKS):
        acc_own   = A.get((i, i))
        acc_final = A.get((i, T - 1))
        bwt_contrib = (acc_final - acc_own) if (acc_final is not None and acc_own is not None) else None
        print(
            f"{task:<20}"
            f" {acc_own*100:>8.1f}%" if acc_own is not None else f" {'N/A':>9}",
            end="",
        )
        print(
            f" {acc_final*100:>9.1f}%" if acc_final is not None else f" {'N/A':>10}",
            end="",
        )
        print(
            f" {bwt_contrib*100:>+11.1f}%" if bwt_contrib is not None else f" {'N/A':>12}",
        )

    # Aggregate metrics
    final_accs = [A[(i, T - 1)] for i in range(T) if (i, T - 1) in A]
    bwt_pairs  = [(A[(i, T - 1)] - A[(i, i)]) for i in range(T - 1)
                  if (i, T - 1) in A and (i, i) in A]

    print()
    if final_accs:
        avg_acc = sum(final_accs) / len(final_accs)
        print(f"Average Accuracy : {avg_acc * 100:.1f}%  (target: 48.4%)")
    else:
        print("Average Accuracy : N/A")

    if bwt_pairs:
        bwt = sum(bwt_pairs) / len(bwt_pairs)
        print(f"Backward Transfer: {bwt * 100:+.1f}%  (target: +7.1%)")
    else:
        print("Backward Transfer: N/A")


if __name__ == "__main__":
    main()
