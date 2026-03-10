"""Analyze singular value spectrum of a single model checkpoint.

For each OSFT-targeted weight matrix, computes SVD and measures how much
spectral energy (knowledge) lives in the low-rank subspace at various
candidate unfreeze ratios.

Usage:
    # Before training (original model):
    python scripts/analyze_checkpoint_spectra.py \
        --checkpoint Qwen/Qwen2.5-3B-Instruct \
        --label original \
        --output-dir results/spectral/original

    # After a training task:
    python scripts/analyze_checkpoint_spectra.py \
        --checkpoint /tmp/checkpoints/.../hf_format/samples_15000.0 \
        --label task_1_C-STANCE \
        --output-dir results/spectral/task_1_C-STANCE

Outputs:
    stats.csv     — per (layer, matrix_type) spectral mass at each ratio
    summary.json  — checkpoint-level averages for aggregation script
"""

import argparse
import csv
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

DEFAULT_RATIOS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

# Per-model_type target suffixes (mirrors make_svd_truncated_model.py)
_TARGET_SUFFIXES_BY_MODEL_TYPE = {
    "llama": (
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
        "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    ),
    "mistral": (
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
        "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    ),
    "gemma": (
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
        "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    ),
    "qwen2": (
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
        "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    ),
    "opt": (
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
        "self_attn.out_proj", "fc1", "fc2",
    ),
}

# Matches layer weight names across architectures
_LAYER_RE = re.compile(
    r"(?:model\.(?:decoder\.)?layers|transformer\.h)\.(\d+)\.(.+)\.weight$"
)


def _get_target_suffixes(model_dir: str) -> tuple[str, ...]:
    """Auto-detect target suffixes from config.json."""
    config_path = Path(model_dir) / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())
        model_type = config.get("model_type", "").lower()
    else:
        model_type = ""
    if model_type in _TARGET_SUFFIXES_BY_MODEL_TYPE:
        return _TARGET_SUFFIXES_BY_MODEL_TYPE[model_type]
    for key in _TARGET_SUFFIXES_BY_MODEL_TYPE:
        if model_type.startswith(key):
            return _TARGET_SUFFIXES_BY_MODEL_TYPE[key]
    # Default to llama-style
    return _TARGET_SUFFIXES_BY_MODEL_TYPE["llama"]


def resolve_local_path(path_or_id: str) -> str:
    """Resolve a checkpoint path or HF model ID to a local directory."""
    if os.path.isdir(path_or_id):
        return path_or_id
    from huggingface_hub import snapshot_download
    try:
        return snapshot_download(
            path_or_id, local_files_only=True,
            ignore_patterns=["*.bin", "*.pt", "original/"],
        )
    except Exception:
        return snapshot_download(
            path_or_id,
            ignore_patterns=["*.bin", "*.pt", "original/"],
        )


def iter_target_tensors(model_dir, suffixes):
    """Yield (name, layer_idx, matrix_type, tensor) for each target weight."""
    model_dir = Path(model_dir)
    index_path = model_dir / "model.safetensors.index.json"
    single_path = model_dir / "model.safetensors"

    if index_path.exists():
        weight_map = json.load(open(index_path))["weight_map"]
        shard_to_keys = {}
        for key, shard in weight_map.items():
            shard_to_keys.setdefault(shard, []).append(key)
    elif single_path.exists():
        with safe_open(str(single_path), framework="pt") as f:
            shard_to_keys = {single_path.name: list(f.keys())}
    else:
        raise FileNotFoundError(f"No safetensors files in {model_dir}")

    for shard_name in sorted(shard_to_keys):
        targets = []
        for key in shard_to_keys[shard_name]:
            m = _LAYER_RE.match(key)
            if m is None:
                continue
            layer_idx, suffix = int(m.group(1)), m.group(2)
            if any(suffix.endswith(s) for s in suffixes):
                targets.append((key, layer_idx, suffix))
        if not targets:
            continue
        shard_path = model_dir / shard_name
        with safe_open(str(shard_path), framework="pt") as f:
            for key, layer_idx, suffix in sorted(targets):
                yield key, layer_idx, suffix, f.get_tensor(key).float()


def compute_spectral_metrics(S: np.ndarray, ratios: list[float]) -> dict:
    """Compute spectral mass at each candidate ratio."""
    s_sq = S ** 2
    total_energy = float(s_sq.sum())
    p = len(S)

    metrics = {}
    for r in ratios:
        k = max(1, int(round(r * p)))
        metrics[f"mass_{r:.2f}"] = float(s_sq[-k:].sum() / total_energy)

    metrics["sv_max"] = float(S[0])
    metrics["sv_min"] = float(S[-1])
    metrics["sv_mean"] = float(S.mean())
    metrics["condition_number"] = float(S[0] / max(S[-1], 1e-15))
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Path to checkpoint dir or HF model ID")
    parser.add_argument("--label", required=True,
                        help="Human-readable label (e.g. 'original', 'task_1_C-STANCE')")
    parser.add_argument("--output-dir", required=True,
                        help="Directory for stats.csv and summary.json")
    parser.add_argument("--ratios", type=float, nargs="+", default=DEFAULT_RATIOS,
                        help="Candidate unfreeze ratios (default: 0.05 0.10 0.15 0.20 0.25 0.30)")
    parser.add_argument("--model-type", default=None,
                        help="Override model type for target suffix selection")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ratios = sorted(args.ratios)

    t0 = time.time()
    print(f"Resolving: {args.checkpoint}")
    local_dir = resolve_local_path(args.checkpoint)
    print(f"Local path: {local_dir}")

    if args.model_type:
        suffixes = _TARGET_SUFFIXES_BY_MODEL_TYPE.get(
            args.model_type, _TARGET_SUFFIXES_BY_MODEL_TYPE["llama"]
        )
    else:
        suffixes = _get_target_suffixes(local_dir)
    print(f"Target suffixes: {suffixes}")

    rows = []
    N_INTERP = 200
    sv_curves = []

    for key, layer_idx, suffix, tensor in iter_target_tensors(local_dir, suffixes):
        # Transpose wide matrices for faster SVD (LAPACK is faster on tall inputs)
        if tensor.shape[0] < tensor.shape[1]:
            tensor = tensor.T.contiguous()

        S = torch.linalg.svdvals(tensor).numpy()
        del tensor

        metrics = compute_spectral_metrics(S, ratios)
        metrics["layer_idx"] = layer_idx
        metrics["matrix_type"] = suffix
        rows.append(metrics)

        # Interpolate to common grid for the mean curve
        rank_frac = np.linspace(0, 1, len(S))
        sv_curves.append(np.interp(np.linspace(0, 1, N_INTERP), rank_frac, S))

        mass_25 = metrics.get("mass_0.25", metrics.get(f"mass_{ratios[-1]:.2f}", 0))
        print(f"  layer {layer_idx:2d} {suffix:<25s}  "
              f"mass@0.25={mass_25:.4f}  cond={metrics['condition_number']:.0f}")

    elapsed = time.time() - t0

    # Write stats CSV
    csv_path = output_dir / "stats.csv"
    ratio_cols = [f"mass_{r:.2f}" for r in ratios]
    fieldnames = ["layer_idx", "matrix_type"] + ratio_cols + \
                 ["sv_max", "sv_min", "sv_mean", "condition_number"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fieldnames})
    print(f"\nSaved: {csv_path}")

    # Build and write summary
    summary = {
        "label": args.label,
        "checkpoint": args.checkpoint,
        "num_layers": max(r["layer_idx"] for r in rows) + 1 if rows else 0,
        "num_matrices": len(rows),
        "elapsed_seconds": round(elapsed, 1),
        "ratios": {},
        "mean_sv_curve": np.mean(sv_curves, axis=0).tolist() if sv_curves else [],
    }
    for r in ratios:
        col = f"mass_{r:.2f}"
        vals = [row[col] for row in rows]
        summary["ratios"][f"{r:.2f}"] = round(float(np.mean(vals)), 6)

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {summary_path}")

    # Print summary table
    print(f"\n{'Ratio':<10} {'Mean spectral mass':>20}")
    print("-" * 32)
    for r in ratios:
        mass = summary["ratios"][f"{r:.2f}"]
        print(f"  {r:.2f}      {mass * 100:>18.2f}%")
    print(f"\nDone in {elapsed:.0f}s ({len(rows)} matrices)")


if __name__ == "__main__":
    main()
