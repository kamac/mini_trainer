# TRACE Benchmark Reproduction Plan

This document describes how to reproduce the OSFT results from
[arXiv:2504.07097](https://arxiv.org/abs/2504.07097) ("Sculpting Subspaces: How We Solved
Continual Learning in LLMs") using the TRACE benchmark.

**Target results to reproduce** (Table from the paper, LLaMA-2-7B-Chat):

| Method  | Avg Accuracy | Backward Transfer |
|---------|-------------|-------------------|
| SeqFT   | 23.0%       | -8.3%             |
| O-LoRA  | 41.3%       | 6.2%              |
| **OSFT (ours)** | **48.4%** | **7.1%** |

---

## Overview

TRACE is a continual learning benchmark consisting of **8 sequential tasks** (5,000 training /
2,000 test examples each). Tasks span multilingual understanding, domain-specific knowledge,
arithmetic reasoning, and code generation. Models are trained on tasks one at a time; after each
task the model is evaluated on all tasks seen so far plus a set of general-ability benchmarks.

---

## Step 1: Environment Setup

### Install mini_trainer

```bash
git clone https://github.com/Red-Hat-AI-Innovation-Team/mini_trainer.git
cd mini_trainer
pip install -e .[cuda] --no-build-isolation
```

### Install TRACE evaluation dependencies

```bash
pip install rouge-score sacrebleu transformers datasets accelerate
```

### Install lm-evaluation-harness (for MMLU)

```bash
pip install lm-eval
# Verify:
lm_eval --version
```

### Clone TRACE repository (for evaluation scripts and data)

```bash
git clone https://github.com/BeyonderXX/TRACE.git /opt/TRACE
cd /opt/TRACE
pip install -r requirements.txt
```

---

## Step 2: Download and Prepare Data

### 2a. Download TRACE datasets

The preprocessed TRACE datasets (8 tasks, each with `train.json`, `eval.json`, `test.json`)
are available from the TRACE repository's Google Drive link. Download and extract them:

```bash
# Download from the Google Drive link listed in https://github.com/BeyonderXX/TRACE
# After extraction, directory structure should be:
# /data/TRACE/
#   task1/train.json  eval.json  test.json
#   task2/train.json  ...
#   ...
#   task8/train.json  ...
```

TRACE JSON format per file:
```json
{"prompt": "...", "answer": "..."}
```

### 2b. Tokenize TRACE data for mini_trainer

mini_trainer requires pre-tokenized JSONL with `input_ids`, `labels`, and `len` fields.
Use the `scripts/process_data.py` utility (or `instructlab-training` APIs) to convert
each task's `train.json` into mini_trainer format.

Create a conversion script `scripts/convert_trace_data.py`:

```python
"""Convert TRACE JSON data to mini_trainer tokenized JSONL format."""

import json
from pathlib import Path
from transformers import AutoTokenizer

TRACE_DATA_DIR = Path("/data/TRACE")
OUTPUT_DIR = Path("/data/TRACE_tokenized")
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
IGNORE_INDEX = -100

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

TRACE_TASKS = [
    "C-STANCE", "FOMC", "MeetingBank", "Py150",
    "ScienceQA", "NumGLUE-cm", "NumGLUE-ds", "20Minuten"
]

def convert_task(task_name: str):
    task_in = TRACE_DATA_DIR / task_name
    task_out = OUTPUT_DIR / task_name
    task_out.mkdir(parents=True, exist_ok=True)

    for split in ["train", "eval", "test"]:
        src = task_in / f"{split}.json"
        if not src.exists():
            continue
        records = json.loads(src.read_text())
        out_path = task_out / f"{split}.jsonl"
        with out_path.open("w") as f:
            for rec in records:
                full_text = rec["prompt"] + rec["answer"]
                prompt_ids = tokenizer(rec["prompt"], add_special_tokens=True).input_ids
                full_ids = tokenizer(full_text, add_special_tokens=True).input_ids
                labels = [IGNORE_INDEX] * len(prompt_ids) + full_ids[len(prompt_ids):]
                assert len(full_ids) == len(labels)
                f.write(json.dumps({
                    "input_ids": full_ids,
                    "labels": labels,
                    "len": len(full_ids),
                }) + "\n")

for task in TRACE_TASKS:
    convert_task(task)
    print(f"Converted {task}")
```

---

## Step 2c: MMLU Baseline Evaluations (run once before any training)

Before any continual learning, establish two baselines that will anchor every subsequent
MMLU comparison:

| Baseline | Description |
|----------|-------------|
| **Original** | The unmodified `meta-llama/Llama-2-7b-chat-hf` |
| **SVD-truncated** | Original model with low singular components zeroed out — the *same subspace* that OSFT trains in, but with nothing learned there yet. This isolates the cost of reserving that subspace from the benefit of learning in it. |

### 2c-i. Evaluate MMLU on the original model

```bash
lm_eval \
    --model hf \
    --model_args pretrained=meta-llama/Llama-2-7b-chat-hf,dtype=bfloat16 \
    --tasks mmlu \
    --num_fewshot 5 \
    --batch_size auto \
    --output_path results/mmlu/original
```

### 2c-ii. Create and evaluate the SVD-truncated baseline

OSFT trains exclusively in the subspace spanned by the **lowest** `osft_unfreeze_rank_ratio`
fraction of singular vectors of each weight matrix. The SVD-truncated baseline answers:
*"What accuracy does the model have if we simply erase that subspace — with no new knowledge
written in — compared to OSFT which actively learns in it?"*

Create `scripts/make_svd_truncated_model.py`:

```python
"""
Create an SVD-truncated version of a model by zeroing out the low singular-value
components — exactly the subspace that OSFT trains in.

For LLaMA-2, OSFT targets (per layer):
  self_attn.{q,k,v,o}_proj   and   mlp.{gate,up,down}_proj

With osft_unfreeze_rank_ratio=0.25, the bottom 25% of singular values are zeroed;
the top 75% are kept unchanged.
"""

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Layer suffixes targeted by OSFT for LLaMA-style models (see osft_utils.py lines 200-207)
LLAMA_TARGET_SUFFIXES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


def truncate_model(model_path: str, output_path: str, unfreeze_rank_ratio: float) -> None:
    """Zero out the bottom `unfreeze_rank_ratio` fraction of singular values in
    every targeted weight matrix, leaving the rest of the model untouched."""

    print(f"Loading model from {model_path} ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="cpu"
    )

    n_truncated = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if not any(name.endswith(f"{suffix}.weight") for suffix in LLAMA_TARGET_SUFFIXES):
                continue
            if param.dim() != 2:
                continue

            W = param.data.float()  # SVD is numerically sensitive; use float32
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)

            # Keep only the top (1 - unfreeze_rank_ratio) singular components
            keep_k = max(1, int(round(S.shape[0] * (1.0 - unfreeze_rank_ratio))))
            S[keep_k:] = 0.0  # zero out the low-value components

            param.data = (U @ torch.diag(S) @ Vh).to(param.dtype)
            n_truncated += 1
            print(f"  Truncated {name}: kept {keep_k}/{S.shape[0]} singular values")

    print(f"\nTruncated {n_truncated} weight matrices.")
    print(f"Saving to {output_path} ...")
    Path(output_path).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path)

    # Copy tokenizer files so the output directory is self-contained
    tok = AutoTokenizer.from_pretrained(model_path)
    tok.save_pretrained(output_path)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--unfreeze-rank-ratio", type=float, default=0.25,
                        help="Fraction of singular values to zero out (must match OSFT config)")
    args = parser.parse_args()
    truncate_model(args.model_path, args.output_path, args.unfreeze_rank_ratio)
```

Run it:

```bash
python scripts/make_svd_truncated_model.py \
    --model-path meta-llama/Llama-2-7b-chat-hf \
    --output-path /checkpoints/trace_osft/svd_truncated_baseline \
    --unfreeze-rank-ratio 0.25
```

Then evaluate MMLU on the truncated model:

```bash
lm_eval \
    --model hf \
    --model_args pretrained=/checkpoints/trace_osft/svd_truncated_baseline,dtype=bfloat16 \
    --tasks mmlu \
    --num_fewshot 5 \
    --batch_size auto \
    --output_path results/mmlu/svd_truncated
```

Record the aggregate `acc` score from both runs — these are the fixed anchors for all
subsequent comparisons.

---

## Step 3: Sequential Continual Learning Training

Train on each of the 8 TRACE tasks in order. After each task, save a checkpoint.

### Training configuration (from paper)

| Parameter | Value |
|-----------|-------|
| Model | `meta-llama/Llama-2-7b-chat-hf` |
| Method | OSFT |
| `osft_unfreeze_rank_ratio` | 0.25 (train the 25% least important subspace) |
| Learning rate | 2e-4 |
| Batch size | 128 |
| Max tokens/GPU | 8,192 (adjust for your GPU memory) |
| Epochs per task | 3 |
| LR scheduler | cosine |
| Warmup steps | 100 |
| Optimizer | AdamW (β1=0.9, β2=0.95) |
| Training dtype | bfloat16 |
| Liger kernels | enabled |

> **Note:** Hyperparameters not listed explicitly in the paper default to mini_trainer defaults.
> Consult Appendix A.6 of arXiv:2504.07097 for the authoritative values.

### Training script `scripts/train_trace_osft.sh`

MMLU is evaluated after each task checkpoint is saved. Results land in
`results/mmlu/osft_after_task_N/` and can be compared directly against the two fixed
baselines established in Step 2c.

```bash
#!/bin/bash
set -euo pipefail

MODEL="meta-llama/Llama-2-7b-chat-hf"
DATA_ROOT="/data/TRACE_tokenized"
CKPT_ROOT="/checkpoints/trace_osft"
MMLU_RESULTS="results/mmlu"
N_GPUS=8

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

CURRENT_MODEL="$MODEL"

for i in "${!TASKS[@]}"; do
    TASK="${TASKS[$i]}"
    TASK_NUM=$((i + 1))
    OUTPUT_DIR="$CKPT_ROOT/task_${TASK_NUM}_${TASK}"

    echo "=== Training on Task $TASK_NUM: $TASK ==="

    torchrun \
        --nnodes=1 \
        --nproc-per-node=$N_GPUS \
        -m mini_trainer.train \
        --model-name-or-path "$CURRENT_MODEL" \
        --data-path "$DATA_ROOT/$TASK/train.jsonl" \
        --output-dir "$OUTPUT_DIR" \
        --batch-size 128 \
        --max-tokens-per-gpu 8192 \
        --learning-rate 2e-4 \
        --num-warmup-steps 100 \
        --lr-scheduler cosine \
        --beta1 0.9 \
        --beta2 0.95 \
        --train-dtype bfloat16 \
        --max-epochs 3 \
        --training-mode epoch \
        --use-liger-kernels \
        --osft \
        --osft-unfreeze-rank-ratio 0.25 \
        --save-final-checkpoint \
        --checkpoint-at-epoch

    echo "=== Finished Task $TASK_NUM: $TASK. Checkpoint: $OUTPUT_DIR ==="

    # --- MMLU evaluation after this task ---
    echo "=== MMLU evaluation after task $TASK_NUM ==="
    lm_eval \
        --model hf \
        --model_args pretrained="$OUTPUT_DIR",dtype=bfloat16 \
        --tasks mmlu \
        --num_fewshot 5 \
        --batch_size auto \
        --output_path "$MMLU_RESULTS/osft_after_task_${TASK_NUM}"
    echo "=== MMLU done for task $TASK_NUM ==="

    # Use this task's checkpoint as the starting point for the next task
    CURRENT_MODEL="$OUTPUT_DIR"
done

echo "=== Sequential training + MMLU evaluations complete ==="
```

### Comparison: SeqFT baseline `scripts/train_trace_seqft.sh`

Same script as above but **without** `--osft` and `--osft-unfreeze-rank-ratio`, and with
`MMLU_RESULTS="results/mmlu_seqft"`. This reproduces the SeqFT (sequential full fine-tuning)
baseline. MMLU is still evaluated after each task so catastrophic forgetting of general knowledge
is visible for both methods.

---

## Step 4: Evaluation Protocol

After training on each task $t$, evaluate the model on:
1. **All TRACE tasks 1..t** — compute per-task accuracy
2. **General ability benchmarks** — the TRACE paper uses GSM8K, MMLU, etc. to measure
   catastrophic forgetting of pre-training knowledge

Use the TRACE repository's evaluation scripts:

```bash
#!/bin/bash
# scripts/eval_trace.sh
CKPT="$1"       # path to checkpoint after task T
TASK_IDX="$2"   # integer 1..8 (how many tasks have been trained)

TASKS=("C-STANCE" "FOMC" "MeetingBank" "Py150" "ScienceQA" "NumGLUE-cm" "NumGLUE-ds" "20Minuten")
DATA_ROOT="/data/TRACE"

for i in $(seq 0 $((TASK_IDX - 1))); do
    TASK="${TASKS[$i]}"
    python /opt/TRACE/metrics.py \
        --model "$CKPT" \
        --test_file "$DATA_ROOT/$TASK/test.json" \
        --task "$TASK" \
        --output_file "results/${TASK}_after_task${TASK_IDX}.json"
done
```

Run this after each task checkpoint is saved.

---

## Step 4b: Collect MMLU Scores Across Checkpoints

`lm_eval` writes a `results.json` into each `--output_path` directory. Parse them all into
a single summary with `scripts/collect_mmlu.py`:

```python
"""Collect MMLU accuracy scores across all checkpoints and print a comparison table."""

import json
from pathlib import Path

RESULTS_ROOT = Path("results/mmlu")

CHECKPOINTS = {
    "original":      RESULTS_ROOT / "original",
    "svd_truncated": RESULTS_ROOT / "svd_truncated",
    **{
        f"osft_task_{i}": RESULTS_ROOT / f"osft_after_task_{i}"
        for i in range(1, 9)
    },
}

TASK_NAMES = ["C-STANCE", "FOMC", "MeetingBank", "Py150",
              "ScienceQA", "NumGLUE-cm", "NumGLUE-ds", "20Minuten"]


def extract_mmlu_acc(results_dir: Path) -> float | None:
    """Return aggregate MMLU accuracy from an lm_eval output directory."""
    path = results_dir / "results.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    # lm_eval stores aggregate under key "mmlu" or as average of subtasks
    results = data.get("results", {})
    if "mmlu" in results:
        return results["mmlu"].get("acc,none") or results["mmlu"].get("acc")
    # Fall back: average all mmlu_* subtasks
    subtask_accs = [
        v.get("acc,none") or v.get("acc")
        for k, v in results.items()
        if k.startswith("mmlu_") and isinstance(v, dict)
    ]
    return sum(subtask_accs) / len(subtask_accs) if subtask_accs else None


print(f"{'Checkpoint':<22} {'MMLU acc':>10}  {'Δ vs original':>14}  {'Δ vs SVD-trunc':>15}")
print("-" * 68)

original_acc = extract_mmlu_acc(CHECKPOINTS["original"])
svd_acc = extract_mmlu_acc(CHECKPOINTS["svd_truncated"])

for label, results_dir in CHECKPOINTS.items():
    acc = extract_mmlu_acc(results_dir)
    acc_str   = f"{acc*100:.2f}%" if acc is not None else "N/A"
    delta_orig = f"{(acc - original_acc)*100:+.2f}%" if (acc and original_acc) else "N/A"
    delta_svd  = f"{(acc - svd_acc)*100:+.2f}%"     if (acc and svd_acc)      else "N/A"
    print(f"{label:<22} {acc_str:>10}  {delta_orig:>14}  {delta_svd:>15}")
```

Run after all training is complete (or incrementally after each task):

```bash
python scripts/collect_mmlu.py
```

Example output shape (fill in actuals):

```
Checkpoint             MMLU acc   Δ vs original   Δ vs SVD-trunc
--------------------------------------------------------------------
original                  63.45%           —               —
svd_truncated             62.10%       -1.35%              —
osft_task_1               62.80%       -0.65%          +0.70%
osft_task_2               62.50%       -0.95%          +0.40%
...
osft_task_8               61.90%       -1.55%          -0.20%
```

**How to read this:**

- **Δ vs original**: total MMLU change attributable to continual training (negative = some
  general knowledge was overwritten).
- **Δ vs SVD-truncated**: how much of the accuracy loss (or gain!) comes from *actively learning
  in the low-singular-value subspace* versus merely *reserving it*. If this delta is near zero,
  OSFT is using its training budget in a way that is neutral to general knowledge; if positive,
  the learned weights actually help on MMLU.

---

## Step 5: Metrics Computation

Let $A_{i,j}$ be the accuracy on task $i$ after training on task $j$ (where $j \geq i$).

### Average Accuracy (after all 8 tasks)

$$\text{AvgAcc} = \frac{1}{T} \sum_{i=1}^{T} A_{i,T}$$

### Backward Transfer

$$\text{BWT} = \frac{1}{T-1} \sum_{i=1}^{T-1} (A_{i,T} - A_{i,i})$$

Positive BWT means learning new tasks improved performance on old ones (desirable).
Negative BWT indicates catastrophic forgetting.

Implement in `scripts/compute_metrics.py`:

```python
"""Compute TRACE continual learning metrics from per-task evaluation results."""

import json
import sys
from pathlib import Path

TASKS = ["C-STANCE", "FOMC", "MeetingBank", "Py150",
         "ScienceQA", "NumGLUE-cm", "NumGLUE-ds", "20Minuten"]
T = len(TASKS)

# A[i][j] = accuracy on task i after training on task j (0-indexed, j >= i)
A = {}
results_dir = Path("results")

for j in range(T):          # after training on task j+1
    for i in range(j + 1):  # evaluate tasks 0..j
        task = TASKS[i]
        path = results_dir / f"{task}_after_task{j+1}.json"
        data = json.loads(path.read_text())
        A[(i, j)] = data["accuracy"]

# Average accuracy after all tasks
avg_acc = sum(A[(i, T-1)] for i in range(T)) / T

# Backward transfer
bwt = sum(A[(i, T-1)] - A[(i, i)] for i in range(T-1)) / (T - 1)

print(f"Average Accuracy: {avg_acc*100:.1f}%")
print(f"Backward Transfer: {bwt*100:.1f}%")
```

---

## Step 6: Expected Results and Validation

### 6a. TRACE continual-learning metrics

After completing the OSFT run, compare against:

| Metric | Paper (OSFT) | Reproduced |
|--------|-------------|------------|
| Avg Accuracy | 48.4% | ___ |
| Backward Transfer | +7.1% | ___ |

A successful reproduction is within ±1–2% of these numbers. Common causes of deviation:
- Different task ordering (TRACE paper uses a fixed order)
- Different random seed
- Different `osft_unfreeze_rank_ratio`
- GPU count / effective batch size differences affecting convergence

### 6b. MMLU tracking table

Fill in from `python scripts/collect_mmlu.py` after all 8 tasks:

| Checkpoint | MMLU acc | Δ vs original | Δ vs SVD-truncated |
|------------|----------|--------------|-------------------|
| Original (Llama-2-7b-chat-hf) | ___ | — | — |
| SVD-truncated (top 75% kept) | ___ | ___ | — |
| After task 1 (C-STANCE) | ___ | ___ | ___ |
| After task 2 (FOMC) | ___ | ___ | ___ |
| After task 3 (MeetingBank) | ___ | ___ | ___ |
| After task 4 (Py150) | ___ | ___ | ___ |
| After task 5 (ScienceQA) | ___ | ___ | ___ |
| After task 6 (NumGLUE-cm) | ___ | ___ | ___ |
| After task 7 (NumGLUE-ds) | ___ | ___ | ___ |
| After task 8 (20Minuten) | ___ | ___ | ___ |

**Key interpretation:**

- If **Δ vs SVD-truncated ≈ 0** throughout: OSFT's training in the low-singular subspace is
  effectively neutral to general knowledge — the reserved subspace is a "free lunch" for
  continual learning.
- If **Δ vs SVD-truncated > 0**: training in that subspace actually *improves* MMLU — the
  low-singular-value directions contain useful general capacity.
- If **Δ vs SVD-truncated < 0**: the new task knowledge written into the subspace partially
  overwrites general-purpose representations that happened to live there.
- The **Δ vs original** column tracks total MMLU drift; for OSFT this should be much smaller
  than for SeqFT, since OSFT confines updates to the least-important subspace.

---

## Step 7: Troubleshooting

| Issue | Resolution |
|-------|-----------|
| OOM during OSFT SVD init | Reduce `--max-tokens-per-gpu` or use fewer tasks per run |
| Tokenization mismatch | Ensure same tokenizer as base model; check BOS/EOS handling |
| Accuracy much lower than baseline | Confirm task order matches TRACE paper; check label masking |
| Checkpoint not loading for next task | Verify `--save-final-checkpoint` is set; check `OUTPUT_DIR` path |

---

## References

- OSFT Paper: [arXiv:2504.07097](https://arxiv.org/abs/2504.07097)
- TRACE Paper: [arXiv:2310.06762](https://arxiv.org/abs/2310.06762)
- TRACE GitHub: [BeyonderXX/TRACE](https://github.com/BeyonderXX/TRACE)
- mini_trainer: [Red-Hat-AI-Innovation-Team/mini_trainer](https://github.com/Red-Hat-AI-Innovation-Team/mini_trainer)
