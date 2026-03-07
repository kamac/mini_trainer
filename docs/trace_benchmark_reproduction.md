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

```bash
#!/bin/bash
set -euo pipefail

MODEL="meta-llama/Llama-2-7b-chat-hf"
DATA_ROOT="/data/TRACE_tokenized"
CKPT_ROOT="/checkpoints/trace_osft"
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

    # Use this task's checkpoint as the starting point for the next task
    CURRENT_MODEL="$OUTPUT_DIR"
    echo "=== Finished Task $TASK_NUM: $TASK. Checkpoint: $OUTPUT_DIR ==="
done

echo "=== Sequential training complete ==="
```

### Comparison: SeqFT baseline `scripts/train_trace_seqft.sh`

Same script as above but **without** `--osft` and `--osft-unfreeze-rank-ratio`. This reproduces
the SeqFT (sequential full fine-tuning) baseline.

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
