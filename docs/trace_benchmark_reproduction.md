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

## Supported models

The pipeline supports any model whose architecture is recognised by OSFT. Verified configs:

| Model | `model_type` | Gated | Notes |
|-------|-------------|-------|-------|
| `meta-llama/Llama-2-7b-chat-hf` | `llama` | Yes (Meta review) | Paper model |
| `meta-llama/Llama-3.2-3B-Instruct` | `llama` | Yes (accept license) | Faster smoke-test |
| `Qwen/Qwen2.5-3B-Instruct` | `qwen2` | No | Ungated, good baseline |
| `facebook/opt-125m` | `opt` | No | Tiny smoke-test |

> **Note on Qwen2.5:** the SVD-truncated baseline drops ~42 pp on MMLU for Qwen2.5-3B
> (vs ~1 pp for OPT-125m and ~1–2 pp expected for LLaMA-2-7B). The low-singular-value
> subspace carries significantly more general knowledge in Qwen2.5 than in the paper's
> target model, making the SVD-truncated baseline uninformative and OSFT's forgetting
> protection weaker for this architecture. Use LLaMA-2-7B-Chat for paper-comparable results.

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

### Install evaluation dependencies

```bash
pip install lm-eval accelerate rouge-score sacrebleu
```

### Clone TRACE repository (for evaluation scripts)

```bash
git clone https://github.com/BeyonderXX/TRACE.git /opt/TRACE
cd /opt/TRACE && pip install -r requirements.txt
```

---

## Step 2: Download and Prepare Data

### 2a. Download TRACE datasets

The preprocessed TRACE datasets (8 tasks, each with `train.json`, `eval.json`, `test.json`)
are available from the TRACE GitHub repository's Google Drive link:

```
https://github.com/BeyonderXX/TRACE  →  README  →  Google Drive link
```

Download the `LLM-CL-Benchmark_5000` variant (5,000 examples/task, used in the paper).
After extraction the directory structure should be:

```
/data/TRACE/
  C-STANCE/train.json  eval.json  test.json
  FOMC/train.json      ...
  ...
  20Minuten/train.json ...
```

TRACE JSON format per record:
```json
{"prompt": "...", "answer": "..."}
```

### 2b. Tokenize TRACE data for mini_trainer

```bash
# For LLaMA-2 (paper model):
python scripts/convert_trace_data.py \
    --model meta-llama/Llama-2-7b-chat-hf \
    --trace-dir /data/TRACE \
    --output-dir /data/TRACE_tokenized_llama2

# For Qwen2.5-3B:
python scripts/convert_trace_data.py \
    --model Qwen/Qwen2.5-3B-Instruct \
    --trace-dir /data/TRACE \
    --output-dir /data/TRACE_tokenized_qwen25
```

---

## Step 3: MMLU Baseline Evaluations

Run once before any training to establish anchors for all subsequent comparisons.

### 3a. Original model MMLU

```bash
# LLaMA-2:
lm_eval --model hf \
    --model_args pretrained=meta-llama/Llama-2-7b-chat-hf,dtype=bfloat16 \
    --tasks mmlu --num_fewshot 5 --batch_size auto \
    --output_path results/mmlu/original

# Qwen2.5:
lm_eval --model hf \
    --model_args pretrained=Qwen/Qwen2.5-3B-Instruct,dtype=bfloat16 \
    --tasks mmlu --num_fewshot 5 --batch_size auto \
    --output_path results/mmlu/original
```

### 3b. SVD-truncated baseline

Creates a version of the model with the bottom 25% of singular values of every OSFT-targeted
weight matrix zeroed out — the "reserved but untrained" control:

```bash
# LLaMA-2:
python scripts/make_svd_truncated_model.py \
    --model-path meta-llama/Llama-2-7b-chat-hf \
    --output-path /checkpoints/svd_truncated \
    --unfreeze-rank-ratio 0.25

# Qwen2.5:
python scripts/make_svd_truncated_model.py \
    --model-path Qwen/Qwen2.5-3B-Instruct \
    --output-path /checkpoints/svd_truncated \
    --unfreeze-rank-ratio 0.25
```

Then evaluate MMLU on the truncated model the same way as 3a, pointing to `/checkpoints/svd_truncated`.

> **Qwen2.5 note:** expect a ~42 pp MMLU drop for the SVD-truncated Qwen2.5-3B model (vs
> ~1–2 pp for LLaMA-2-7B). This indicates that the low-singular-value subspace encodes
> significant general knowledge in Qwen2.5, which limits how much OSFT can protect it.

---

## Step 4: Sequential OSFT Training

Use the all-in-one orchestration script which handles tokenization, baselines, training, and
per-task MMLU automatically with full resume support:

```bash
# LLaMA-2 (paper configuration):
export HF_TOKEN=<your_token>   # required for gated model
bash scripts/run_trace_llama3.sh   # edit MODEL= line to set meta-llama/Llama-2-7b-chat-hf

# Qwen2.5-3B (ungated, no token needed):
bash scripts/run_trace_llama3.sh   # MODEL defaults to Qwen/Qwen2.5-3B-Instruct
```

Or invoke the training loop directly for more control:

```bash
bash scripts/train_trace_osft.sh \
    --model meta-llama/Llama-2-7b-chat-hf \   # or Qwen/Qwen2.5-3B-Instruct
    --data-root /data/TRACE_tokenized_llama2 \ # match tokenizer above
    --ckpt-root /checkpoints/trace_osft \
    --results-dir results \
    --n-gpus 1 \
    --max-tokens-per-gpu 8192 \
    --batch-size 128 \
    --unfreeze-rank-ratio 0.25 \
    --max-epochs 3 \
    --no-liger-kernels \   # omit if liger-kernels is installed
    --skip-trace-eval      # omit if /opt/TRACE is cloned
```

### Training configuration (from paper)

| Parameter | Value |
|-----------|-------|
| Model | `meta-llama/Llama-2-7b-chat-hf` |
| `osft_unfreeze_rank_ratio` | 0.25 |
| Learning rate | 2e-4 |
| Batch size | 128 |
| Max tokens/GPU | 8,192 |
| Epochs per task | 3 |
| LR scheduler | cosine |
| Warmup steps | 100 |
| Optimizer | AdamW (β1=0.9, β2=0.95) |
| Training dtype | bfloat16 |

> Consult Appendix A.6 of arXiv:2504.07097 for the authoritative hyperparameter values.

### Disk management

Each 7B checkpoint is ~14 GB; each 3B checkpoint is ~6.4 GB. The training script automatically
deletes the checkpoint two tasks back before writing a new one, so only two checkpoints ever
coexist on disk. On a 50 GB root filesystem, 3B models fit comfortably; 7B models require
either a larger filesystem or a dedicated checkpoint path on external storage.

---

## Step 5: Collect Results

```bash
python scripts/collect_mmlu.py --results-dir results
```

Example output (LLaMA-2-7B, expected):

```
Checkpoint                      MMLU acc  Δ vs original  Δ vs SVD-trunc
-----------------------------------------------------------------------
original                          63.45%         +0.00%          +1.35%
svd_truncated                     62.10%         -1.35%          +0.00%
osft_task_1 (C-STANCE)            63.20%         -0.25%          +1.10%
osft_task_2 (FOMC)                63.10%         -0.35%          +1.00%
...
osft_task_8 (20Minuten)           62.90%         -0.55%          +0.80%
```

---

## Step 6: TRACE Metrics

```bash
python scripts/compute_metrics.py --results-dir results
```

Target numbers from the paper:

| Metric | OSFT (paper) |
|--------|-------------|
| Avg Accuracy | 48.4% |
| Backward Transfer | +7.1% |

---

## Step 7: Troubleshooting

| Issue | Resolution |
|-------|-----------|
| OOM during SVD init | Reduce `--max-tokens-per-gpu` |
| `No space left on device` | Only one checkpoint at a time; ensure `--checkpoint-at-epoch` is not set |
| MMLU result not found by `collect_mmlu.py` | lm_eval ≥0.4 writes into a model-named subdir; script handles this automatically |
| SVD-truncated MMLU drops >10 pp | Expected for Qwen2.5; use LLaMA-2 for paper-comparable results |
| `Parameter not in torch.bfloat16` | Ensure `TESTING=true` is not overriding dtype; check `osft_utils.py` meta-device fix |
| `Cannot find transformer block container` | Check `setup_model_for_training.py` for your model's layer path |

---

## References

- OSFT Paper: [arXiv:2504.07097](https://arxiv.org/abs/2504.07097)
- TRACE Paper: [arXiv:2310.06762](https://arxiv.org/abs/2310.06762)
- TRACE GitHub: [BeyonderXX/TRACE](https://github.com/BeyonderXX/TRACE)
- mini_trainer: [Red-Hat-AI-Innovation-Team/mini_trainer](https://github.com/Red-Hat-AI-Innovation-Team/mini_trainer)
