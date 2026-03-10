# Teaching a Language Model Eight Things Without Forgetting Any of Them

*An experiment in continual learning with Orthogonal Subspace Fine-Tuning on Qwen2.5-3B-Instruct*

---

## The Problem: Catastrophic Forgetting

When you fine-tune a neural network on a new task, it tends to forget what it already knew. This is **catastrophic forgetting** — the weights that encoded previous knowledge get overwritten as the model adapts to new data. For large language models deployed in practice, this is a serious obstacle: you can't simply keep retraining the same model on new tasks without degrading everything it learned before.

The [TRACE benchmark](https://arxiv.org/abs/2310.05792) was designed to measure this problem. It presents 8 diverse tasks sequentially:

| # | Task | Type | Metric |
|---|------|------|--------|
| 1 | C-STANCE | Chinese stance detection | Accuracy |
| 2 | FOMC | Fed monetary policy classification | Accuracy |
| 3 | MeetingBank | Meeting summarisation | BLEU / ROUGE-L |
| 4 | Py150 | Python code completion | Fuzzy similarity |
| 5 | ScienceQA | Science multi-choice + reasoning | Accuracy + BLEU |
| 6 | NumGLUE-cm | Numerical reasoning (commonsense) | Accuracy |
| 7 | NumGLUE-ds | Numerical reasoning (diverse) | Accuracy |
| 8 | 20Minuten | German news summarisation | BLEU / ROUGE-L |

The goal: train on all 8, one at a time, while preserving performance on everything you've already learned.

---

## The Approach: OSFT

**Orthogonal Subspace Fine-Tuning (OSFT)** is a continual learning technique that tries to sidestep overwriting by restricting *where* new learning happens in weight space.

The key idea: every weight matrix has a singular value decomposition **W = UΣV^T**. The largest singular values correspond to the "important" directions — the ones most responsible for the model's general capabilities. OSFT reserves those directions and only allows updates in the subspace spanned by the **smallest** singular value components.

Formally, for a weight matrix **W ∈ ℝ^{m×n}** with SVD **W = UΣV^T**:
- Freeze the top **(1 − r)** fraction of singular components
- Allow gradient flow only through the bottom **r** fraction

The parameter `r` — the **unfreeze rank ratio** — controls the trade-off between plasticity (can the model learn?) and stability (does it retain what it knew?).

---

## Choosing the Right Ratio: Spectral Analysis

Before training, we ran an SVD on every weight matrix in Qwen2.5-3B-Instruct and measured the **spectral mass** at each candidate ratio: how much of the model's total Frobenius norm energy lives in the bottom-r subspace?

![Ratio selection chart](spectral/summary/ratio_selection.png)

The spectral mass is defined as:

$$\text{mass}(r) = \frac{\sum_{i > (1-r)p} \sigma_i^2}{\sum_i \sigma_i^2}$$

Results for the original model:

| Ratio | Spectral mass |
|-------|--------------|
| 0.05  | 0.67% |
| 0.10  | 1.69% |
| 0.15  | 2.90% |
| 0.20  | 4.38% |
| **0.25** | **6.08%** |
| 0.30  | 8.04% |

We initially ran with ratio=0.25 and observed catastrophic MMLU degradation. Switching to **ratio=0.10** (1.69% spectral mass) produced stable results — the model can still learn without disrupting the dominant knowledge-encoding directions.

> **A subtle point**: spectral mass measures singular *value magnitudes*, not the directions the model actually learned. OSFT primarily rotates singular *vectors* (U, V) rather than changing magnitudes (Σ). This is why the spectral summary barely changes across tasks (see table below) — the metric is a proxy for "how much capacity are we reserving", not a direct readout of knowledge encoded per task.

Spectral mass across all 8 tasks (barely moves — as expected):

| Checkpoint | mass@0.10 | mass@0.20 | mass@0.30 |
|-----------|-----------|-----------|-----------|
| original | 1.69% | 4.38% | 8.04% |
| after task 1 (C-STANCE) | 1.70% | 4.39% | 8.05% |
| after task 4 (Py150) | 1.72% | 4.43% | 8.09% |
| after task 8 (20Minuten) | 1.75% | 4.48% | 8.14% |

---

## The SVD-Truncated Baseline

Before any fine-tuning, we created an **SVD-truncated** baseline: a version of Qwen2.5-3B where the bottom 10% of each weight matrix's singular components are zeroed out — no training, just ablation.

MMLU results tell the story:

| Model | MMLU (5-shot) |
|-------|--------------|
| Original Qwen2.5-3B | **66.5%** |
| SVD-truncated (bottom 10% zeroed) | **24.0%** |

The truncated model drops catastrophically — **−42.5pp**. This is surprising given those components only hold 1.69% of spectral mass. It tells us the low-rank subspace, while small in energy, is not noise: it carries real information the model relies on.

This result is consistent with [Staats, Thamm & Rosenow (2024)](https://arxiv.org/abs/2410.17770), who apply random matrix theory to pretrained transformers (BERT, Pythia, Llama) and find that *small singular values matter — but mainly once the model has been fine-tuned*. In their analysis, singular vectors corresponding to outlier (non-noise) values substantially overlap with eigenvectors of the activation covariance matrix — i.e., the directions the model actually activates during inference. Zeroing those components, as our SVD-truncated baseline does, removes exactly the structure the model depends on.

OSFT does not zero these components. It trains *within* that subspace, which is why the model can learn without destroying its general capabilities.

---

## General Knowledge: The MMLU Trajectory

We tracked MMLU accuracy (5-shot, 57 subtasks) after every OSFT task:

![MMLU trajectory](visualizations/mmlu_trajectory.png)

A few things stand out:

**FOMC causes a significant trough.** After training on FOMC (Fed policy classification — labels are single letters: A/B/C), MMLU drops from 58.6% to 49.2% (−9.4pp). FOMC's narrow label distribution appears to collapse the model's output diversity — a phenomenon related to what [Nait Saada, Naderi & Tanner (2024)](https://arxiv.org/abs/2410.07799) call **rank collapse**: when the effective rank of a layer's representations shrinks, tokens converge toward identical outputs and the model loses discriminative capacity. In our case, training on single-letter labels (A/B/C) pushes the model into a low-rank output regime that temporarily degrades multi-class reasoning across MMLU.

**The model recovers.** By task 5 (ScienceQA), MMLU has climbed back to 60.3%. By task 8, it reaches 60.2% — still −6.3pp from the original 66.5%, but the recovery after FOMC is clear.

The trajectory suggests the early tasks (C-STANCE, FOMC) have an outsized impact on general capabilities, while later tasks in more diverse domains (ScienceQA, NumGLUE) help restore it.

---

## Task Performance: Baseline vs. OSFT

How much did the model actually learn from each task?

![Baseline vs final performance](visualizations/baseline_vs_final.png)

The gains are substantial — particularly on the tasks where the base model was near-zero:

| Task | Baseline | After OSFT | Δ |
|------|----------|-----------|---|
| C-STANCE (acc) | 54.7% | 52.1% | −2.6pp |
| FOMC (acc) | 55.4% | 67.7% | +12.3pp |
| MeetingBank (ROUGE-L) | 19.4% | 33.9% | +14.5pp |
| Py150 (similarity) | 21.4 | 55.5 | +34.1 |
| ScienceQA (acc) | 80.0% | 90.9% | +10.9pp |
| NumGLUE-cm (acc) | 4.9% | 67.9% | +63.0pp |
| NumGLUE-ds (acc) | 1.2% | 68.6% | +67.4pp |
| 20Minuten (ROUGE-L) | 15.7% | 18.3% | +2.6pp |

The NumGLUE tasks are the most dramatic: the base model was essentially guessing (1–5%), and OSFT fine-tuning brought both to ~68%. ScienceQA improved from a strong 80% baseline to 91%.

C-STANCE is the only regression (−2.6pp), but notably the model is still competitive — and as we'll see, it largely holds that through subsequent training.

---

## Catastrophic Forgetting: The Full Picture

The key question for continual learning: after training on later tasks, how much does the model forget earlier ones?

![Forgetting heatmap](visualizations/forgetting_heatmap.png)

Each row is a task, each column is a snapshot in time (right after training on that task). The diagonal shows performance *when the task was just trained*. Off-diagonal values show retention.

**Reading the heatmap:**
- Warm colours = high performance (relative to that task's peak)
- Cool colours = degraded performance

**Key observations:**

1. **C-STANCE is remarkably stable.** It scores 52.1% right after training (task 1), and holds at 53.1% all the way through task 8. Training seven more tasks on top of it caused essentially zero forgetting. This is OSFT working as intended.

2. **FOMC actually improves over time.** After task 2 training: 67.7%. After task 8: 70.4%. The orthogonal subspace updates from later tasks seem to *refine* FOMC performance rather than overwrite it.

3. **MeetingBank and NumGLUE are stable once learned.** Their scores show little degradation in subsequent tasks.

4. **20Minuten (task 8) is the last task** — we don't see post-training forgetting for it.

---

## C-STANCE Retention in Detail

![C-STANCE retention](visualizations/cstance_retention.png)

C-STANCE is the harshest test of OSFT's anti-forgetting guarantee: it's trained first, then seven more tasks are learned on top. The score stays within ±2pp of its initial value throughout.

This is the core OSFT result: **knowledge in the high-rank subspace is left untouched**. Because subsequent tasks only update the low-rank subspace, and C-STANCE's representation is encoded primarily in the high-rank directions, it survives.

---

## What OSFT Can and Cannot Do

Based on this experiment:

**OSFT works well for:**
- Tasks that encode naturally in the low-rank subspace (NumGLUE, Py150, ScienceQA)
- Preserving previously-learned tasks (C-STANCE retention is near-perfect)
- Scaling across diverse task types without catastrophic forgetting

**OSFT struggles with:**
- Tasks with very narrow label distributions (FOMC's single-letter labels cause temporary general capability degradation)
- Measuring learning progress via spectral metrics — the singular values barely move even though the model is learning. Better diagnostics would track subspace *direction* changes, not just magnitudes.

**Open questions:**
- The SVD-truncated baseline dropping 42pp despite only 1.69% spectral mass suggests the low-rank subspace encodes more than its energy suggests. What is it encoding?
- At ratio=0.10, MMLU never fully recovers to the original 66.5%. Is this fundamental to OSFT, or a training hyperparameter issue?
- The FOMC trough recovery suggests some form of "consolidation" happens during later tasks. Is this an OSFT property or a general fine-tuning property?

---

## Setup

- **Model**: Qwen/Qwen2.5-3B-Instruct
- **OSFT ratio**: 0.10 (bottom 10% of each weight matrix's singular components are trainable)
- **Training**: 3 epochs per task, cosine LR schedule, 2e-4 peak LR, batch size 128
- **Evaluation**: MMLU 5-shot (lm-evaluation-harness), custom TRACE task evaluation
- **Hardware**: Single GPU

---

*Experiment code: [mini_trainer](https://github.com/maciejkozik3/mini-trainer)*

---

## References

- Baijiong Lin et al. (2023). **TRACE: A Comprehensive Benchmark for Continual Learning in Large Language Models.** arXiv:2310.05792.
- Max Staats, Matthias Thamm & Bernd Rosenow (2024). **Small Singular Values Matter: A Random Matrix Analysis of Transformer Models.** arXiv:2410.17770.
- Thiziri Nait Saada, Alireza Naderi & Jared Tanner (2024). **Mind the Gap: a Spectral Analysis of Rank Collapse and Signal Propagation in Attention Layers.** arXiv:2410.07799.
