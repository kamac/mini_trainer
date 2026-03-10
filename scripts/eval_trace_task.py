"""Evaluate a model checkpoint on a TRACE task.

Runs greedy inference on the test split and computes task-appropriate metrics.
Designed to replace the TRACE repo's DeepSpeed-based inference pipeline.

Metric per task:
  C-STANCE, FOMC, NumGLUE-cm, NumGLUE-ds : accuracy (exact first-char match)
  ScienceQA                               : accuracy + BLEU/ROUGE on reasoning
  MeetingBank, 20Minuten                  : BLEU-1, BLEU-4, ROUGE-L
  Py150                                   : fuzzy similarity (fuzzywuzzy)

Usage:
    python scripts/eval_trace_task.py \
        --model /tmp/checkpoints/.../hf_format/samples_15000.0 \
        --task C-STANCE \
        --test-file /tmp/TRACE/C-STANCE/test.json \
        --output-file results/trace/C-STANCE_after_task1.json
"""

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


# ── Metrics ───────────────────────────────────────────────────────────────────

def _first_char(s: str) -> str:
    s = s.strip()
    return s[0].upper() if s else ""


def accuracy(preds: list[str], gts: list[str]) -> float:
    correct = sum(_first_char(p) == _first_char(g) for p, g in zip(preds, gts))
    return correct / len(gts) if gts else 0.0


def bleu(preds: list[str], gts: list[str], n: int) -> float:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    smooth = SmoothingFunction().method1
    weights = tuple(1.0 / n for _ in range(n))
    scores = []
    for p, g in zip(preds, gts):
        p_tok = p.split()
        g_tok = g.split()
        if not p_tok or not g_tok:
            scores.append(0.0)
            continue
        scores.append(sentence_bleu([g_tok], p_tok, weights=weights,
                                    smoothing_function=smooth))
    return sum(scores) / len(scores) if scores else 0.0


def rouge_l(preds: list[str], gts: list[str]) -> float:
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    scores = [scorer.score(g, p)["rougeL"].fmeasure for p, g in zip(preds, gts)]
    return sum(scores) / len(scores) if scores else 0.0


def fuzz_sim(preds: list[str], gts: list[str]) -> float:
    from fuzzywuzzy import fuzz as fz
    scores = [fz.ratio(p, g) for p, g in zip(preds, gts)]
    return sum(scores) / len(scores) if scores else 0.0


def _postprocess_py150(code: str) -> str:
    code = code.replace("<NUM_LIT>", "0").replace("<STR_LIT>", "").replace("<CHAR_LIT>", "")
    pattern = re.compile(r"<(STR|NUM|CHAR)_LIT:(.*?)>", re.S)
    for kind, val in re.findall(pattern, code):
        code = code.replace(f"<{kind}_LIT:{val}>", val)
    return code


TASK_METRICS = {
    "C-STANCE":    lambda p, g: {"accuracy": accuracy(p, g)},
    "FOMC":        lambda p, g: {"accuracy": accuracy(p, g)},
    "NumGLUE-cm":  lambda p, g: {"accuracy": accuracy(p, g)},
    "NumGLUE-ds":  lambda p, g: {"accuracy": accuracy(p, g)},
    "MeetingBank": lambda p, g: {"bleu-1": bleu(p, g, 1), "bleu-4": bleu(p, g, 4),
                                  "rouge-L": rouge_l(p, g)},
    "20Minuten":   lambda p, g: {"bleu-1": bleu(p, g, 1), "bleu-4": bleu(p, g, 4),
                                  "rouge-L": rouge_l(p, g)},
    "Py150":       lambda p, g: {"similarity": fuzz_sim(
                                     [_postprocess_py150(x) for x in p],
                                     [_postprocess_py150(x) for x in g])},
    "ScienceQA":   lambda p, g: {
                                  "accuracy": accuracy(
                                      [x[0] if x else "" for x in p],
                                      [x[0] if x else "" for x in g]),
                                  "bleu-1": bleu([x[2:] for x in p], [x[2:] for x in g], 1),
                                  "rouge-L": rouge_l([x[2:] for x in p], [x[2:] for x in g]),
                                 },
}


# ── Inference ─────────────────────────────────────────────────────────────────

def generate_predictions(model_path: str, prompts: list[str],
                          batch_size: int = 4, max_new_tokens: int = 64) -> list[str]:
    print(f"Loading model from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    predictions = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Inferring"):
        batch = prompts[i: i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=1024).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        # Decode only the newly generated tokens
        for j, out in enumerate(outputs):
            new_tokens = out[inputs["input_ids"].shape[1]:]
            predictions.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())

    return predictions


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True,
                        help="Path to HF checkpoint directory")
    parser.add_argument("--task", required=True,
                        choices=list(TASK_METRICS.keys()),
                        help="TRACE task name")
    parser.add_argument("--test-file", required=True,
                        help="Path to test.json for this task")
    parser.add_argument("--output-file", required=True,
                        help="Where to write JSON results")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit number of test samples (for debugging)")
    args = parser.parse_args()

    # Load test data
    data = json.loads(Path(args.test_file).read_text())
    if args.max_samples:
        data = data[:args.max_samples]

    prompts = [d["prompt"] for d in data]
    ground_truths = [d["answer"] for d in data]

    print(f"Task: {args.task}  |  {len(prompts)} examples")

    predictions = generate_predictions(
        args.model, prompts, args.batch_size, args.max_new_tokens
    )

    # Compute metrics
    metric_fn = TASK_METRICS[args.task]
    results = metric_fn(predictions, ground_truths)

    print(f"\nResults for {args.task}:")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")

    # Save output
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    output = {
        "task": args.task,
        "model": args.model,
        "num_examples": len(prompts),
        "metrics": results,
        "predictions": predictions[:20],  # save first 20 for spot-checking
    }
    Path(args.output_file).write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\nSaved: {args.output_file}")


if __name__ == "__main__":
    main()
