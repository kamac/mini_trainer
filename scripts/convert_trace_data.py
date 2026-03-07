"""Convert TRACE JSON data to mini_trainer tokenized JSONL format.

Usage:
    python scripts/convert_trace_data.py \
        --model meta-llama/Llama-2-7b-chat-hf \
        --trace-dir /data/TRACE \
        --output-dir /data/TRACE_tokenized
"""

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

TRACE_TASKS = [
    "C-STANCE", "FOMC", "MeetingBank", "Py150",
    "ScienceQA", "NumGLUE-cm", "NumGLUE-ds", "20Minuten",
]
IGNORE_INDEX = -100


def convert_task(
    task_name: str,
    trace_dir: Path,
    output_dir: Path,
    tokenizer,
    max_seq_length: int,
) -> None:
    task_in = trace_dir / task_name
    task_out = output_dir / task_name
    task_out.mkdir(parents=True, exist_ok=True)

    for split in ("train", "eval", "test"):
        src = task_in / f"{split}.json"
        if not src.exists():
            continue
        records = json.loads(src.read_text())
        out_path = task_out / f"{split}.jsonl"
        n_written = 0
        n_truncated = 0
        with out_path.open("w") as f:
            for rec in records:
                prompt_ids = tokenizer(rec["prompt"], add_special_tokens=True).input_ids
                answer_ids = tokenizer(rec["answer"], add_special_tokens=False).input_ids
                # Ensure sequence ends with EOS
                if not answer_ids or answer_ids[-1] != tokenizer.eos_token_id:
                    answer_ids = answer_ids + [tokenizer.eos_token_id]

                full_ids = prompt_ids + answer_ids
                labels   = [IGNORE_INDEX] * len(prompt_ids) + answer_ids

                if len(full_ids) > max_seq_length:
                    full_ids = full_ids[:max_seq_length]
                    labels   = labels[:max_seq_length]
                    n_truncated += 1

                f.write(json.dumps({
                    "input_ids": full_ids,
                    "labels": labels,
                    "len": len(full_ids),
                }) + "\n")
                n_written += 1

        trunc_note = f", {n_truncated} truncated to {max_seq_length}" if n_truncated else ""
        print(f"  {task_name}/{split}: {n_written} examples → {out_path}{trunc_note}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--trace-dir", default="/data/TRACE")
    parser.add_argument("--output-dir", default="/data/TRACE_tokenized")
    parser.add_argument(
        "--max-seq-length", type=int, default=2048,
        help="Truncate sequences longer than this many tokens (default: 2048)",
    )
    args = parser.parse_args()

    trace_dir = Path(args.trace_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token

    for task in TRACE_TASKS:
        print(f"Converting {task} ...")
        convert_task(task, trace_dir, output_dir, tokenizer, args.max_seq_length)

    print(f"\nAll tasks tokenized → {output_dir}")


if __name__ == "__main__":
    main()
