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


def convert_task(task_name: str, trace_dir: Path, output_dir: Path, tokenizer) -> None:
    task_in = trace_dir / task_name
    task_out = output_dir / task_name
    task_out.mkdir(parents=True, exist_ok=True)

    for split in ("train", "eval", "test"):
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
        print(f"  {task_name}/{split}: {len(records)} examples → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--trace-dir", default="/data/TRACE")
    parser.add_argument("--output-dir", default="/data/TRACE_tokenized")
    args = parser.parse_args()

    trace_dir = Path(args.trace_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token

    for task in TRACE_TASKS:
        print(f"Converting {task} ...")
        convert_task(task, trace_dir, output_dir, tokenizer)

    print(f"\nAll tasks tokenized → {output_dir}")


if __name__ == "__main__":
    main()
