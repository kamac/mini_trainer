"""Create an SVD-truncated version of a model.

Zeros out the low singular-value components of every targeted weight matrix —
exactly the subspace that OSFT trains in. This produces the "erased subspace,
nothing learned" baseline for MMLU comparison.

For LLaMA-2, OSFT targets (per transformer layer):
  self_attn.{q,k,v,o}_proj  and  mlp.{gate,up,down}_proj  (osft_utils.py:200-207)

With --unfreeze-rank-ratio 0.25, the bottom 25% of singular values are zeroed
and the top 75% are kept unchanged.

Usage:
    python scripts/make_svd_truncated_model.py \
        --model-path meta-llama/Llama-2-7b-chat-hf \
        --output-path /checkpoints/trace_osft/svd_truncated_baseline \
        --unfreeze-rank-ratio 0.25
"""

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Layer suffixes targeted by OSFT for LLaMA-style models (osft_utils.py lines 200-207)
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
    print(f"Loading model from {model_path} ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="cpu"
    )

    n_truncated = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if not any(name.endswith(f"{s}.weight") for s in LLAMA_TARGET_SUFFIXES):
                continue
            if param.dim() != 2:
                continue

            W = param.data.float()  # SVD is numerically sensitive in bfloat16
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)

            keep_k = max(1, int(round(S.shape[0] * (1.0 - unfreeze_rank_ratio))))
            S[keep_k:] = 0.0

            param.data = (U @ torch.diag(S) @ Vh).to(param.dtype)
            n_truncated += 1
            print(f"  {name}: kept {keep_k}/{S.shape[0]} singular values")

    print(f"\nTruncated {n_truncated} weight matrices.")
    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path)

    tok = AutoTokenizer.from_pretrained(model_path)
    tok.save_pretrained(output_path)
    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--unfreeze-rank-ratio", type=float, default=0.25)
    args = parser.parse_args()
    truncate_model(args.model_path, args.output_path, args.unfreeze_rank_ratio)


if __name__ == "__main__":
    main()
