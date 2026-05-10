"""
Build a tiny preference dataset for DPO/ORPO smoke testing.

For real work, replace the synthetic pairs with:
- Pairs you generated from two model variants and rated yourself.
- Pairs labelled by a stronger LLM-as-judge (acceptable for training, not
  for shipping if the judge is cloud and you're claiming on-device privacy).
"""
from __future__ import annotations
import argparse
import json
import random


SEEDS = [
    ("Explain unified memory in two sentences.",
     "On Apple Silicon, the CPU and GPU share one DRAM pool, eliminating cross-device copies. "
     "This raises bandwidth ceilings for KV-cache-heavy workloads.",
     "Unified memory is when the memory is unified across the CPU and GPU and they share it together unifiedly."),
    ("Suggest a 4-bit quantization config for a 7B on a 32 GB Mac.",
     "Use 4-bit weights with group size 64 and 4-bit KV cache; keep the OS headroom around 8 GB.",
     "Just quantize it bro, 4-bit is fine, don't overthink it."),
    ("How do you enable speculative decoding in mlx-lm?",
     "Pass --speculative quantspec for self-speculative decoding, or specify --draft-model for EAGLE-3.",
     "There is a flag somewhere I think, look it up."),
    ("Why does MoE decode faster than dense at matched 4-bit on Mac?",
     "Decode is bandwidth-bound; MoE only streams the active experts per token, so per-token bytes drop.",
     "Because Apple optimizes them better in MLX or something."),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="prefs.jsonl")
    ap.add_argument("--repeats", type=int, default=20)
    args = ap.parse_args()

    random.seed(0)
    rows = []
    for _ in range(args.repeats):
        for prompt, chosen, rejected in SEEDS:
            rows.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
    random.shuffle(rows)

    with open(args.output, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(rows)} preference pairs to {args.output}")


if __name__ == "__main__":
    main()
