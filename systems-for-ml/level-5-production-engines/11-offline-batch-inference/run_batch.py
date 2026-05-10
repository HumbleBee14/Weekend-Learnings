"""
11 - vLLM offline batch inference: run N prompts as fast as possible.

Compare wall time to running the same N prompts through the OpenAI server
one-at-a-time (or even with concurrency 8). The offline mode should
crush the online numbers because the engine has full visibility into the
workload and can max-batch it.

Prereqs:
    pip install vllm

Run:
    python run_batch.py --n 1000 --max-tokens 128
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

try:
    from vllm import LLM, SamplingParams  # type: ignore
except ImportError:
    print("Run this on a CUDA box: pip install vllm")
    raise SystemExit(0)


SHARED_INSTRUCTION = (
    "You are a careful classifier. Read the document and produce a JSON object "
    "with fields {category, sentiment, key_phrases}. Return only the JSON. "
)

DOCS = [
    "Quarterly earnings beat analyst expectations on cloud growth.",
    "The bridge tournament concluded with a surprise winner from a junior team.",
    "A new study links gut microbiome diversity to mood regulation.",
    "Critics panned the sequel for failing to recapture the original's tone.",
    "Wildfires in the foothills prompted a regional air quality advisory.",
    "Two startups merged to build a neuromorphic edge inference chip.",
    "The municipal council approved a tax credit for residential heat pumps.",
    "An archaeology team uncovered a 4th-century mosaic beneath a parking lot.",
]


def gen_prompts(n: int) -> list[str]:
    return [
        f"{SHARED_INSTRUCTION}\n\nDocument: {DOCS[i % len(DOCS)]} (#{i})"
        for i in range(n)
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--out", default="results.jsonl")
    args = ap.parse_args()

    print(f"Loading {args.model} (offline mode)...")
    t0 = time.perf_counter()
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=0.92,
        max_num_seqs=512,
        max_num_batched_tokens=8192,
        enable_prefix_caching=True,
    )
    print(f"  ready in {time.perf_counter() - t0:.1f}s")

    prompts = gen_prompts(args.n)
    params = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)

    t1 = time.perf_counter()
    outputs = llm.generate(prompts, params)
    wall = time.perf_counter() - t1

    out_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    print(f"\n{args.n} prompts, {out_tokens} output tokens in {wall:.2f}s")
    print(f"  agg throughput   {out_tokens / wall:.0f} tok/s output")
    print(f"  per-prompt avg   {wall / args.n * 1000:.1f} ms")

    Path(args.out).write_text(
        "\n".join(
            json.dumps({"prompt": p, "output": o.outputs[0].text})
            for p, o in zip(prompts, outputs)
        )
    )
    print(f"  results -> {args.out}")
    print(
        "\nThe shared instruction across all prompts means prefix caching is "
        "doing real work. Disable it (enable_prefix_caching=False) and re-run "
        "to see the gap."
    )


if __name__ == "__main__":
    main()
