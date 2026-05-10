"""
12 - Speculative decoding measurement against vLLM.

Drives a vLLM server twice with the same workload — once with spec decode
on, once off — and reports throughput delta, TTFT delta, and the server's
self-reported acceptance rate (read from /metrics).

Prereqs:
    Server with spec decode off:
        vllm serve <model> --port 8010

    Server with n-gram spec decode on:
        vllm serve <model> --port 8011 \
            --speculative-config '{"method":"ngram","prompt_lookup_max":4,"num_speculative_tokens":4}'

Run:
    pip install openai httpx
    python measure_spec.py --baseline http://localhost:8010 --spec http://localhost:8011

Workload: a mix of chat, code, and reasoning prompts so you can see how
acceptance rate varies across them. Spec wins on chat/code, loses or
breaks even on hard reasoning.
"""

from __future__ import annotations

import argparse
import asyncio
import re
import statistics
import time

import httpx
from openai import AsyncOpenAI


PROMPTS = {
    "chat": [
        "Hi, what's a good way to remember someone's name when you meet them?",
        "Tell me a quick story about a robot that learned to garden.",
        "What's a polite way to ask a coworker to lower their voice?",
    ],
    "code": [
        "Write a Python function to flatten a nested list using recursion.",
        "Convert this for-loop into a list comprehension: result = []\nfor x in xs:\n  if x>0:\n    result.append(x*2)",
        "Implement a sliding-window max for a list of ints, window size k.",
    ],
    "reasoning": [
        "If a train leaves A at 9am at 60mph and another leaves B at 10am at 75mph, "
        "and A and B are 300 miles apart, when do they meet? Show your work.",
        "Five hats: 2 red, 3 blue. Three people each get one hat, can see the others' "
        "but not their own. The first two say 'I don't know my hat color'. What does the third know?",
    ],
}


async def hit(client: AsyncOpenAI, model: str, prompt: str, max_tokens: int) -> tuple[float, int, float]:
    t0 = time.perf_counter()
    ttft: float | None = None
    n = 0
    stream = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        stream=True,
        temperature=0.0,
    )
    async for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if not delta or not delta.content:
            continue
        if ttft is None:
            ttft = time.perf_counter() - t0
        n += 1
    return ttft or 0.0, n, time.perf_counter() - t0


async def run_one(server: str, model: str, max_tokens: int) -> dict[str, dict]:
    client = AsyncOpenAI(base_url=f"{server}/v1", api_key="EMPTY")
    out: dict[str, dict] = {}
    for category, prompts in PROMPTS.items():
        t0 = time.perf_counter()
        results = await asyncio.gather(*[hit(client, model, p, max_tokens) for p in prompts])
        wall = time.perf_counter() - t0
        toks = sum(r[1] for r in results)
        ttfts = [r[0] * 1000 for r in results]
        out[category] = {
            "wall_s": wall,
            "throughput_tok_s": toks / wall if wall else 0.0,
            "ttft_mean_ms": statistics.mean(ttfts) if ttfts else 0.0,
            "n_tokens": toks,
        }
    return out


def parse_acceptance(metrics_text: str) -> float | None:
    """Read accepted/draft from vLLM's /metrics, return acceptance rate."""
    accepted = re.search(r"^vllm:spec_decode_num_accepted_tokens_total\s+(\d+\.?\d*)", metrics_text, re.M)
    draft = re.search(r"^vllm:spec_decode_num_draft_tokens_total\s+(\d+\.?\d*)", metrics_text, re.M)
    if not accepted or not draft:
        return None
    a = float(accepted.group(1))
    d = float(draft.group(1))
    return a / d if d else None


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="http://localhost:8010")
    ap.add_argument("--spec", default="http://localhost:8011")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--max-tokens", type=int, default=256)
    args = ap.parse_args()

    print(f"[baseline] {args.baseline}")
    base = await run_one(args.baseline, args.model, args.max_tokens)
    print(f"[spec    ] {args.spec}")
    spec = await run_one(args.spec, args.model, args.max_tokens)

    async with httpx.AsyncClient(timeout=10) as hc:
        try:
            r = await hc.get(f"{args.spec}/metrics")
            acc = parse_acceptance(r.text)
        except Exception:
            acc = None

    print(f"\n{'category':12s} {'baseline tok/s':>15s}  {'spec tok/s':>12s}  {'speedup':>8s}  {'TTFT base ms':>14s}  {'TTFT spec ms':>14s}")
    for k in PROMPTS:
        b = base[k]
        s = spec[k]
        speedup = s["throughput_tok_s"] / b["throughput_tok_s"] if b["throughput_tok_s"] else 0
        print(
            f"{k:12s} {b['throughput_tok_s']:15.0f}  {s['throughput_tok_s']:12.0f}  "
            f"{speedup:8.2f}x  {b['ttft_mean_ms']:14.0f}  {s['ttft_mean_ms']:14.0f}"
        )

    if acc is not None:
        print(f"\nServer-reported acceptance rate (cumulative): {acc:.2%}")
    print("\nExpected shape: speedup is high on chat/code, modest or <1x on reasoning.")
    print("If speedup<1x everywhere, spec is hurting — wrong draft for this workload.")


if __name__ == "__main__":
    asyncio.run(main())
