"""
05 - llama.cpp local benchmark wrapper.

Drives llama-server (OpenAI-compatible) with a small batch-1 / batch-N
workload and prints TTFT, ITL, throughput. Use the same shape as
serve_and_hit.py from Topic 01 so the bake-off (Topic 07) can compare
all engines through one client harness.

Prereqs:
    brew install llama.cpp                    # Mac
    or build from source: https://github.com/ggml-org/llama.cpp

    # download a GGUF
    huggingface-cli download bartowski/Qwen2.5-7B-Instruct-GGUF \
        Qwen2.5-7B-Instruct-Q4_K_M.gguf --local-dir ./models

    # serve
    llama-server -m ./models/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
        --host 0.0.0.0 --port 8003 --ctx-size 8192 -ngl 999 \
        --parallel 8

Then:
    pip install openai
    python bench_local.py --base-url http://localhost:8003/v1 --concurrency 1
    python bench_local.py --base-url http://localhost:8003/v1 --concurrency 8
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time

from openai import AsyncOpenAI


PROMPTS = [
    "Explain paged KV cache to a database engineer.",
    "Why is GGUF a good portability choice?",
    "When does CPU inference beat GPU on $/Mtok?",
    "Sketch the difference between K-quants and i-quants.",
    "What did Unsloth Dynamic v2.0 change?",
    "Where does FP4 fit on Apple Silicon?",
    "Why is Metal a serious backend in 2026?",
    "When should I not use llama.cpp?",
]


async def one(client: AsyncOpenAI, model: str, prompt: str, max_tokens: int) -> tuple[float, int, float]:
    t0 = time.perf_counter()
    ttft: float | None = None
    n = 0
    stream = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        stream=True,
        temperature=0.7,
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


async def run(base_url: str, model: str, n: int, concurrency: int, max_tokens: int) -> None:
    client = AsyncOpenAI(base_url=base_url, api_key="EMPTY")
    sem = asyncio.Semaphore(concurrency)

    async def bounded(p: str) -> tuple[float, int, float]:
        async with sem:
            return await one(client, model, p, max_tokens)

    prompts = [PROMPTS[i % len(PROMPTS)] for i in range(n)]
    t0 = time.perf_counter()
    out = await asyncio.gather(*[bounded(p) for p in prompts])
    wall = time.perf_counter() - t0

    ttfts = [r[0] for r in out]
    toks = sum(r[1] for r in out)

    def pct(xs: list[float], p: float) -> float:
        return sorted(xs)[max(0, min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1)))))]

    print(f"\nllama.cpp via {base_url}")
    print(f"  n={n}  concurrency={concurrency}  max_tokens={max_tokens}")
    print(f"  wall              {wall:.2f}s")
    print(f"  agg throughput    {toks / wall:.0f} tok/s")
    print(f"  TTFT p50/p95/p99  {pct(ttfts, 50)*1000:.0f} / {pct(ttfts, 95)*1000:.0f} / {pct(ttfts, 99)*1000:.0f}  ms")
    print(f"  TTFT mean         {statistics.mean(ttfts)*1000:.0f}  ms")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8003/v1")
    # llama-server accepts any model name; the file loaded at startup is what's served.
    ap.add_argument("--model", default="local-gguf")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=128)
    args = ap.parse_args()
    asyncio.run(run(args.base_url, args.model, args.n, args.concurrency, args.max_tokens))


if __name__ == "__main__":
    main()
