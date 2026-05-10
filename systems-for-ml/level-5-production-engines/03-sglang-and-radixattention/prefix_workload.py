"""
03 - SGLang vs vLLM on a prefix-heavy workload.

Runs N "chatbot-style" requests that all share a 4KB system prompt but vary
in user-turn content, then measures TTFT and throughput against an OpenAI-
compatible endpoint. Point it at SGLang on one port and vLLM on another;
the delta on TTFT is RadixAttention's edge.

Prereqs:
    pip install sglang[all] openai

Server side, two terminals:
    python -m sglang.launch_server --model-path Qwen/Qwen2.5-7B-Instruct --port 8001
    vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000

Client:
    python prefix_workload.py --base-url http://localhost:8001/v1 --label sglang
    python prefix_workload.py --base-url http://localhost:8000/v1 --label vllm
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time

from openai import AsyncOpenAI


# ~4 KB of "system prompt" — what a real assistant config looks like.
SYSTEM_PROMPT = (
    "You are a careful, friendly assistant for a fictional support team. "
    "You answer concisely. Always cite sources when you can. "
    "Refuse to provide medical, legal, or financial advice and instead "
    "redirect the user to a qualified professional. " * 12
)

USER_TURNS = [
    "What's the difference between paged and contiguous KV cache?",
    "Why does TTFT matter more than throughput for chat?",
    "Explain RadixAttention in two sentences.",
    "When would I pick TensorRT-LLM over vLLM?",
    "How does chunked prefill interact with continuous batching?",
    "Give me three reasons SGLang might lose to vLLM.",
    "What is FlashInfer and why isn't it a separate engine?",
    "When does spec decode hurt instead of help?",
    "Explain prefix caching to a Postgres engineer.",
    "Why is the GIL still relevant in 2026 inference servers?",
]


async def one(client: AsyncOpenAI, model: str, user_turn: str, max_tokens: int) -> tuple[float, int, float]:
    t0 = time.perf_counter()
    ttft = None
    n_tokens = 0
    stream = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_turn},
        ],
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
        n_tokens += 1
    total = time.perf_counter() - t0
    return ttft or total, n_tokens, total


async def run(base_url: str, model: str, n: int, concurrency: int, max_tokens: int, label: str) -> None:
    client = AsyncOpenAI(base_url=base_url, api_key="EMPTY")
    sem = asyncio.Semaphore(concurrency)

    async def bounded(turn: str) -> tuple[float, int, float]:
        async with sem:
            return await one(client, model, turn, max_tokens)

    turns = [USER_TURNS[i % len(USER_TURNS)] for i in range(n)]
    t0 = time.perf_counter()
    out = await asyncio.gather(*[bounded(t) for t in turns])
    wall = time.perf_counter() - t0

    ttfts = [r[0] for r in out]
    toks = sum(r[1] for r in out)

    def pct(xs: list[float], p: float) -> float:
        return sorted(xs)[max(0, min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1)))))]

    print(f"\n[{label}] base_url={base_url}")
    print(f"  requests           {n}  concurrency {concurrency}")
    print(f"  wall               {wall:.2f}s")
    print(f"  agg throughput     {toks / wall:.0f} tok/s")
    print(f"  TTFT  p50/p95/p99  {pct(ttfts, 50)*1000:.0f} / {pct(ttfts, 95)*1000:.0f} / {pct(ttfts, 99)*1000:.0f}  ms")
    print(f"  TTFT  mean/stdev   {statistics.mean(ttfts)*1000:.0f}  /  {statistics.stdev(ttfts)*1000 if len(ttfts) > 1 else 0:.0f}  ms")
    print("  (Same prompt prefix across all requests — prefix cache should keep TTFT low after the first.)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--label", default="engine")
    args = ap.parse_args()
    asyncio.run(run(args.base_url, args.model, args.n, args.concurrency, args.max_tokens, args.label))


if __name__ == "__main__":
    main()
