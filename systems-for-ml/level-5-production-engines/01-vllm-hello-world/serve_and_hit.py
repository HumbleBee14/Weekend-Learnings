"""
01 - vLLM hello world: hit a running vLLM server and capture TTFT, ITL, throughput.

Prereqs (on a Linux + NVIDIA box):
    pip install vllm openai
    vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000 --max-model-len 8192

Then run this script from the same machine (or set BASE_URL).

What this measures:
- TTFT (time to first token)
- ITL (median inter-token latency)
- Throughput (output tokens / sec) per request and aggregate
- p50 / p95 / p99 across N concurrent requests

This script is intentionally minimal so you can compare it against the same
shape of harness pointed at SGLang (Topic 03), TRT-LLM (Topic 04), and
llama-server (Topic 05). Project 2 generalizes this into runner.py.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from dataclasses import dataclass, field

from openai import AsyncOpenAI


@dataclass
class RequestResult:
    ttft_s: float
    itl_s: list[float] = field(default_factory=list)
    output_tokens: int = 0
    total_s: float = 0.0

    @property
    def median_itl_s(self) -> float:
        return statistics.median(self.itl_s) if self.itl_s else 0.0

    @property
    def tokens_per_sec(self) -> float:
        return self.output_tokens / self.total_s if self.total_s else 0.0


async def one_request(client: AsyncOpenAI, model: str, prompt: str, max_tokens: int) -> RequestResult:
    t_start = time.perf_counter()
    t_last = None
    res = RequestResult(ttft_s=0.0)

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
        now = time.perf_counter()
        if t_last is None:
            res.ttft_s = now - t_start
        else:
            res.itl_s.append(now - t_last)
        t_last = now
        res.output_tokens += 1  # rough: 1 chunk ≈ 1 token; for exact tokens use the usage field on completion

    res.total_s = time.perf_counter() - t_start
    return res


async def run(base_url: str, model: str, n_requests: int, concurrency: int, prompt: str, max_tokens: int) -> None:
    client = AsyncOpenAI(base_url=base_url, api_key="EMPTY")
    sem = asyncio.Semaphore(concurrency)

    async def bounded() -> RequestResult:
        async with sem:
            return await one_request(client, model, prompt, max_tokens)

    t0 = time.perf_counter()
    results = await asyncio.gather(*[bounded() for _ in range(n_requests)])
    wall = time.perf_counter() - t0

    ttfts = sorted(r.ttft_s for r in results)
    itls = [r.median_itl_s for r in results]
    tps = [r.tokens_per_sec for r in results]
    total_tokens = sum(r.output_tokens for r in results)

    def pct(xs: list[float], p: float) -> float:
        if not xs:
            return 0.0
        k = max(0, min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1)))))
        return sorted(xs)[k]

    print(f"\n--- vLLM hello world: {n_requests} requests @ concurrency {concurrency} ---")
    print(f"model               {model}")
    print(f"prompt len (chars)  {len(prompt)}")
    print(f"max_tokens          {max_tokens}")
    print(f"wall time           {wall:.2f}s")
    print(f"agg throughput      {total_tokens / wall:.1f} tok/s")
    print(f"TTFT  p50 / p95 / p99   {pct(ttfts, 50)*1000:.0f} / {pct(ttfts, 95)*1000:.0f} / {pct(ttfts, 99)*1000:.0f}  ms")
    print(f"ITL   p50 / p95         {pct(itls, 50)*1000:.1f} / {pct(itls, 95)*1000:.1f}  ms")
    print(f"per-req tok/s   median  {statistics.median(tps):.1f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument(
        "--prompt",
        default="Explain why paged KV cache exists. Be concise.",
    )
    args = ap.parse_args()
    asyncio.run(run(args.base_url, args.model, args.n, args.concurrency, args.prompt, args.max_tokens))


if __name__ == "__main__":
    main()
