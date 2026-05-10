"""
Concurrent benchmark for any OpenAI-compatible local server.

Reports:
- TTFT p50 / p90 / p99
- Output tokens/sec (aggregate across concurrent streams)
- Wall-time and completion rate

Works against vLLM-MLX, Ollama (with /v1 path), LM Studio, mlx_lm.server,
llama-server. The point is to highlight the continuous-batching gap.
"""
from __future__ import annotations
import argparse
import asyncio
import statistics
import time

import httpx


PROMPTS = [
    "Write 200 words on unified memory architecture.",
    "Explain MoE routing in three sentences.",
    "List ten optimizations for LLM inference.",
    "Describe the role of KV cache in transformer decoding.",
    "Compare MLX and llama.cpp Metal at 4-bit.",
    "Outline a local agentic loop's structure.",
    "What is speculative decoding? Be concrete.",
    "Why is bandwidth, not FLOPs, the LLM bottleneck?",
]


async def one_request(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
) -> tuple[float, int]:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "max_tokens": max_tokens,
    }
    start = time.perf_counter()
    ttft = None
    out_tokens = 0
    async with client.stream(
        "POST", f"{base_url}/chat/completions", json=body, timeout=120.0
    ) as resp:
        async for line in resp.aiter_lines():
            if not line or not line.startswith("data:"):
                continue
            if "[DONE]" in line:
                break
            if ttft is None:
                ttft = time.perf_counter() - start
            out_tokens += 1  # one chunk ~ one token (good enough for bench)
    return ttft or 0.0, out_tokens


async def main_async(args: argparse.Namespace) -> None:
    sem = asyncio.Semaphore(args.concurrency)
    ttfts: list[float] = []
    total_tokens = 0
    completed = 0

    async with httpx.AsyncClient() as client:
        async def worker(i: int) -> None:
            nonlocal total_tokens, completed
            async with sem:
                prompt = PROMPTS[i % len(PROMPTS)]
                ttft, toks = await one_request(
                    client, args.base_url, args.model, prompt, args.max_tokens
                )
                ttfts.append(ttft)
                total_tokens += toks
                completed += 1

        wall_start = time.perf_counter()
        await asyncio.gather(*[worker(i) for i in range(args.requests)])
        wall = time.perf_counter() - wall_start

    if not ttfts:
        print("No completed requests.")
        return

    ttfts_sorted = sorted(ttfts)
    p50 = ttfts_sorted[len(ttfts_sorted) // 2] * 1000
    p90 = ttfts_sorted[int(len(ttfts_sorted) * 0.9)] * 1000
    p99 = ttfts_sorted[int(len(ttfts_sorted) * 0.99)] * 1000

    print(f"Engine: {args.base_url}  model={args.model}")
    print(f"Concurrency={args.concurrency}  Requests={args.requests}")
    print(f"TTFT  p50={p50:.0f}ms  p90={p90:.0f}ms  p99={p99:.0f}ms")
    print(f"Output tok/s aggregate: ~{total_tokens / wall:.0f}")
    print(f"Wall time: {wall:.1f}s    Completed: {completed}/{args.requests}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", required=True,
                   help="e.g. http://localhost:11434/v1")
    p.add_argument("--model", required=True)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--requests", type=int, default=32)
    p.add_argument("--max-tokens", type=int, default=128)
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
