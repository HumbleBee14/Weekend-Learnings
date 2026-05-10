"""
Drive the router with a chatbot-shaped workload to measure the prefix-vs-random
TTFT delta.

Usage:
    python bench.py --router http://localhost:8080 \
        --shared-prefix 4096 --suffix-tokens 64 --requests 200
"""

import argparse
import asyncio
import statistics
import time

import httpx


PREFIX_TEMPLATE = "You are a helpful assistant. " * 200  # ~4KB-ish at small models
SUFFIX_POOL = [f"User question {i}: explain topic #{i}." for i in range(10000)]


async def one_request(client, url, body):
    t0 = time.perf_counter()
    ttft = None
    async with client.stream("POST", url, json=body) as r:
        async for chunk in r.aiter_raw():
            if ttft is None and chunk:
                ttft = time.perf_counter() - t0
                break
    return ttft if ttft is not None else (time.perf_counter() - t0)


async def run(args):
    prefix = PREFIX_TEMPLATE
    if args.shared_prefix > 0:
        prefix = (PREFIX_TEMPLATE * (args.shared_prefix // len(PREFIX_TEMPLATE) + 1))[: args.shared_prefix]

    bodies = []
    for i in range(args.requests):
        suffix = SUFFIX_POOL[i % len(SUFFIX_POOL)]
        bodies.append({
            "model": args.model,
            "messages": [
                {"role": "system", "content": prefix},
                {"role": "user", "content": suffix},
            ],
            "stream": True,
            "max_tokens": 16,
        })

    sem = asyncio.Semaphore(args.concurrency)
    ttfts = []

    async def worker(body):
        async with sem, httpx.AsyncClient(timeout=60) as client:
            t = await one_request(client, f"{args.router}/v1/chat/completions", body)
            ttfts.append(t)

    await asyncio.gather(*[worker(b) for b in bodies])

    ttfts.sort()
    n = len(ttfts)
    print(f"requests:   {n}")
    print(f"TTFT mean:  {statistics.mean(ttfts)*1000:.1f} ms")
    print(f"TTFT p50:   {ttfts[n//2]*1000:.1f} ms")
    print(f"TTFT p95:   {ttfts[int(n*0.95)]*1000:.1f} ms")
    print(f"TTFT p99:   {ttfts[int(n*0.99)]*1000:.1f} ms")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--router", default="http://localhost:8080")
    p.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    p.add_argument("--shared-prefix", type=int, default=4096)
    p.add_argument("--suffix-tokens", type=int, default=64)
    p.add_argument("--requests", type=int, default=200)
    p.add_argument("--concurrency", type=int, default=8)
    args = p.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
