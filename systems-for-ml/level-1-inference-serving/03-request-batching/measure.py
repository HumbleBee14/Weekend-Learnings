"""
Sweep concurrency levels against the batched server and record throughput + latency.

Run:
    1. Terminal A: uvicorn server:app --workers 1 --port 8000
    2. Terminal B: python measure.py
"""

import asyncio
import statistics
from time import perf_counter

import httpx

URL = "http://localhost:8000/generate"
PROMPT_POOL = [
    "Define recursion in two sentences.",
    "What is a hash table?",
    "Explain TCP vs UDP.",
    "What is virtual memory?",
    "Explain merge sort.",
    "What is a database index?",
    "Define entropy in information theory.",
    "What does idempotent mean?",
    "Explain the chain rule.",
    "What is a context manager in Python?",
] * 4  # 40 prompts total


async def one_request(client: httpx.AsyncClient, prompt: str) -> dict:
    t0 = perf_counter()
    resp = await client.post(URL, json={"prompt": prompt, "max_tokens": 50}, timeout=300)
    end_to_end = (perf_counter() - t0) * 1000
    body = resp.json()
    return {
        "end_to_end_ms": end_to_end,
        "queue_wait_ms": body["queue_wait_ms"],
        "batch_run_ms": body["batch_run_ms"],
        "batch_size_seen": body["batch_size"],
        "tokens": body["tokens_generated"],
    }


async def run_at_concurrency(client: httpx.AsyncClient, concurrency: int, n_total: int) -> dict:
    """Send n_total requests with at most `concurrency` in flight at any moment."""
    sem = asyncio.Semaphore(concurrency)
    prompts = (PROMPT_POOL * (n_total // len(PROMPT_POOL) + 1))[:n_total]

    async def bounded(p):
        async with sem:
            return await one_request(client, p)

    t0 = perf_counter()
    results = await asyncio.gather(*[bounded(p) for p in prompts])
    total_s = perf_counter() - t0

    end_to_ends = sorted(r["end_to_end_ms"] for r in results)
    total_tokens = sum(r["tokens"] for r in results)
    batch_sizes = [r["batch_size_seen"] for r in results]

    return {
        "concurrency": concurrency,
        "n": n_total,
        "throughput_tokens_per_sec": total_tokens / total_s,
        "p50_ms": end_to_ends[len(end_to_ends) // 2],
        "p95_ms": end_to_ends[int(len(end_to_ends) * 0.95)],
        "p99_ms": end_to_ends[int(len(end_to_ends) * 0.99)],
        "avg_batch_size": statistics.mean(batch_sizes),
    }


async def main():
    async with httpx.AsyncClient() as client:
        # Warmup
        print("Warmup...")
        await one_request(client, "Say hi.")

        print("\n{:<12} {:<10} {:<14} {:<10} {:<10} {:<10} {:<10}".format(
            "concurrency", "n", "tok/s", "p50_ms", "p95_ms", "p99_ms", "avg_batch"))

        for concurrency in [1, 2, 4, 8, 16]:
            stats = await run_at_concurrency(client, concurrency, n_total=20)
            print("{:<12} {:<10} {:<14.1f} {:<10.0f} {:<10.0f} {:<10.0f} {:<10.2f}".format(
                stats["concurrency"], stats["n"], stats["throughput_tokens_per_sec"],
                stats["p50_ms"], stats["p95_ms"], stats["p99_ms"], stats["avg_batch_size"]))

        print("\n--- Notes ---")
        print("Throughput should rise with concurrency, then flatten when batch fills consistently.")
        print("Latency stays roughly stable until concurrency exceeds the batcher's capacity.")
        print("avg_batch_size shows whether requests are actually getting batched together.")


if __name__ == "__main__":
    asyncio.run(main())
