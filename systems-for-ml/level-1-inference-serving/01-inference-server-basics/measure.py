"""
Measure the three numbers from the CONCEPTS doc:
  1. Single-request latency (baseline)
  2. Sequential 10-request latency (~10× the baseline)
  3. Concurrent 10-request latency (the interesting one — almost the same as sequential)

The gap between (2) and (3) on a single-worker server is the GIL+CUDA serialization story.
This is the motivation for batching, which we'll add in topic 03.

Run it:
    1. Start the server in another terminal:  uvicorn server:app --workers 1
    2. python measure.py
"""

import asyncio
import statistics
from time import perf_counter

import httpx

URL = "http://localhost:8000/generate"
PROMPTS = [
    "Explain how a CPU cache works in two sentences.",
    "What is a vector database used for?",
    "Define recursion.",
    "What does the word 'idempotent' mean?",
    "Compare TCP and UDP briefly.",
    "What is the chain rule in calculus?",
    "Explain a hash table in one paragraph.",
    "What is a process vs a thread?",
    "Define entropy in information theory.",
    "Explain merge sort briefly.",
]


async def one_request(client: httpx.AsyncClient, prompt: str) -> dict:
    """Send one request, return (server-side latency, end-to-end latency)."""
    t0 = perf_counter()
    resp = await client.post(URL, json={"prompt": prompt, "max_tokens": 50}, timeout=120)
    end_to_end = (perf_counter() - t0) * 1000
    body = resp.json()
    return {
        "server_ms": body["latency_ms"],
        "end_to_end_ms": end_to_end,
        "tokens": body["tokens_generated"],
    }


async def sequential(client: httpx.AsyncClient, prompts: list[str]) -> list[dict]:
    """Run requests one after another. Each waits for the previous to finish."""
    return [await one_request(client, p) for p in prompts]


async def concurrent(client: httpx.AsyncClient, prompts: list[str]) -> list[dict]:
    """Run requests in parallel via asyncio.gather. They overlap on the network — but on the server?"""
    return await asyncio.gather(*[one_request(client, p) for p in prompts])


def summarize(label: str, results: list[dict], total_ms: float) -> None:
    end_to_ends = [r["end_to_end_ms"] for r in results]
    server_sides = [r["server_ms"] for r in results]
    total_tokens = sum(r["tokens"] for r in results)
    print(f"\n{label}")
    print(f"  total wall time:       {total_ms:.0f} ms")
    print(f"  per-request end-to-end (median): {statistics.median(end_to_ends):.0f} ms")
    print(f"  per-request server-side (median): {statistics.median(server_sides):.0f} ms")
    print(f"  throughput:            {total_tokens / (total_ms / 1000):.1f} tokens/sec")


async def main():
    async with httpx.AsyncClient() as client:
        # Warmup — first request includes CUDA context init + JIT, don't include it in measurements.
        print("Warming up...")
        await one_request(client, "Say hello.")

        # 1. Single request — baseline
        print("\n[1/3] Single request baseline...")
        t0 = perf_counter()
        results = await sequential(client, PROMPTS[:1])
        summarize("SINGLE", results, (perf_counter() - t0) * 1000)

        # 2. Ten requests, sequential
        print("\n[2/3] 10 requests, sequential...")
        t0 = perf_counter()
        results = await sequential(client, PROMPTS)
        summarize("SEQUENTIAL (n=10)", results, (perf_counter() - t0) * 1000)

        # 3. Ten requests, concurrent
        print("\n[3/3] 10 requests, concurrent (asyncio.gather)...")
        t0 = perf_counter()
        results = await concurrent(client, PROMPTS)
        summarize("CONCURRENT (n=10)", results, (perf_counter() - t0) * 1000)

        print("\n--- Insight ---")
        print("If sequential ~ concurrent in total wall time, you're seeing GIL+CUDA serialization.")
        print("There is no real concurrency happening on the server — requests queue up.")
        print("This motivates request batching (topic 03), where multiple requests share one forward pass.")


if __name__ == "__main__":
    asyncio.run(main())
