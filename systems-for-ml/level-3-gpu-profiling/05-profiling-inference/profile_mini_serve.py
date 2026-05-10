"""
Profile your Level-1 mini-serve under load using torch.profiler.

This is a CLIENT-SIDE driver script. It assumes mini-serve (or vLLM) is running
at http://localhost:8000.

Run:
    pip install torch httpx
    # Start mini-serve in another terminal first
    python profile_mini_serve.py

For deeper analysis, run mini-serve under nsys:
    nsys profile -t cuda,nvtx --cuda-graph-trace=node \\
      --capture-range=cudaProfilerApi --capture-range-end=stop \\
      -o mini_serve_trace.nsys-rep \\
      python -m uvicorn server:app --workers 1
    # Then, while it's serving, run this script to send load.
"""

import asyncio
import statistics
import time
from pathlib import Path

import httpx

URL = "http://localhost:8000/generate"

# Three workloads — pick one or run all three to characterize different regimes
WORKLOADS = {
    "prefill_dominant": {
        "prompt": "x " * 1000,           # ~1000 tokens of input
        "max_tokens": 1,                  # almost no decode
        "concurrency": 16,
        "n_requests": 32,
    },
    "decode_dominant": {
        "prompt": "Hello.",               # tiny input
        "max_tokens": 200,                # long output
        "concurrency": 16,
        "n_requests": 32,
    },
    "mixed_realistic": {
        "prompt": "Explain how a hash table works in three paragraphs.",
        "max_tokens": 100,
        "concurrency": 16,
        "n_requests": 32,
    },
}


async def one_request(client: httpx.AsyncClient, payload: dict):
    t0 = time.perf_counter()
    resp = await client.post(URL, json=payload, timeout=300)
    end_to_end = (time.perf_counter() - t0) * 1000
    body = resp.json()
    return end_to_end, body


async def run_workload(client: httpx.AsyncClient, name: str, cfg: dict):
    print(f"\n{name}: {cfg['n_requests']} requests at concurrency {cfg['concurrency']}")
    print(f"  payload: prompt[{len(cfg['prompt'])}], max_tokens={cfg['max_tokens']}")

    sem = asyncio.Semaphore(cfg["concurrency"])
    payload = {
        "prompt": cfg["prompt"],
        "max_tokens": cfg["max_tokens"],
    }

    async def bounded():
        async with sem:
            return await one_request(client, payload)

    t0 = time.perf_counter()
    results = await asyncio.gather(*[bounded() for _ in range(cfg["n_requests"])])
    total_s = time.perf_counter() - t0

    latencies = sorted([r[0] for r in results])
    print(f"  total wall time: {total_s:.2f}s")
    print(f"  median latency:  {statistics.median(latencies):.0f}ms")
    print(f"  p95 latency:     {latencies[int(len(latencies) * 0.95)]:.0f}ms")
    print(f"  p99 latency:     {latencies[int(len(latencies) * 0.99)]:.0f}ms")


async def main():
    async with httpx.AsyncClient() as client:
        # Warmup — discard
        print("Warming up...")
        await one_request(client, {"prompt": "Hi", "max_tokens": 5})

        for name, cfg in WORKLOADS.items():
            await run_workload(client, name, cfg)

        print("\nNow profile the server while you run this script:")
        print("  - torch.profiler: VLLM_TORCH_PROFILER_DIR=/tmp/traces /start_profile + /stop_profile")
        print("  - nsys: nsys profile ... python -m uvicorn server:app")
        print("  - For your Level-1 mini-serve, wrap the model.generate() call in nvtx.range")


if __name__ == "__main__":
    asyncio.run(main())
