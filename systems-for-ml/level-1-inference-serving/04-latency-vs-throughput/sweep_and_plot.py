"""
Run the throughput-vs-latency sweep and plot G1.

Hits the topic-03 batched server at multiple concurrency levels, records throughput and
percentiles, then writes a CSV and a PNG plot.

Run:
    1. Terminal A: from topic 03's folder, uvicorn server:app --workers 1 --port 8000
    2. Terminal B: python sweep_and_plot.py
"""

import asyncio
import csv
import statistics
from time import perf_counter

import httpx

URL = "http://localhost:8000/generate"
PROMPTS = [
    "Define recursion in two sentences.",
    "What is a hash table?",
    "Explain TCP vs UDP briefly.",
    "What is virtual memory?",
    "Explain merge sort in one paragraph.",
] * 8  # 40 prompts


async def one_request(client, prompt):
    t0 = perf_counter()
    resp = await client.post(URL, json={"prompt": prompt, "max_tokens": 50}, timeout=300)
    end_to_end = (perf_counter() - t0) * 1000
    body = resp.json()
    return end_to_end, body["tokens_generated"]


async def sweep_one(client, concurrency: int, n_total: int = 30):
    sem = asyncio.Semaphore(concurrency)

    async def bounded(p):
        async with sem:
            return await one_request(client, p)

    prompts = (PROMPTS * (n_total // len(PROMPTS) + 1))[:n_total]
    t0 = perf_counter()
    results = await asyncio.gather(*[bounded(p) for p in prompts])
    total_s = perf_counter() - t0

    latencies = sorted(r[0] for r in results)
    total_tokens = sum(r[1] for r in results)
    return {
        "concurrency": concurrency,
        "throughput_tps": total_tokens / total_s,
        "p50_ms": latencies[len(latencies) // 2],
        "p95_ms": latencies[int(len(latencies) * 0.95)],
        "p99_ms": latencies[int(len(latencies) * 0.99)],
    }


async def main():
    async with httpx.AsyncClient() as client:
        # Warmup
        await one_request(client, "Say hi.")

        rows = []
        for c in [1, 2, 4, 8, 12, 16, 24, 32]:
            row = await sweep_one(client, c)
            print(row)
            rows.append(row)

        # Write CSV
        with open("g1_data.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        print("Wrote g1_data.csv")

        # Plot
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Skipping plot (pip install matplotlib).")
            return

        concurrencies = [r["concurrency"] for r in rows]
        throughputs = [r["throughput_tps"] for r in rows]
        p99s = [r["p99_ms"] for r in rows]

        fig, ax1 = plt.subplots(figsize=(8, 5))
        color1 = "tab:blue"
        ax1.set_xlabel("concurrency")
        ax1.set_ylabel("throughput (tokens/sec)", color=color1)
        ax1.plot(concurrencies, throughputs, "o-", color=color1, label="throughput")
        ax1.tick_params(axis="y", labelcolor=color1)

        ax2 = ax1.twinx()
        color2 = "tab:red"
        ax2.set_ylabel("p99 latency (ms)", color=color2)
        ax2.plot(concurrencies, p99s, "s--", color=color2, label="p99")
        ax2.tick_params(axis="y", labelcolor=color2)

        plt.title("G1: throughput vs p99 latency")
        fig.tight_layout()
        plt.savefig("g1_plot.png", dpi=120)
        print("Wrote g1_plot.png")


if __name__ == "__main__":
    asyncio.run(main())
