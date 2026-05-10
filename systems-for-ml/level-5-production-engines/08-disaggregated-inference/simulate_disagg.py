"""
08 - Toy simulation of disaggregated prefill/decode timing.

This is NOT a real disaggregated server. It's a single-process simulator
that lets you reason about when disagg helps and when it doesn't, by
plugging in numbers for:

    prefill_cost_per_token_us      compute-bound, prefill phase
    decode_cost_per_token_us       memory-bandwidth-bound, decode phase
    kv_transfer_per_token_us       network cost to move KV from prefill->decode
    n_prefill_workers              pool size of prefill GPUs
    n_decode_workers               pool size of decode GPUs

Compares two architectures on the same workload:

    A. Co-located: each request prefills+decodes on the same worker.
    B. Disaggregated: prefill on prefill pool, decode on decode pool, KV
                       transferred between them.

For each, prints aggregate throughput, mean TTFT, and total wall time
for a stream of N requests with mixed prompt/output lengths.

Use it to explore: at what QPS does disagg start winning? How much KV-
transfer overhead does it tolerate? When does asymmetric pool sizing
(more decode workers than prefill) win?
"""

from __future__ import annotations

import argparse
import heapq
import random
import statistics
from dataclasses import dataclass


@dataclass
class Request:
    id: int
    prompt_tokens: int
    output_tokens: int
    arrival_s: float
    ttft_s: float = 0.0
    finish_s: float = 0.0


def gen_requests(n: int, qps: float, seed: int = 0) -> list[Request]:
    rng = random.Random(seed)
    reqs: list[Request] = []
    t = 0.0
    for i in range(n):
        t += rng.expovariate(qps)  # Poisson arrivals
        reqs.append(
            Request(
                id=i,
                prompt_tokens=rng.choice([128, 512, 1024, 2048, 4096]),
                output_tokens=rng.choice([64, 128, 256, 512]),
                arrival_s=t,
            )
        )
    return reqs


def simulate_colocated(
    reqs: list[Request], n_workers: int, prefill_us: float, decode_us: float
) -> list[Request]:
    """
    Each worker can serve one request at a time end-to-end. (Toy: no batching.)
    Heap of worker free-times; each request picks the earliest-free worker.
    """
    workers = [0.0] * n_workers
    heap = [(0.0, i) for i in range(n_workers)]
    heapq.heapify(heap)

    for r in reqs:
        free_at, w = heapq.heappop(heap)
        start = max(free_at, r.arrival_s)
        prefill_done = start + r.prompt_tokens * prefill_us / 1e6
        r.ttft_s = prefill_done - r.arrival_s
        r.finish_s = prefill_done + r.output_tokens * decode_us / 1e6
        heapq.heappush(heap, (r.finish_s, w))
    return reqs


def simulate_disagg(
    reqs: list[Request],
    n_prefill: int,
    n_decode: int,
    prefill_us: float,
    decode_us: float,
    kv_transfer_us: float,
) -> list[Request]:
    """
    Prefill workers handle prompts; decode workers handle generation.
    KV transfer cost = kv_transfer_us * prompt_tokens.
    """
    p_heap = [(0.0, i) for i in range(n_prefill)]
    d_heap = [(0.0, i) for i in range(n_decode)]
    heapq.heapify(p_heap)
    heapq.heapify(d_heap)

    for r in reqs:
        pf_free, pw = heapq.heappop(p_heap)
        pf_start = max(pf_free, r.arrival_s)
        pf_done = pf_start + r.prompt_tokens * prefill_us / 1e6
        heapq.heappush(p_heap, (pf_done, pw))

        kv_done = pf_done + r.prompt_tokens * kv_transfer_us / 1e6

        d_free, dw = heapq.heappop(d_heap)
        d_start = max(d_free, kv_done)
        r.ttft_s = d_start - r.arrival_s  # decoder fires its first token here
        r.finish_s = d_start + r.output_tokens * decode_us / 1e6
        heapq.heappush(d_heap, (r.finish_s, dw))
    return reqs


def report(label: str, reqs: list[Request]) -> None:
    ttfts = sorted(r.ttft_s for r in reqs)
    finishes = sorted(r.finish_s for r in reqs)
    wall = max(r.finish_s for r in reqs) - min(r.arrival_s for r in reqs)
    total_out = sum(r.output_tokens for r in reqs)

    def pct(xs: list[float], p: float) -> float:
        return xs[max(0, min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1)))))]

    print(f"\n[{label}]")
    print(f"  wall                 {wall:.2f}s")
    print(f"  agg out tok/s        {total_out / wall:.0f}")
    print(f"  TTFT  mean           {statistics.mean(ttfts) * 1000:.0f} ms")
    print(f"  TTFT  p50/p95/p99    {pct(ttfts, 50)*1000:.0f} / {pct(ttfts, 95)*1000:.0f} / {pct(ttfts, 99)*1000:.0f} ms")
    print(f"  end-to-end p99       {pct(finishes, 99) * 1000:.0f} ms (since arrival)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--qps", type=float, default=8.0)
    ap.add_argument("--prefill-us", type=float, default=50.0, help="us per prompt token, prefill")
    ap.add_argument("--decode-us", type=float, default=15.0, help="us per output token, decode")
    ap.add_argument("--kv-transfer-us", type=float, default=2.0, help="us per token of KV transfer")
    ap.add_argument("--total-workers", type=int, default=4, help="for the colocated baseline")
    ap.add_argument("--n-prefill", type=int, default=1)
    ap.add_argument("--n-decode", type=int, default=3)
    args = ap.parse_args()

    base = gen_requests(args.n, args.qps)

    a = simulate_colocated(
        [Request(r.id, r.prompt_tokens, r.output_tokens, r.arrival_s) for r in base],
        n_workers=args.total_workers,
        prefill_us=args.prefill_us,
        decode_us=args.decode_us,
    )
    report(f"co-located, {args.total_workers} workers", a)

    b = simulate_disagg(
        [Request(r.id, r.prompt_tokens, r.output_tokens, r.arrival_s) for r in base],
        n_prefill=args.n_prefill,
        n_decode=args.n_decode,
        prefill_us=args.prefill_us,
        decode_us=args.decode_us,
        kv_transfer_us=args.kv_transfer_us,
    )
    report(f"disagg, {args.n_prefill} prefill + {args.n_decode} decode", b)

    print(
        "\nTry: bump --qps, vary --kv-transfer-us, change pool sizes. "
        "The crossover where disagg starts winning is workload-dependent."
    )


if __name__ == "__main__":
    main()
