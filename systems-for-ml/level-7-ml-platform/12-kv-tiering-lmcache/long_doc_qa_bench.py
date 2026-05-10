"""
Long-doc Q&A workload to surface KV-tiering gains.

Same document prefix (large), varied questions. First request pays full prefill;
subsequent requests should hit LMCache and TTFT collapses.

    python long_doc_qa_bench.py --base http://localhost:8000 \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --doc-tokens 16000 --questions 20
"""

import argparse
import asyncio
import time
import statistics

import httpx


DOC_TEMPLATE = (
    "The following is a technical document on distributed inference systems. "
    "It describes paged KV cache, continuous batching, prefix caching, and "
    "cross-replica coherence in detail. " * 1000
)

QUESTIONS = [
    "What is paged KV cache?",
    "Explain continuous batching.",
    "How does prefix caching work?",
    "What is NIXL used for?",
    "Define cross-replica coherence.",
    "What is the role of LMCache?",
    "Why does TTFT matter?",
    "What is chunked prefill?",
    "Explain disaggregated inference.",
    "What does Mooncake do?",
] * 5  # repeat to reach a stable hit-rate


async def ask(client, base, model, doc, q):
    t0 = time.perf_counter()
    ttft = None
    async with client.stream(
        "POST", f"{base}/v1/chat/completions",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": doc},
                {"role": "user", "content": q},
            ],
            "stream": True,
            "max_tokens": 32,
        },
    ) as r:
        async for chunk in r.aiter_raw():
            if ttft is None and chunk:
                ttft = time.perf_counter() - t0
                break
    return ttft if ttft is not None else (time.perf_counter() - t0)


async def main_async(args):
    doc = DOC_TEMPLATE[: args.doc_tokens * 4]   # ~4 chars per token approximation
    qs = QUESTIONS[: args.questions]

    async with httpx.AsyncClient(timeout=600) as client:
        ttfts = []
        for i, q in enumerate(qs):
            t = await ask(client, args.base, args.model, doc, q)
            print(f"Q{i:>2}  TTFT={t*1000:7.0f}ms   '{q[:32]}'")
            ttfts.append(t)

    cold = ttfts[0]
    warm = statistics.median(ttfts[1:]) if len(ttfts) > 1 else float("nan")
    print(f"\ncold (first): {cold*1000:.0f} ms")
    print(f"warm (median rest): {warm*1000:.0f} ms")
    print(f"speedup: {cold/warm:.1f}x")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="http://localhost:8000")
    p.add_argument("--model", required=True)
    p.add_argument("--doc-tokens", type=int, default=16000)
    p.add_argument("--questions", type=int, default=20)
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
