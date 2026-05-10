"""
Drive the cancellation-propagating router with N% client-disconnect requests.
Measure how fast decode slots recover vs how long they would otherwise hang.

    python cancellation_test.py --base http://localhost:8080 \\
        --requests 50 --disconnect-pct 0.30
"""

import argparse
import asyncio
import random
import time

import httpx


async def maybe_cancel(args, idx: int):
    will_cancel = random.random() < args.disconnect_pct
    cancel_at = random.uniform(0.5, 3.0) if will_cancel else None

    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=120) as c:
            async with c.stream("POST", f"{args.base}/v1/chat/completions",
                                json={"messages": [], "stream": True}) as r:
                async for _ in r.aiter_raw():
                    if cancel_at is not None and time.perf_counter() - t0 > cancel_at:
                        print(f"[req {idx:>3}] disconnect at {cancel_at:.2f}s")
                        return
    except (httpx.ReadError, asyncio.TimeoutError):
        pass
    print(f"[req {idx:>3}] complete in {time.perf_counter()-t0:.2f}s")


async def main_async(args):
    sem = asyncio.Semaphore(args.concurrency)

    async def one(i):
        async with sem:
            await maybe_cancel(args, i)

    await asyncio.gather(*[one(i) for i in range(args.requests)])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="http://localhost:8080")
    p.add_argument("--requests", type=int, default=50)
    p.add_argument("--disconnect-pct", type=float, default=0.30)
    p.add_argument("--concurrency", type=int, default=8)
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
