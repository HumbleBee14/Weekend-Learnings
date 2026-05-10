"""
Three backpressure primitives, as drop-in middleware for the Topic 06 router:

  1. BoundedQueue       — reject (429) past max depth.
  2. SLOAwareAdmission  — reject if predicted W exceeds SLO.
  3. Hedger             — fire duplicate request after timeout, cancel slower.

Wire them in front of `router.pick` / upstream forward call.

Demo:
    python backpressure.py
"""

import asyncio
import time
import statistics
import random


class BoundedQueue:
    def __init__(self, max_depth: int):
        self.max_depth = max_depth
        self.depth = 0

    def admit(self) -> bool:
        if self.depth >= self.max_depth:
            return False
        self.depth += 1
        return True

    def done(self):
        self.depth = max(0, self.depth - 1)


class SLOAwareAdmission:
    """
    Reject up front if the predicted wait already busts the SLO.
    Throughput is an EWMA of recent completed-rps; queue depth is shared.
    """

    def __init__(self, slo_seconds: float, alpha: float = 0.2):
        self.slo = slo_seconds
        self.alpha = alpha
        self.throughput = 1.0    # req/s; bootstrap > 0 so first request admits
        self.queue_depth = 0
        self._last_completion = time.monotonic()
        self._completed_window = []

    def admit(self, est_service_time: float) -> bool:
        predicted_W = self.queue_depth / max(self.throughput, 0.1)
        if predicted_W + est_service_time > self.slo:
            return False
        self.queue_depth += 1
        return True

    def done(self):
        self.queue_depth = max(0, self.queue_depth - 1)
        now = time.monotonic()
        self._completed_window.append(now)
        # Trim window to last 5s.
        cutoff = now - 5.0
        self._completed_window = [t for t in self._completed_window if t > cutoff]
        rate = len(self._completed_window) / 5.0
        self.throughput = (1 - self.alpha) * self.throughput + self.alpha * rate


class Hedger:
    """Fire a hedge after `timeout_s` if no first-token yet."""

    def __init__(self, timeout_s: float):
        self.timeout_s = timeout_s

    async def race(self, primary_coro, hedge_coro_factory):
        primary = asyncio.create_task(primary_coro)
        try:
            return await asyncio.wait_for(asyncio.shield(primary), self.timeout_s)
        except asyncio.TimeoutError:
            hedge = asyncio.create_task(hedge_coro_factory())
            done, pending = await asyncio.wait(
                {primary, hedge}, return_when=asyncio.FIRST_COMPLETED
            )
            for p in pending:
                p.cancel()
            return list(done)[0].result()


# ---------- demo ----------

async def fake_call(latency_p50=0.04, latency_p99=0.4):
    # Most fast, occasional slow.
    if random.random() < 0.05:
        await asyncio.sleep(latency_p99)
        return "slow"
    await asyncio.sleep(latency_p50)
    return "fast"


async def _demo():
    bq = BoundedQueue(max_depth=4)
    admitted = rejected = 0
    for _ in range(20):
        if bq.admit():
            admitted += 1
            asyncio.create_task(asyncio.sleep(0.02)).add_done_callback(lambda _: bq.done())
        else:
            rejected += 1
        await asyncio.sleep(0.005)
    print(f"BoundedQueue: admitted={admitted}, rejected={rejected}")

    h = Hedger(timeout_s=0.10)
    lat = []
    for _ in range(200):
        t0 = time.perf_counter()
        await h.race(fake_call(), fake_call)
        lat.append(time.perf_counter() - t0)
    lat.sort()
    print(f"Hedger p50={lat[100]*1000:.1f}ms p99={lat[198]*1000:.1f}ms")

    lat = []
    for _ in range(200):
        t0 = time.perf_counter()
        await fake_call()
        lat.append(time.perf_counter() - t0)
    lat.sort()
    print(f"NoHedge p50={lat[100]*1000:.1f}ms p99={lat[198]*1000:.1f}ms")


if __name__ == "__main__":
    asyncio.run(_demo())
