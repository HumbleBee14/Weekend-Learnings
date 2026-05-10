"""
Token-aware WFQ admission for the mini-platform router.

Two pieces:
  1. TokenBucket — gateway-side rate limit per tenant.
  2. DRRAdmission — Deficit Round Robin admission over per-tenant queues,
     where the head-of-queue cost is measured in tokens (input + max_output).

The admit() coroutine yields a request when its turn comes up. Wire this
between the FastAPI handler and the upstream vLLM call.

Run the demo:
    python wfq_admission.py
"""

from __future__ import annotations

import asyncio
import collections
import time
from dataclasses import dataclass, field


# ---------- gateway: token-bucket rate limit ----------

@dataclass
class TokenBucket:
    capacity: int                   # max tokens stored
    refill_rate_per_s: float        # tokens added per second
    tokens: float = 0.0
    last_refill: float = field(default_factory=time.monotonic)

    def __post_init__(self):
        self.tokens = float(self.capacity)

    def try_consume(self, n: int) -> bool:
        now = time.monotonic()
        self.tokens = min(self.capacity,
                          self.tokens + (now - self.last_refill) * self.refill_rate_per_s)
        self.last_refill = now
        if self.tokens >= n:
            self.tokens -= n
            return True
        return False


class TenantQuotas:
    """Per-tenant token-bucket: input tok/min and output tok/min."""

    def __init__(self, config: dict):
        self.config = config
        self.input_buckets: dict[str, TokenBucket] = {}
        self.output_buckets: dict[str, TokenBucket] = {}

    def _bucket_for(self, tenant: str, kind: str) -> TokenBucket:
        cache = self.input_buckets if kind == "input" else self.output_buckets
        if tenant not in cache:
            cfg = self.config.get(tenant) or self.config["default"]
            per_min = cfg[f"{kind}_tok_per_min"]
            cache[tenant] = TokenBucket(capacity=per_min, refill_rate_per_s=per_min / 60.0)
        return cache[tenant]

    def admit_request(self, tenant: str, in_tokens: int, max_out_tokens: int) -> bool:
        cfg = self.config.get(tenant) or self.config["default"]
        if max_out_tokens > cfg["max_tokens_per_request"]:
            return False
        if not self._bucket_for(tenant, "input").try_consume(in_tokens):
            return False
        # Reserve output tokens up front; refunded on cancellation if you want strict accounting.
        if not self._bucket_for(tenant, "output").try_consume(max_out_tokens):
            return False
        return True


# ---------- router: DRR over per-tenant queues ----------

@dataclass
class PendingRequest:
    tenant: str
    cost_tokens: int
    fut: asyncio.Future


class DRRAdmission:
    """
    Deficit Round Robin admission. Each tenant has weight w_i; one DRR
    "round" gives tenant i a quantum of w_i tokens. The head-of-queue
    request is admitted when its tenant's deficit covers its token cost.
    """

    def __init__(self, weights: dict[str, int]):
        self.weights = weights
        self.queues: dict[str, collections.deque[PendingRequest]] = {
            t: collections.deque() for t in weights
        }
        self.deficit: dict[str, int] = {t: 0 for t in weights}
        self._cv = asyncio.Condition()
        self._closed = False

    async def submit(self, tenant: str, cost_tokens: int) -> None:
        """Wait until DRR admits this request."""
        if tenant not in self.queues:
            self.queues[tenant] = collections.deque()
            self.deficit[tenant] = 0
            self.weights[tenant] = self.weights.get("default", 1)

        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        async with self._cv:
            self.queues[tenant].append(PendingRequest(tenant, cost_tokens, fut))
            self._cv.notify_all()
        await fut

    async def run(self):
        """Admission loop. Run as a background task."""
        while not self._closed:
            async with self._cv:
                # Wait until there is at least one pending request.
                while not any(self.queues.values()):
                    await self._cv.wait()

                progress = False
                for tenant, q in self.queues.items():
                    self.deficit[tenant] += self.weights[tenant]
                    while q and q[0].cost_tokens <= self.deficit[tenant]:
                        head = q.popleft()
                        self.deficit[tenant] -= head.cost_tokens
                        head.fut.set_result(None)
                        progress = True
                if not progress:
                    # Avoid a tight spin: yield briefly when no head fits any deficit.
                    await asyncio.sleep(0.001)


# ---------- demo ----------

async def _demo():
    quotas = TenantQuotas({
        "default": {"input_tok_per_min": 10_000, "output_tok_per_min": 10_000,
                    "max_tokens_per_request": 2048},
        "tenant_b": {"input_tok_per_min": 60_000, "output_tok_per_min": 60_000,
                     "max_tokens_per_request": 8192},
    })
    admit = DRRAdmission(weights={"tenant_a": 2, "tenant_b": 1, "default": 1})
    asyncio.create_task(admit.run())

    async def emit(tenant: str, cost: int, label: str):
        ok = quotas.admit_request(tenant, in_tokens=cost // 2, max_out_tokens=cost // 2)
        if not ok:
            print(f"[gateway 429] {label} {tenant} cost={cost}")
            return
        await admit.submit(tenant, cost)
        print(f"[admitted] {label} {tenant} cost={cost}")

    # tenant_a: many small chats; tenant_b: one giant prompt.
    tasks = [emit("tenant_a", 200, f"a-{i}") for i in range(8)]
    tasks += [emit("tenant_b", 4000, "b-big")]
    await asyncio.gather(*tasks)
    await asyncio.sleep(0.05)


if __name__ == "__main__":
    asyncio.run(_demo())
