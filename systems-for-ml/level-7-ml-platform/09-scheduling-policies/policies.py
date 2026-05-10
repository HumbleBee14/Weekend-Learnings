"""
Three scheduling policies as drop-in queue orderings for the router.

Each policy exposes:
    enqueue(request)
    next() -> request | None

  FCFS:     simple FIFO.
  Priority: lower priority value = served first; aging promotes.
  SJF:      lowest projected token-cost first; aging promotes.

Run the simulation:
    python policies.py
"""

from __future__ import annotations

import collections
import heapq
import time
import random
import statistics
from dataclasses import dataclass, field


@dataclass(order=False)
class Request:
    rid: int
    arrival: float
    cost_tokens: int               # prompt + projected output
    priority: int = 5              # 0=highest, 9=lowest
    started: float | None = None
    finished: float | None = None


# ---------- policies ----------

class FCFS:
    name = "FCFS"

    def __init__(self):
        self.q: collections.deque[Request] = collections.deque()

    def enqueue(self, r: Request):
        self.q.append(r)

    def next(self) -> Request | None:
        return self.q.popleft() if self.q else None


class Priority:
    """
    Min-heap on (effective_priority, arrival).
    effective_priority = priority - aging_seconds * AGING_SLOPE
    """
    name = "Priority+aging"

    AGING_SLOPE = 1.0   # waited 1s -> -1 to priority

    def __init__(self):
        self.h: list[tuple[float, float, int, Request]] = []
        self._tick = 0

    def enqueue(self, r: Request):
        eff = r.priority
        heapq.heappush(self.h, (eff, r.arrival, self._tick, r))
        self._tick += 1

    def next(self) -> Request | None:
        if not self.h:
            return None
        # Re-age every call to keep the heap honest at the head.
        now = time.monotonic()
        while self.h:
            eff, arr, tk, r = self.h[0]
            new_eff = r.priority - (now - r.arrival) * self.AGING_SLOPE
            if new_eff != eff:
                heapq.heapreplace(self.h, (new_eff, arr, tk, r))
            else:
                heapq.heappop(self.h)
                return r
        return None


class SJF:
    """
    Shortest projected job first, with aging to prevent long-prompt starvation.
    Effective key = cost - aging_bonus(now - arrival).
    """
    name = "SJF+aging"

    AGING_TOK_PER_S = 200    # for each second waited, treat as if 200 fewer tokens.

    def __init__(self):
        self.h: list[tuple[float, float, int, Request]] = []
        self._tick = 0

    def enqueue(self, r: Request):
        heapq.heappush(self.h, (r.cost_tokens, r.arrival, self._tick, r))
        self._tick += 1

    def next(self) -> Request | None:
        if not self.h:
            return None
        now = time.monotonic()
        while self.h:
            key, arr, tk, r = self.h[0]
            new_key = r.cost_tokens - (now - r.arrival) * self.AGING_TOK_PER_S
            if new_key != key:
                heapq.heapreplace(self.h, (new_key, arr, tk, r))
            else:
                heapq.heappop(self.h)
                return r
        return None


# ---------- simulator ----------

def simulate(policy, *, total_requests: int = 600, mix_long_pct: float = 0.10,
             arrival_rate_rps: float = 40.0, service_tok_per_s: float = 4000.0,
             concurrency: int = 4) -> dict:
    """
    Discrete-event-ish simulator. We pretend the engine processes
    `service_tok_per_s` tokens of work per second across `concurrency` slots.
    """
    rng = random.Random(0)
    workload: list[Request] = []
    t = 0.0
    for i in range(total_requests):
        is_long = rng.random() < mix_long_pct
        cost = rng.randint(8000, 32000) if is_long else rng.randint(150, 400)
        prio = 8 if is_long else 5     # tag long as low priority for Priority demo
        t += rng.expovariate(arrival_rate_rps)
        workload.append(Request(rid=i, arrival=t, cost_tokens=cost, priority=prio))

    workload.sort(key=lambda r: r.arrival)
    pending = collections.deque(workload)
    sim_now = 0.0
    finished: list[Request] = []
    slots: list[Request | None] = [None] * concurrency

    while pending or any(slots):
        # Enqueue everything that's arrived by now.
        while pending and pending[0].arrival <= sim_now:
            policy.enqueue(pending.popleft())

        # Fill empty slots.
        for i, s in enumerate(slots):
            if s is None:
                r = policy.next()
                if r is None:
                    continue
                r.started = sim_now
                slots[i] = r

        # Advance time by min(next slot completion, next arrival).
        slot_completions = [
            s.started + s.cost_tokens / service_tok_per_s
            for s in slots if s is not None
        ]
        next_arrival = pending[0].arrival if pending else float("inf")
        next_event = min([next_arrival] + slot_completions) if slot_completions else next_arrival
        sim_now = max(sim_now, next_event)

        # Free completed slots.
        for i, s in enumerate(slots):
            if s is None:
                continue
            done_at = s.started + s.cost_tokens / service_tok_per_s
            if done_at <= sim_now + 1e-9:
                s.finished = done_at
                finished.append(s)
                slots[i] = None

        if not pending and not any(slots):
            break

    short = [r for r in finished if r.cost_tokens < 1000]
    long_ = [r for r in finished if r.cost_tokens >= 1000]

    def pct(xs, p):
        xs = sorted(xs)
        if not xs:
            return float("nan")
        return xs[min(len(xs) - 1, int(len(xs) * p))]

    short_ttft = [r.started - r.arrival for r in short]
    long_ttft = [r.started - r.arrival for r in long_]

    return {
        "policy": policy.name,
        "n_short": len(short),
        "n_long": len(long_),
        "short_ttft_p50_ms": round(statistics.median(short_ttft) * 1000, 1) if short_ttft else None,
        "short_ttft_p99_ms": round(pct(short_ttft, 0.99) * 1000, 1) if short_ttft else None,
        "long_ttft_p50_ms": round(statistics.median(long_ttft) * 1000, 1) if long_ttft else None,
        "long_ttft_p99_ms": round(pct(long_ttft, 0.99) * 1000, 1) if long_ttft else None,
    }


def main():
    import json
    for cls in (FCFS, Priority, SJF):
        # Each fresh policy instance.
        out = simulate(cls())
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
