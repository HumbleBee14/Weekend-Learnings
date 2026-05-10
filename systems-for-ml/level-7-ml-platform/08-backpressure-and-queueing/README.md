# 08 — Backpressure and Queueing

## Files

- `CONCEPTS.md` — Little's Law as a debugging tool, three backpressure mechanisms, where the queue lives, hedging tradeoffs.
- `littles_law_check.py` — Pulls L, λ, W from Prometheus and verifies L ≈ λW. Returns nonzero exit on >10% drift.
- `backpressure.py` — `BoundedQueue`, `SLOAwareAdmission`, `Hedger` primitives plus a demo.

## Quickstart

```bash
python littles_law_check.py --prom http://localhost:9090 --window 5m
python backpressure.py
```

## Expected output

```
$ python littles_law_check.py
{
  "L_observed": 7.2,
  "lambda_rps": 18.4,
  "W_seconds": 0.385,
  "L_predicted": 7.084,
  "relative_error": -0.016,
  "verdict": "PASS"
}
```

`>10%` drift fails the check — the signal that one of (`L`, `λ`, `W`) is mis-instrumented. Common cause: `λ` measured at the router instead of the gateway, missing rate-limited rejections.

```
$ python backpressure.py
BoundedQueue: admitted=4, rejected=16
Hedger  p50=42.1ms p99=120.5ms
NoHedge p50=41.3ms p99=403.2ms
```

The hedger flattens p99 by ~3x with marginal p50 cost — the Tail-at-Scale finding.

## Try

- **Drift the law on purpose.** Hold a request without decrementing the running counter. Re-run; confirm the check fails. This is exactly the bug-class the script catches.
- **G13.** Drive load up linearly; sample (depth, latency); plot; overlay predicted curve from L = λW.
- **SLO admission.** Swap `BoundedQueue` for `SLOAwareAdmission(slo=0.5)`. Mix slow + fast costs; observe early shedding of slow ones.
- **Smart hedge.** Have `Hedger` consult Topic 06's PrefixStore — hedge only to a replica holding >50% of the prompt's blocks.

## Where this goes

- Topic 09: scheduling policies sit one layer above — among admitted requests, which runs next.
- Topic 10: KEDA scales on `num_requests_waiting`, the same signal backpressure reads.
- Topic 15: cancellation propagation is the missing fourth backpressure primitive — most commonly skipped.
