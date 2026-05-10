# 15 — Reasoning-Aware Serving

## Files

- `CONCEPTS.md` — what reasoning workloads break (output variance, PD ratios, KV pressure, cost projection, cancellation), reasoning budgets as gateway knobs, full cancellation-propagation path, hedging policy under long outputs.
- `cancellation_router.py` — minimal FastAPI router fragment that propagates client disconnect to the upstream stream so the engine frees its decode slot. Includes a stub upstream for self-contained testing.
- `cancellation_test.py` — workload that disconnects 30% of clients mid-stream; verify decode-slot recovery.

## Quickstart

```bash
# Terminal 1: router + stub upstream.
python cancellation_router.py

# Terminal 2: drive 30% client disconnects.
python cancellation_test.py --requests 50 --disconnect-pct 0.30
```

## Expected output

In Terminal 1 you should see lines like:
```
[stub] decode cancelled at token 7, freeing slot.
[stub] decode cancelled at token 12, freeing slot.
```
for each request that the test client disconnected mid-stream.

In Terminal 2:
```
[req   3] disconnect at 0.94s
[req   7] complete in 30.20s
[req  11] disconnect at 1.43s
...
```

The point: each "disconnect" line in T2 corresponds to a "decode cancelled" line in T1, with no per-request lag. That's correct cancellation propagation.

## Try

- **Break propagation on purpose.** In `cancellation_router.py`, replace the `is_disconnected()` poll with a `try/except` that swallows the cancellation. Re-run. Decode slots stay occupied for the full 30s — exactly the failure mode this topic is about.
- **Real vLLM upstream.** Replace the stub with a real vLLM serving a reasoning model. Disconnect 30% of clients on a long-reasoning workload. Confirm `vllm:num_requests_running` drops promptly on each disconnect.
- **Reasoning budget.** Add a `reasoning_effort` header parsed at the gateway. Map to per-tenant `max_thinking_tokens` caps and reject over-budget requests with 400.
- **Disable hedging.** Toggle a flag that disables hedging for `reasoning_effort >= medium`. Re-run a long-output workload; observe GPU-second cost halves.

## Where this goes

- Topic 08: cancellation is the missing fourth backpressure primitive. Without it, all the others over-count load.
- Topic 10: autoscaler signal selection (`time_in_queue` instead of just `num_requests_waiting`) directly addresses long-output regimes.
- Topic 13: cost projection for reasoning tiers needs p99 output-length budgets, not means.
- `reports/platform.md`: include the disconnect-handling experiment as a numbered finding.

## References

- vLLM cancellation handling — https://docs.vllm.ai/en/latest/
- httpx streaming + cancellation — https://www.python-httpx.org/async/
- llm-d Variant Autoscaler — https://llm-d.ai/docs/architecture
- Artificial Analysis reasoning benchmarks — https://artificialanalysis.ai/
