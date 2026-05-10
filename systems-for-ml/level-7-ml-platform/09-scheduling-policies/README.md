# 09 — Scheduling Policies

## Files

- `CONCEPTS.md` — what scheduling means inside continuous-batching, three policies (FCFS / Priority+aging / SJF+aging), how chunked prefill changes the picture.
- `policies.py` — three policy implementations + a deterministic simulator that drives mixed-length traffic and reports per-class TTFT.

## Quickstart

```bash
python policies.py
```

## Expected output (shape)

```json
{
  "policy": "FCFS",
  "n_short": 540, "n_long": 60,
  "short_ttft_p50_ms": 12.0,  "short_ttft_p99_ms": 4830.0,
  "long_ttft_p50_ms":  6.0,   "long_ttft_p99_ms":   55.0
}
{
  "policy": "Priority+aging",
  "short_ttft_p99_ms": 1140.0,
  "long_ttft_p99_ms": 18000.0
}
{
  "policy": "SJF+aging",
  "short_ttft_p99_ms": 95.0,
  "long_ttft_p99_ms": 8400.0
}
```

The story:
- **FCFS** lets long prompts wreck short-prompt p99.
- **Priority** (with long tagged low-priority) protects short-prompt p99 at the cost of long-prompt p99 — by design.
- **SJF** also protects short-prompt p99, with aging keeping long-prompt p99 finite.

These numbers are simulated. Real numbers from your engine will differ; the *direction* is what generalises.

## Try

- **Workload mix.** Drive `mix_long_pct=0.30`. Watch FCFS short-prompt p99 grow proportionally; SJF stays bounded as long as aging is on.
- **Aging slope.** Halve `SJF.AGING_TOK_PER_S`. Long prompts wait longer; short-prompt p99 stays low until the queue grows.
- **Real engine.** Run vLLM with `--scheduling-policy fcfs` vs `--scheduling-policy priority` and re-run Topic 06's `bench.py` with mixed-length workload.
- **Chunked prefill on/off.** Toggle `--enable-chunked-prefill`. The policy gap should shrink dramatically — chunked prefill is doing some of SJF's job for free.

## G16 measurement plan

- Same workload, same hardware, same engine.
- Three runs: FCFS, Priority+aging, SJF (real or simulated; ideally real vLLM).
- Capture: short-prompt p99 TTFT, long-prompt p99 TTFT, total throughput.
- The headline is the trade-off, not a winner.

## Where this goes

- Topic 07: WFQ across tenants is composable with these policies — WFQ between tenants, in-tenant policy is FCFS / Priority / SJF.
- Topic 15: reasoning-aware serving stretches the long-job problem; SJF logic must use real-time decode-rate estimates rather than `max_tokens`.

## References

- vLLM engine args — https://docs.vllm.ai/en/latest/serving/engine_args.html
- vLLM V1 design — https://docs.vllm.ai/en/latest/contributing/design/v1/
- SGLang scheduler — https://docs.sglang.ai/
