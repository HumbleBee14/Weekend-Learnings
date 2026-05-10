# 11 — Offline Batch Inference

## Files

- `CONCEPTS.md` — why offline is its own optimization regime, the throughput math, Ray Data integration
- `run_batch.py` — runs N prompts through `vllm.LLM().generate()` and reports tok/s and $/Mtok

## Quickstart

```bash
pip install vllm
python run_batch.py --n 1000 --max-tokens 128
```

## Expected output

On an L4 with 7B BF16 + prefix caching:

```
1000 prompts, 128000 output tokens in 24.6s
  agg throughput   5203 tok/s output
  per-prompt avg   24.6 ms
```

Compare to the same 1000 prompts through `vllm serve` at concurrency 8 — likely 3-5× slower wall-clock.

## Try

- **Disable prefix caching** (`enable_prefix_caching=False` in the script). Re-run. The shared-instruction prompts should suddenly be much slower.
- **Bump `max_num_seqs` to 1024.** More concurrency in the engine; throughput should climb until KV cache caps out.
- **Same workload through the OpenAI server.** Wall time difference is the cost of HTTP + per-request overhead.
- **Wire Ray Data**: shard 100K prompts across 4 GPUs, write outputs to parquet, resume on failure. This is the production shape.
- **$/Mtok calculation:** `($/hr) / (out_tok/s) * 1e6 / 3600`. Compare to OpenAI's batch API price for the same model class — sometimes you win, sometimes you don't.

## Where this goes

- Project 2 G9 (cost per million tokens) includes the offline-batch number as a row
- Level 7 — your `mini-platform` may run nightly batch jobs on the same engine pool used for online serving
