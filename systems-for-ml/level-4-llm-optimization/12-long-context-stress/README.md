# 12 — Long-Context Stress

## Files

- `CONCEPTS.md` — what changes at 32K-128K-1M context, the 2026 production recipe (chunked prefill + pipeline parallelism + MLA), LMCache + Mooncake offload, MLA architectural compression

## What you do this topic

Stress-test the paged KV from Topics 10-11 on a long-context workload. Add chunked prefill. Measure deltas.

## Quickstart

```bash
# Compare three cache implementations on a 32K-token request
python ../09-kv-cache-naive/naive_kv_cache.py    # will OOM or extremely slow at 32K
python ../10-kv-cache-paged/paged_kv_cache.py    # works but slow TTFT (no chunking)

# Add chunked prefill to your mini-vllm; re-test
# (no canonical script here — this topic is mostly about extending what you have)
```

## What to test

1. **Naive cache, 32K tokens**: see it crash or take forever
2. **Paged cache, 32K tokens, naive prefill**: works but TTFT is 10s+
3. **Paged cache, 32K tokens, chunked prefill (4K chunks)**: TTFT drops to ~3s, decode for other requests stays fast
4. **Paged + chunked + 4 concurrent 32K requests**: stress the block pool. Confirm OOM avoidance (eviction or backpressure).
5. **vLLM serving the same 32K workload**: production reference. Match the shape, not the absolute numbers.

## Try

- **Run vLLM with `--max-model-len 32768`** and chunked prefill on. Measure TTFT and total throughput. Compare to your `mini-vllm`.
- **FP8 KV cache** — vLLM supports it via `--kv-cache-dtype fp8`. KV memory halves; quality should drop <0.5% on most tasks.
- **Try MLA-architecture model** — DeepSeek-V3 (or a derivative). KV cache memory is dramatically smaller despite the model being huge.
- **Read [LMCache architecture](https://docs.lmcache.ai/developer_guide/architecture.html)** and identify which tier each piece of metadata lives in.
- **Read [SGLang's chunked-PP blog](https://www.lmsys.org/blog/2026-01-15-chunked-pipeline/)** — 3.31× prefill throughput at 128K.

## What you should walk away with

- `mini-vllm` that handles 32K+ context without crashing
- Chunked prefill implemented and measured
- A clear sense of when the curriculum's `mini-vllm` runs out vs when production engines need LMCache, Mooncake, or MLA
- Awareness that MLA is architectural KV compression — orthogonal to quantization, not a replacement for it

## Where this goes

The KV-cache sub-arc closes here. Topics 13, 17 add speculative decoding. Topic 14 adds continuous batching. Topic 15 adds structured output. Topic 16 covers the concurrency primitives.

After all of those, `mini-vllm` is recognizably a real serving engine. Project 1 — Level 4's deliverable — is "mini-vllm with paged KV + continuous batching, benchmarked against the Level 1 baseline and the production engines."
