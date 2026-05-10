# 03 — SGLang and RadixAttention

## Files

- `CONCEPTS.md` — radix-tree prefix sharing, when SGLang wins / loses, the overlap scheduler, the frontend DSL
- `prefix_workload.py` — prefix-heavy chatbot workload pointed at any OpenAI-compatible endpoint; run it twice (once at SGLang, once at vLLM) and compare

## Quickstart

```bash
pip install "sglang[all]" openai

# server
python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-7B-Instruct --port 8001

# in another shell, also have vLLM up on 8000 (Topic 01)
python prefix_workload.py --base-url http://localhost:8001/v1 --label sglang
python prefix_workload.py --base-url http://localhost:8000/v1 --label vllm
```

## Expected output (rough shape, real numbers depend on hardware)

```
[sglang] base_url=http://localhost:8001/v1
  TTFT  p50/p95/p99  85 / 140 / 180  ms
  agg throughput     2200 tok/s

[vllm] base_url=http://localhost:8000/v1
  TTFT  p50/p95/p99  120 / 220 / 280  ms
  agg throughput     1800 tok/s
```

The TTFT delta on prefix-heavy traffic is the headline. On a generic completions workload (no shared prefix) you'll see the two engines within a few percent of each other.

## Try

- **Run with `--n 200 --concurrency 16`** — more requests amortize the radix tree's lookup cost; SGLang's lead grows.
- **Replace the system prompt with random tokens per request** — kills prefix caching; SGLang and vLLM should converge to within a few percent.
- **Use the SGLang DSL directly** instead of OpenAI-compat. The DSL lets the runtime see `fork` / shared-prefix semantics that the OpenAI shape hides.
- **Inspect the cache.** SGLang exposes `/get_server_info` with cache stats; vLLM exposes `/metrics` with `vllm:gpu_prefix_cache_hits_total`. Confirm hits climb after the first request.

## Where this goes

- Topic 07 (the bake-off) reuses this exact prefix-heavy workload as one of the four scenarios.
- Topic 08 — both engines support disaggregated serving; the choice between them in disagg mode is dominated by which KV connector you target.
