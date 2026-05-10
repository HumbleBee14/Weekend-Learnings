# 06 — Inference Routing (KV-cache-aware)

## Files

- `CONCEPTS.md` — L4/L7/ext-proc/sidecar, the five-step KV-aware routing algorithm, hot-spot risk and load blending, llm-d EPP, Iris (Rust), SGLang RadixAttention.
- `router.py` — Python FastAPI router. Block-hash chain (SHA-256, 16-token blocks), PrefixStore, multi-objective scorer, refresh loop polling pod `/kv-blocks`.
- `bench.py` — chatbot-shaped workload generator. Drives the router with shared prefix + varied suffix; reports TTFT p50/p95/p99.

## Quickstart

```bash
# 1. Bring up two vLLM workers (any model, same model both pods):
vllm serve meta-llama/Llama-3.2-1B-Instruct --port 8001 &
vllm serve meta-llama/Llama-3.2-1B-Instruct --port 8002 &

# 2. Add a /kv-blocks shim per pod (vLLM does not yet expose this on the OpenAI port).
#    Easiest path: a small sidecar that subscribes to vLLM's kv-events stream and
#    serves the current block-hash set. For local experiments, mock this endpoint.

# 3. Start the router:
python router.py --pods http://localhost:8001 http://localhost:8002 --policy prefix

# 4. Run the bench, twice:
python bench.py --policy prefix --requests 200 --shared-prefix 4096
# Restart router with --policy random and re-run for comparison.
```

## Expected output

On a workload with a 4KB shared system prompt and varied suffix:

```
policy=random
TTFT p50: ~ X ms
TTFT p99: ~ Y ms

policy=prefix
TTFT p50: ~ X/3 ms
TTFT p99: ~ Y/4 ms
```

The exact ratio depends on model and hardware — what matters is the *direction* and that p99 collapses harder than p50 (the long-prefix requests benefit most). On a no-shared-prefix workload, the two policies should be within noise.

## Try

- **Hot-spot demo.** Set `w_p = 1.0, w_l = 0`. Send the same prefix from many concurrent users. Watch one pod saturate while the other idles. Restore default weights.
- **Index staleness.** Crank `period_s` to 30s in `refresh_loop`. Watch routing decisions degrade as the PrefixStore lags reality.
- **Cache salting.** Add a `tenant_id` mix-in to the first block's hash input. Confirm that two tenants with identical prompts now route independently — the foundation of multi-tenant prefix isolation.
- **SGLang radix.** Replace `PrefixStore` with a radix tree over token sequences. Compare lookup time on long prompts.

## Where this goes

- Topic 07: WFQ admission policy lives next to `router.pick` — call it before forwarding upstream.
- Topic 08: `pod.inflight` and observed latency feed Little's Law validation.
- Topic 12: cross-replica KV coherence — when a pod is picked but doesn't actually hold the prefix, NIXL-pull from the holder. Same data structure (PrefixStore), different downstream action.
- Topic 15: cancellation propagation closes the upstream `httpx.AsyncClient.stream` on client disconnect.

## References

- vLLM Production Stack KV-aware tutorial — https://docs.vllm.ai/projects/production-stack/en/latest/tutorials/kvaware.html
- llm-d KV-aware routing — https://developers.redhat.com/articles/2025/10/07/master-kv-cache-aware-routing-llm-d-efficient-ai-inference
- vLLM Semantic Router (Iris) — https://github.com/vllm-project/semantic-router
