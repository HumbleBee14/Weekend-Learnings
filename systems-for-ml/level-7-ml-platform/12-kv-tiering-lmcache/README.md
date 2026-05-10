# 12 — KV Tiering (LMCache) + Cross-Replica Coherence

## Files

- `CONCEPTS.md` — LMCache four-tier model, block-hash kv-connector standard, the four cross-replica coherence strategies (sticky / pull-on-demand via NIXL / write-through / replicated), why KV coherence is easier than DB coherence, worked byte-path example.
- `lmcache-vllm.yaml` — vLLM Deployment with LMCache enabled, DRAM + NVMe + Redis tiers, plus a Redis service for the remote tier.
- `long_doc_qa_bench.py` — drives a same-prefix-different-question workload to surface tier hit rates.

## Quickstart

```bash
# K8s path (with the rest of the platform up):
kubectl apply -f lmcache-vllm.yaml

# Local Docker path:
pip install lmcache vllm
LMCACHE_LOCAL_CPU=True LMCACHE_MAX_LOCAL_CPU_SIZE=20 \
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'

# Drive the workload:
python long_doc_qa_bench.py --base http://localhost:8000 \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --doc-tokens 16000 --questions 20
```

## Expected output

```
Q 0  TTFT=  6800ms   'What is paged KV cache?'
Q 1  TTFT=   420ms   'Explain continuous batching.'
Q 2  TTFT=   380ms   'How does prefix caching work?'
...
cold (first): 6800 ms
warm (median rest): 400 ms
speedup: 17.0x
```

The exact ratio depends on hardware and document length. Bigger doc = bigger relative speedup, because the prefill saved is larger.

## Try

- **Tier breakdown.** Restart the pod between runs to drop HBM. Confirm the second run is "warm-from-DRAM" (slower than HBM-warm but much faster than cold). Repeat after `rm -rf /var/lib/lmcache/disk` to drop NVMe; final fallback is the Redis remote tier.
- **Two replicas, no sharing.** Comment out `LMCACHE_REMOTE_URL`. Run the workload through Topic 06's router with two replicas. The replica that didn't see the first request still pays cold prefill.
- **Two replicas, write-through.** Re-add the Redis URL. Re-run. The second replica's cold-hit cost should drop dramatically — it reads blocks from Redis.
- **Pull-on-demand.** Pair LMCache with NIXL between replicas (config in llm-d). Compare RDMA-pull latency to Redis fetch.
- **Salting.** Add a per-tenant salt (Topic 07). Confirm tenant-A's blocks are unreachable to tenant-B even with identical prompts.

## Build steps 5-7 (from CONCEPTS.md)

5. Read llm-d's [KV-cache-aware routing + transfer architecture](https://developers.redhat.com/articles/2025/10/07/master-kv-cache-aware-routing-llm-d-efficient-ai-inference). It uses pull-on-demand via NIXL for hot prefixes plus write-through for durability — a hybrid of strategies 2 and 3.
6. Read [LMCache's architecture page](https://docs.lmcache.ai/developer_guide/architecture.html). Block hashes live in a metadata index in DRAM; block contents tier through HBM / DRAM / NVMe / remote.
7. Sketch the byte path on paper for the 4-replica + hot-prefix scenario in `CONCEPTS.md`. This is the strongest comprehension test for this topic.

## Where this goes

- Topic 06: PrefixStore decisions improve when augmented with which tier each block lives in (HBM > DRAM > NVMe > remote).
- Topic 13: $/Mtok includes KV tier costs (Redis instance, NVMe), and benefits include reduced prefill GPU-seconds.
- Topic 15: long reasoning outputs make per-replica KV pressure worse; tiering becomes mandatory, not optional.

## References

- LMCache architecture — https://docs.lmcache.ai/developer_guide/architecture.html
- llm-d KV-aware routing & transfer — https://developers.redhat.com/articles/2025/10/07/master-kv-cache-aware-routing-llm-d-efficient-ai-inference
- NIXL — https://github.com/ai-dynamo/nixl
- Mooncake — https://github.com/kvcache-ai/Mooncake
