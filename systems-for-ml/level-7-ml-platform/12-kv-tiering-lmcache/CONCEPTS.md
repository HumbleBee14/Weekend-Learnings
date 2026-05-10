# 12 — KV Tiering (LMCache) + Cross-Replica Coherence

## Why KV tiering exists in 2026

128K and 1M context windows are standard. Prefilling 128K every cold request is unaffordable: TTFT grows linearly with prompt length, and the GPU spends seconds-to-minutes computing K/V it has computed before. **LMCache** addresses this by adding a multi-tier KV cache *behind* vLLM's HBM-resident cache.

```
GPU HBM (engine native)
   ↓ async push on eviction
CPU DRAM (pinned, hot tier)            ~1 GB/s read, 100 GB capacity
   ↓ async push when DRAM full
Local NVMe (warm tier)                 ~5 GB/s read, TBs of capacity
   ↓ async push when NVMe full
Remote backend                         ~1-10 GB/s over RDMA, persistent
   (Redis | Mooncake | InfiniStore | Ceph)
```

Reported numbers (LMCache + vLLM, 128K system prompt on H100):
- TTFT cold: 11s
- TTFT warm (block hit in DRAM): 1.5s
- Multi-turn QA / doc analysis throughput: up to 15x

LMCache is the canonical KV-tier offload — default in vLLM Production Stack and llm-d.

References:
- LMCache — https://docs.lmcache.ai/developer_guide/architecture.html
- LMCache GitHub — https://github.com/LMCache/LMCache

## Block hashing and the kv-connector standard

LMCache uses the same **block-hash** identifiers as vLLM's prefix cache (Topic 06): SHA-256 chained over 16-token blocks. This isn't an accident — it makes the **block-hash kv-connector** a cross-engine standard. vLLM, LMCache, NIXL, Mooncake, and llm-d all speak the same block-hash language; a block computed on engine A is addressable from engine B.

This is the boring-but-load-bearing 2026 standardisation. Without it, every KV transfer needed format translation and the disaggregated/tiered designs would be fragile.

References:
- NIXL — https://github.com/ai-dynamo/nixl

## Cross-replica KV coherence — the four strategies

Once you have multiple vLLM workers behind a router (Topic 06), the same prefix exists in *one* worker's HBM but might be requested through *any* worker. Topic 06's KV-aware router *steers* requests to where the prefix already lives. But what happens when:

- The prefix-holding pod is overloaded?
- Two requests arrive nearly simultaneously, each picks a different pod?
- A new pod just spun up and holds nothing?

Four standard strategies. Pick by your throughput / latency / capex tradeoff.

### Strategy 1 — Pure prefix-aware routing

Single owner per prefix. The router always sends a prefix-matching request to its owner. Other replicas don't have it; they pay the prefill cost when first asked.

```
prefix P -> pod A (owner)
all P-requests -> pod A
```

- Pros: simplest, no cross-replica IO.
- Cons: hot-key skew kills the owner. Useful when prefix distribution is roughly uniform.

### Strategy 2 — Pull on demand via NIXL

Prefix lives on pod A's HBM. Pod B receives a request matching that prefix. Pod B issues a NIXL pull from pod A's HBM (over RDMA / NVLink) before starting decode. Latency cost: one network round trip — still cheaper than re-prefilling 128K.

```
P-request -> pod B (overloaded routing decided B over A)
pod B: NIXL.pull(P, from=pod A)        ~10-50ms over RDMA for 100s of MB
pod B: decode
```

- Pros: load-balances hot prefixes; preserves cache reuse.
- Cons: requires RDMA/NVLink fabric; both pods pay coordination cost.

This is the same machinery as Level 6's RDMA + GPUDirect work and Level 5's disaggregated inference. Different access pattern, identical primitives.

References:
- NIXL — https://github.com/ai-dynamo/nixl
- llm-d KV transfer architecture — https://developers.redhat.com/articles/2025/10/07/master-kv-cache-aware-routing-llm-d-efficient-ai-inference

### Strategy 3 — Write-through to LMCache backend

Every block, on eviction from HBM, writes through to a shared backend (Redis, Mooncake, InfiniStore, Ceph). Any worker can read it later from the shared tier.

```
pod A: prefill P -> HBM
pod A: HBM evict -> Redis  (the canonical, addressable copy)
pod B: P-request arrives
pod B: read Redis -> HBM
```

- Pros: cleanest cluster-wide view; tolerates pod death (KV survives).
- Cons: write amplification, storage cost, tail latency on tier-miss reads.

**Mooncake** (Moonshot AI) is the production reference. Joined PyTorch Ecosystem Feb 2026. KV-cache disaggregation as a service, RDMA-backed, designed for cluster-wide coherence.

References:
- Mooncake — https://github.com/kvcache-ai/Mooncake
- LMCache architecture — https://docs.lmcache.ai/developer_guide/architecture.html

### Strategy 4 — Replicated tier (Mooncake-style hot-block replication)

Hot blocks proactively replicated to N workers. Trades capacity for parallelism on hot prefixes.

```
P is hot                ->  replicated to pods A, B, C
P-requests              ->  load-balanced across A, B, C without cross-pod IO
```

- Pros: best for hot-prefix workloads (RAG with shared system prompts).
- Cons: capacity overhead; replica freshness vs eviction policy gets subtle.

## Coherence semantics — why this is easier than database coherence

KV cache is **append-only and immutable** once a token is computed. The K/V at position P always represents tokens 0..P; it doesn't change. Therefore:

- **No invalidations needed.** A block hash uniquely identifies its content.
- **No version vectors.** Different replicas reading the same hash get the same content.
- **No two-phase writes.** Eviction is a hint, not a state change.

What's *not* trivial:

- **Transfer latency.** A pull from another HBM still costs round-trip + bandwidth. The router must decide whether pull-or-prefill is cheaper.
- **Garbage collection.** Hot blocks evict from HBM, then DRAM, then NVMe, then remote. Eviction policies cascade; simple LRU at each tier works in practice.
- **Capacity.** Across all tiers, capacity is enormous but not unlimited. Per-block accounting matters — Topic 04's eval gate equivalent for KV.

## Per-block byte path (worked example)

Setup: 4-replica cluster, hot shared system prompt (4KB ≈ 256 tokens ≈ 16 blocks at 16 tok/block). Strategy: pull-on-demand (Strategy 2) with write-through to Redis (Strategy 3) layered as the slow-tier fallback.

A request matching the system prompt arrives at pod B (which doesn't hold it):

```
1. Router computes block hashes b0..b15. PrefixStore says pod A holds them.
   (Capacity-aware scoring picked B anyway because A is at 95% load.)

2. Pod B looks up b0..b15 locally. HBM miss. DRAM miss. NVMe miss.

3. Pod B issues NIXL.pull(b0..b15, from=pod A).
   - 16 blocks × 2 layers (K, V) × ~64KB/block = ~2 MB
   - over RDMA: ~0.5ms transfer + ~50us setup = ~1ms
   - alternative re-prefill: ~50-200ms for 256 tokens. NIXL wins.

4. Pod B installs blocks in HBM, starts decode of the user's suffix.

5. Eviction (later): when HBM pressure rises, pod B evicts to its own DRAM tier
   (LMCache layer), and on DRAM pressure, writes through to Redis.

6. Pod C, much later, gets a same-prefix request: PrefixStore now lists A and B
   as holders. Pod C either NIXL-pulls from the closer one, or reads from Redis
   if both A and B have evicted.
```

This is the byte path you should be able to draw without reference. **Build step 7** in the README is exactly this exercise.

## Build steps

1. Enable LMCache on a vLLM Deployment (`--kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'`).
2. Run a long-doc Q&A workload — same document prefix, varied questions.
3. Measure TTFT cold (first question) vs warm (subsequent questions hitting the cached prefix).
4. Document the tier where each block lives during the run (HBM / DRAM / NVMe / remote).
5. Read llm-d's KV-aware routing + transfer architecture page. Identify which of Strategies 1-4 it uses.
6. Read LMCache's architecture page. Annotate which tier each piece of metadata lives in (block hashes vs blocks themselves).
7. Sketch the byte path for a single KV-block read in a 4-replica setup with a hot shared system prompt. Where does it live? How does worker B get it when its request arrives?

## Pitfalls

1. **Tier-mismatch surprise.** A "warm" tenant on NVMe tier still has 1-100x DRAM latency. KV cache hit ≠ uniform speedup; the speedup is the tier you actually hit.
2. **Forgetting cache salting.** Multi-tenant cross-prefix matches across tenants are subtle leaks (Topic 07).
3. **Pure prefix-aware routing in production.** Hot-key skew kills it. Always blend with load.
4. **No eviction telemetry.** Without `vllm:cpu_cache_usage_perc` and analogous LMCache metrics, you can't tell if the warm-tier hit rate is what you think.
5. **Mooncake / Redis as a SPOF.** The remote tier is a real distributed system with availability concerns. Treat it accordingly.
6. **Confusing tiers with replicas.** A tier is *vertical* (HBM->DRAM->NVMe->remote on one node, then cluster-wide). A replica is *horizontal* (different pods). Both reduce prefill cost; they aren't interchangeable.

## References

- LMCache architecture — https://docs.lmcache.ai/developer_guide/architecture.html
- LMCache GitHub — https://github.com/LMCache/LMCache
- llm-d KV-aware routing & transfer — https://developers.redhat.com/articles/2025/10/07/master-kv-cache-aware-routing-llm-d-efficient-ai-inference
- NIXL — https://github.com/ai-dynamo/nixl
- Mooncake — https://github.com/kvcache-ai/Mooncake
- vLLM kv-connector RFCs — https://github.com/vllm-project/vllm/issues
