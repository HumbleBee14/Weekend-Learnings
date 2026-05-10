# 08 — Disaggregated Inference

## The asymmetry that makes this necessary

LLM inference has two phases with completely different bottlenecks:

```
Prefill                                Decode
──────                                 ──────
Compute-bound (GEMM-heavy)             Memory-bandwidth-bound (HBM reads)
Processes the entire prompt at once    Generates one token per step
Saturates SMs                          Stresses HBM bandwidth, leaves SMs idle
Variable cost (∝ prompt length)        Roughly constant cost per step
Low frequency (once per request)       High frequency (once per token)
```

Run prefill and decode on the *same* GPU and you waste one resource or the other every step. A 4K-token prefill briefly pegs the SMs while the decode pool's HBM reads stall. A long decode stretches HBM bandwidth while the SMs idle.

**Disaggregation** = split prefill and decode onto different GPU pools, sized to each phase's bottleneck.

## The architecture

```
                  ┌─────────────────────────┐
   request  ───►  │  Router                 │
                  │  - prefill replicas list│
                  │  - decode replicas list │
                  │  - KV-cache-aware       │
                  │    placement            │
                  └─────────────────────────┘
                       │            │
                       ▼            ▼
            ┌─────────────────┐  ┌──────────────────┐
            │ Prefill workers │  │ Decode workers   │
            │  - large batch  │  │  - many slots    │
            │  - SM-bound     │  │  - HBM-bound     │
            │  - GPUs sized   │  │  - GPUs sized    │
            │    for compute  │  │    for KV memory │
            └─────────────────┘  └──────────────────┘
                       │            ▲
                       │  KV transfer
                       └──────────────────┘
                       NIXL / NCCL / RDMA / LMCache
```

After prefill finishes, the prefill worker's KV cache must move to a decode worker. That transfer is the new variable in the system.

## KV transfer mechanisms

Two main strategies in 2026:

```
1. Direct GPU-to-GPU transport (NIXL, NCCL P2P, GPUDirect RDMA)
   ──────────────────────────────────────────────────────────────
   - Lowest latency
   - Tightly couples placement: prefill[i] → decode[j] is a fixed pair
   - Falls over if a decode worker dies mid-transfer
   - Best when KV cache fits and requests don't migrate

2. KV cache as cluster storage (LMCache, Mooncake, CMX/BlueField)
   ──────────────────────────────────────────────────────────────
   - Slightly higher latency (extra hop through cache layer)
   - Decoupled placement: any decode worker can pull any KV
   - Survives worker failures (KV is in the cache layer)
   - Enables prefix-cache reuse across requests on different workers
   - Mooncake joined PyTorch Ecosystem in Feb 2026 — mainstream now
```

The block-hash kv-connector standard (Level 4 Topic 11) is what makes the second strategy interoperable across engines. vLLM, SGLang, LMCache all agree on a block-hash format so a KV block written by vLLM-prefill can be read by SGLang-decode.

## When disaggregation helps

```
Workload                                            Disagg helps?
──────────────────────────────                      ─────────────
High QPS, long prompts (RAG, code review)           Strongly yes
Many concurrent decodes, short prompts              Marginal
Low QPS / batch-like                                No (overhead > savings)
Heterogeneous GPU fleet (H100 + L40)                Yes — prefill on H100, decode on L40
Strict TTFT SLA + relaxed throughput                Yes — prefill pool sized for TTFT
```

The dirty truth: disagg adds operational complexity (router logic, KV transfer plumbing, two replica pools to autoscale). For low-QPS deployments it's overhead. The 2026 production frontier runs disagg because their QPS justifies it.

## What "almost every production framework" supports it

> Almost every production-grade LLM serving framework — NVIDIA Dynamo, llm-d, Ray Serve LLM, SGLang, vLLM, LMCache, MoonCake — runs on disaggregation.
> — Hao AI Lab DistServe retrospective (https://hao-ai-lab.github.io/blogs/distserve-retro/)

The remaining work is the orchestration *above* the engines: routing, autoscaling per pool, KV-cache-aware placement. That's what NVIDIA Dynamo and llm-d do (Topic 09).

## The KV-cache-aware router

A naive router sends each request to the least-loaded worker. A KV-cache-aware router sends each request to the worker whose **prefix cache already has the request's prompt prefix**. The hit means zero KV recompute and a TTFT win. The miss means business as usual.

```
incoming prompt = [system_tokens, user_tokens]
hash chain = h0=hash(empty || system_tokens[0..15]),
             h1=hash(h0 || system_tokens[16..31]),
             ...
router checks: which worker has the longest matching hash chain in cache?
              route to that worker — its prefix cache already has the prefix
```

This requires every worker to expose its prefix-cache state to the router. vLLM emits cache events; LMCache exposes cluster-wide state; Dynamo does the aggregation in its KV router.

## Pitfalls

1. **Disagg without enough QPS.** Below ~10-50 RPS per replica, the transfer overhead eats the gains. Measure.
2. **Forgetting failure modes.** Decode worker dies mid-stream — the request needs to be retried or migrated. NIXL/NCCL pairs are tighter; LMCache makes failover cleaner.
3. **No KV-cache-aware routing.** Disagg without it leaves prefix-cache wins on the table. The router is half the value.
4. **Treating KV transfer as free.** A 32K-token KV cache on a 7B is hundreds of MB. Even on NVLink, that's a few ms; on RDMA across a leaf-spine, more. Budget it.
5. **Per-engine KV format mismatch.** vLLM and SGLang both support disagg, but their KV layouts differ. Cross-engine disagg requires the block-hash kv-connector standard, which is still firming up.

## What to do this topic (conceptual)

You won't run a real disaggregated cluster this week — too much infra. Instead:

1. Read [llm-d's prefill/decode disaggregation guide](https://llm-d.ai/docs/guide/Installation/pd-disaggregation).
2. Read [Dynamo's KV-cache-aware routing docs](https://docs.nvidia.com/dynamo/latest/user-guides/kv-cache-aware-routing.html).
3. Read the [Hao AI Lab DistServe retrospective](https://hao-ai-lab.github.io/blogs/distserve-retro/) — it's the best single piece on what worked and didn't.
4. Run `simulate_disagg.py` (this folder) — a single-process toy that exposes the round-trip and KV-transfer overhead in a way you can reason about.

The transfer mechanics — NIXL, RDMA, GPUDirect — are systems primitives covered in **Level 6's NCCL + RDMA topic**. This topic is *when* and *what to transfer*; that one is *how the bytes move*.

## References

- DistServe retrospective (the canonical 2026 read) — https://hao-ai-lab.github.io/blogs/distserve-retro/
- DistServe paper (OSDI '24) — https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin
- Splitwise (Microsoft, 2024) — https://www.microsoft.com/en-us/research/publication/splitwise-efficient-generative-llm-inference-using-phase-splitting/
- Mooncake architecture — https://github.com/kvcache-ai/Mooncake
- Mooncake → PyTorch Ecosystem (Feb 2026) — https://pytorch.org/blog/mooncake-joins-pytorch-ecosystem/
- LMCache — https://docs.lmcache.ai/
- NIXL — https://github.com/ai-dynamo/nixl
- llm-d disaggregation — https://llm-d.ai/docs/guide/Installation/pd-disaggregation
- Dynamo KV-cache-aware routing — https://docs.nvidia.com/dynamo/latest/user-guides/kv-cache-aware-routing.html
- vLLM cache salting RFC #16016 — https://github.com/vllm-project/vllm/issues/16016
