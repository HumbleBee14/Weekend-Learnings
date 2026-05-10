# 06 — Inference Routing (KV-cache-aware)

## The 2026 single-biggest delta vs naive serving

Random round-robin -> KV-cache-aware routing changes TTFT by 3-10x on prefix-heavy workloads (chatbots with shared system prompts, RAG, multi-turn). It is the most impactful single piece of platform code in this level.

The reason is mechanical. A vLLM worker that has already prefilled "You are a helpful assistant. The user's name is..." (4KB shared system prompt) holds those KV blocks. If the next request hits the *same* worker, prefill on those tokens is a cache hit — TTFT is set by the suffix only. If it hits a *different* worker, prefill recomputes the full prefix.

## L4 vs L7 vs ext-proc vs sidecar

Four places a router can live. They differ in what they can see.

| Layer | Sees | Cannot see | Examples |
|---|---|---|---|
| L4 (TCP) | source IP, dest port, conn count | request body | classic cloud LB, kube-proxy |
| L7 (HTTP) | path, headers, body | model semantics unless parsed | nginx, HAProxy, Envoy |
| L7 + ext-proc | full request body, can mutate, can call out to scheduler | nothing | Envoy + Endpoint Picker (llm-d, GKE Inference Gateway) |
| Sidecar | same as ext-proc, but in-process | extra network hop | vLLM Production Stack router |

For LLMs you need at least L7 because the routing decision depends on the *prompt body* (you tokenize and block-hash it). The 2026 default is **L7 + ext-proc** via the Gateway API Inference Extension: Envoy as the data plane, an Endpoint Picker (EPP) as the brain. llm-d, GKE Inference Gateway, and Kong AI Gateway all converge on this shape.

References:
- Gateway API Inference Extension — https://gateway-api-inference-extension.sigs.k8s.io/
- llm-d EPP — https://llm-d.ai/docs/architecture
- vLLM Production Stack router — https://github.com/vllm-project/production-stack

## The KV-cache-aware routing algorithm

Five steps. Memorise this; every production router is a variation.

```
1. tokenize(prompt)                                     -> [t0, t1, ..., tN]
2. block_hashes = []
   parent = 0
   for chunk in chunks(tokens, block_size=16):
       parent = SHA256(parent || chunk)                 # vLLM 0.11+ default
       block_hashes.append(parent)
3. for each pod, maintain block_index : Set[hash]       (kv events stream)
4. router PrefixStore: hash -> Set[pod]
   walk block_hashes from index 0, intersect candidate set,
   stop when candidate set empties or intersection becomes empty
   -> longest_prefix_pod
5. score(pod) = w_p * matched_prefix_len
              + w_l * (1 - load_factor)
              + w_t * tokens_remaining_capacity
   pick argmax. tie-break: lower in-flight count.
```

The block size (16) and hash (SHA-256) are vLLM defaults. Both are deliberately conservative:
- Block size 16 -> small enough that partial-prefix matches stay useful, large enough that hash-table overhead doesn't dominate.
- SHA-256 (vs xxHash) -> collision probability negligible enough to act on without verification. The cost is hash CPU; in 2026 routers (Iris in Rust) it's a non-issue.

## Hot-spot risk and mixed scoring

Pure prefix-aware routing creates hot spots: one shared system prompt -> all traffic to one pod. Production routers therefore *blend* prefix score with load score. llm-d's EPP uses a multi-objective scorer; vLLM Production Stack tunes weights via config.

Practical defaults (start here, then tune):

```
w_p = 0.6   prefix length, normalised to max prompt length
w_l = 0.3   load (1 - inflight/capacity)
w_t = 0.1   KV headroom (1 - kv_usage_perc / 100)
```

If `w_p = 1.0`, you'll see one pod at 100% load and the rest idle. If `w_p = 0`, you've reverted to load-balanced random. The interesting territory is in between.

## SGLang RadixAttention — a different data structure

SGLang takes a different angle. Instead of a flat block-hash -> pods map, it maintains a **radix tree of token sequences** with O(prefix-length) match/insert/evict. The router walks the tree to find the longest matching path; eviction works on tree leaves. Same goal (prefix locality), different implementation. RadixAttention is in-engine; the SGLang router rides on top of it.

Reference: https://docs.sglang.ai/

## llm-d's specific architecture

```
client
  │
  ▼
Gateway API + Inference Extension (Envoy)
  │  ext-proc gRPC
  ▼
Endpoint Picker (EPP)
  - reads kv-events stream from each vLLM pod
  - maintains kvblock.Index
  - multi-objective scorer
  │
  ▼
selected vLLM pod
```

The kv-events stream is the load-bearing piece: vLLM publishes block-hash add/evict events; the EPP keeps a near-real-time view. Latency to the EPP's index is sub-millisecond; staleness is bounded by event-stream lag (typically <100ms).

Reference: https://developers.redhat.com/articles/2025/10/07/master-kv-cache-aware-routing-llm-d-efficient-ai-inference

## Iris — the Rust router (vLLM Semantic Router v0.1, Jan 2026)

Rust where Python hurt. The vLLM Semantic Router (codename Iris) does:
- HuggingFace Candle for embedding inference (semantic intent classification).
- Tokio for async request handling.
- Custom prefix-hash logic on stable hashmaps with Tokio mutexes.

Reported numbers vs the Python reference router: +25% throughput, -1200ms TTFT under load. The wins come from no GIL contention in the hot path and no GC pauses. This is the same lesson surfaced by every other latency-critical infra component in the field (Envoy, Pingora, ScyllaDB).

For `mini-platform`, write Python first. It is enough to teach the algorithm and measure the prefix vs random delta. Rewrite in Rust under the `compiler-and-kernels` track when the latency floor matters.

## Sticky sessions vs stateless prefix-aware

Two ways to get prefix locality:

- **Sticky sessions** — bind a session_id to a pod. Cheap, dumb, works for chat. Breaks when a session moves between users or when one session has many simultaneous in-flight prompts.
- **Stateless prefix-aware** — recompute pod selection per request from the actual prompt. Robust, more compute. The 2026 default.

In practice you blend: sticky-by-default with a fallback to prefix-aware if the sticky pod is unhealthy or overloaded.

## Build steps

1. Two vLLM replicas behind a Python FastAPI router.
2. Implement block-level prefix hashing on the router.
3. Each replica reports its current block-hash set on a `/kv-blocks` endpoint (poll every 1s) — a stand-in for vLLM's kv-events stream.
4. Run a chatbot-shaped workload (shared 4KB system prompt, varied suffix). Measure TTFT with random vs prefix-aware.
5. Run a no-prefix-overlap workload — confirm prefix-aware ≈ random there.

## Pitfalls

1. **Hashing tokens, not text.** Tokenizer-version drift destroys hashes. Always block-hash on tokens, with the tokenizer pinned.
2. **No load blend.** Pure prefix routing -> hot spot. Always include load in the score.
3. **Polling each pod's full block set.** O(N_pods × N_blocks) network. Use a kv-events stream (push) or at least a delta endpoint.
4. **Stale index.** A pod evicts a block; router still thinks it's hot. Bound staleness with a short TTL on the index.
5. **Cache salting forgotten.** Multi-tenant deployments need per-tenant salt in the block hash to avoid cross-tenant prefix matches (vLLM RFC #16016). Default off; turn on for multi-tenant.
6. **Routing on raw bytes, not tokens.** Two prompts with identical tokenization but different whitespace will block-hash differently. Tokenize first.

## References

- vLLM Production Stack KV-aware tutorial — https://docs.vllm.ai/projects/production-stack/en/latest/tutorials/kvaware.html
- llm-d KV-aware routing — https://developers.redhat.com/articles/2025/10/07/master-kv-cache-aware-routing-llm-d-efficient-ai-inference
- Gateway API Inference Extension — https://gateway-api-inference-extension.sigs.k8s.io/
- Envoy AI Gateway — https://aigateway.envoyproxy.io/
- vLLM Semantic Router (Iris) — https://github.com/vllm-project/semantic-router
- SGLang — https://docs.sglang.ai/
