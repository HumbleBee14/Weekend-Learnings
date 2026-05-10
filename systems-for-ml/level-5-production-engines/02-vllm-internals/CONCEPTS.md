# 02 — vLLM Internals

The point of this topic is *reading*. You walk through the actual vLLM source, mapping each component to what your `mini-vllm` already does (or punts on).

Required reading: [Inside vLLM: anatomy of a high-throughput LLM inference system](https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html) — the official Sept 2025 deep-dive. Single best document on serving-engine internals.

## The V1 process model

```
┌────────────────────────────────────────────────────────────────┐
│ Client process (your code)                                     │
└────────────────────────────────────────────────────────────────┘
                       │  HTTP / OpenAI client
                       ▼
┌────────────────────────────────────────────────────────────────┐
│ AsyncLLM (front-end Python process)                            │
│  vllm/v1/engine/async_llm.py                                   │
│  - HTTP routes (OpenAI-compatible)                             │
│  - tokenization, sampling-param parsing                        │
│  - SSE streaming                                               │
│  - request lifecycle on the front-end side                     │
└────────────────────────────────────────────────────────────────┘
                       │  ZMQ over /dev/shm
                       ▼
┌────────────────────────────────────────────────────────────────┐
│ EngineCore (back-end process)                                  │
│  vllm/v1/engine/core.py                                        │
│  - Scheduler (token-budget, mixes prefill and decode)          │
│  - Block manager (paged KV, prefix cache, free list)           │
│  - KVConnector (LMCache, NIXL — for disaggregated/offload)     │
│  - Driver-rank Worker for TP/PP coordination                   │
└────────────────────────────────────────────────────────────────┘
                       │  shared-memory + IPC
                       ▼
┌────────────────────────────────────────────────────────────────┐
│ Worker(s) — one per GPU rank                                   │
│  vllm/v1/worker/gpu_worker.py                                  │
│  vllm/v1/worker/gpu_model_runner.py                            │
│  - Model executor (forward pass, piecewise CUDA graphs)        │
│  - FlashInfer / FlashAttention attention                       │
│  - Sampler (top-k/p/temp, xgrammar, repetition)                │
│  - KV-cache GPU tensors live here                              │
└────────────────────────────────────────────────────────────────┘
```

The **front-end / back-end split** is the single most important architectural fact. In V0, scheduling and HTTP handling shared a Python process; the GIL made every step pay HTTP-handler overhead. V1 splits them. The back-end runs in a tight C++/CUDA-flavored loop; the front-end can be slow without hurting throughput.

## The scheduler — token budget, not request count

V1's scheduler operates on a **token budget per step**:

```
budget = max_num_batched_tokens          # default ~2048 tokens
for each step:
    consume_from_running_decodes()       # 1 token per active decode
    consume_from_chunked_prefills()      # up to budget remainder
    if budget remains and queue not empty:
        admit new requests (prefill chunk fills the rest)
```

This is why "chunked prefill" stopped being a flag and became the architecture. Mixing prefill and decode in the same step keeps the GPU saturated.

What you read in the source:

- `vllm/v1/core/sched/scheduler.py` — the loop above
- `vllm/v1/core/sched/output.py` — `SchedulerOutput` (the diff sent to workers)
- `vllm/v1/core/kv_cache_manager.py` — block allocator + prefix cache hashing

## Block manager and prefix cache

```
                  ┌──────────────────────────────────────┐
                  │ KVCacheManager                       │
                  │  - block_pool: free-list of block_ids│
                  │  - block_table[req_id] = [b0,b1,...] │
                  │  - prefix_cache: hash → block_id     │
                  └──────────────────────────────────────┘

prefix cache hash is per-block:
  hash(block_i) = sha256(parent_hash || token_ids_in_block || extra_keys)
```

`extra_keys` is what enables multimodal hashing (image bytes for VLMs — Topic 14) and per-tenant isolation via cache salting (RFC #16016).

In 2026 this hash format is becoming the **block-hash kv-connector standard**: vLLM ↔ LMCache ↔ Mooncake all agree on the hashing so KV blocks can move between engines.

Source: `vllm/v1/core/block_pool.py`, `vllm/v1/core/kv_cache_utils.py`.

## The model runner and FlashInfer

The actual forward pass lives in `gpu_model_runner.py`. What it does each step:

1. Take the `SchedulerOutput` (which requests, which positions, which blocks)
2. Build the input tensors (token ids, positions, block tables for attention)
3. Run the model (piecewise CUDA graphs — Level 4 Topic 07)
4. Sample (top-k, top-p, temperature, xgrammar masking, repetition penalty)
5. Return new tokens + KV updates back to scheduler

Attention is a FlashInfer kernel call. The block table is passed as a tensor; FlashInfer gathers K/V from non-contiguous physical blocks. You don't write that kernel; nobody does anymore.

Source: `vllm/v1/worker/gpu_model_runner.py`, `vllm/v1/attention/backends/flashinfer.py`.

## CUDA graphs — piecewise capture

Why piecewise: a full-graph capture would require fixed shapes; LLM serving has dynamic shapes (variable batch, variable seq lengths). The compromise:

```
per shape bucket [batch_size]:
    capture a CUDA graph for the *static portions* of the forward pass
    leave attention (dynamic-shape) outside the graph
```

The model is split into "graph-able" pieces and a few op runs that change shape. Replay the graphs, run the dynamic ops eagerly.

Result: kernel-launch overhead disappears for the bulk of the model. ~1.5-2× speedup on small models where launch overhead dominated.

V1 captures these automatically. `--enforce-eager` disables capture (debugging only).

## Sampler

`vllm/v1/sample/sampler.py` and `vllm/v1/sample/ops/`. Implements:

- Greedy / multinomial / beam (mostly removed in V1)
- Top-k, top-p, min-p
- Temperature, repetition penalty, frequency / presence penalties
- Logit bias
- Structured output via xgrammar (Level 4 Topic 15)

The sampler runs on GPU. Earlier versions did sampling on CPU and paid a sync cost.

## What your `mini-vllm` skipped

If you built `mini-vllm` through Level 4, list three things it punts on that vLLM does:

1. **Piecewise CUDA-graph capture.** You ran eager.
2. **Front-end / back-end process split.** You're single-process, GIL-bound.
3. **SHA-256 block hashing for prefix cache.** You probably used a simpler hash, didn't handle the parent-chain dependency, didn't support `extra_keys`.

Honest: there are at least ten more (multi-LoRA punica kernels, structured-output FSM compilation, NCCL TP plumbing, KV-connector for disagg, EAGLE-3 spec head, NVFP4 quant kernels). The point isn't to feel bad — it's to know exactly what each flag corresponds to in the source.

## What to actually do this topic

1. Clone vLLM. Open the four files referenced above in your editor side-by-side with your `mini-vllm` equivalents.
2. Read the anatomy blog start to finish.
3. Sketch a one-page diagram of the request lifecycle: HTTP arrives → AsyncLLM → IPC → EngineCore.schedule() → worker.execute_model() → sample → result back to AsyncLLM → SSE chunk to client. This is what `walk_lifecycle.py` lays out.

## References

- Anatomy blog (required) — https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html
- Life of an inference request (Ubicloud, V1-focused) — https://www.ubicloud.com/blog/life-of-an-inference-request-vllm-v1
- vLLM source — https://github.com/vllm-project/vllm
- V1 design RFC — https://docs.vllm.ai/en/stable/design/v1/
- vLLM cache salting RFC #16016 — https://github.com/vllm-project/vllm/issues/16016
- FlashInfer — https://github.com/flashinfer-ai/flashinfer
