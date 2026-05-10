# 03 — SGLang and RadixAttention

## What SGLang is

A fast LLM serving runtime, born at LMSYS / Berkeley (Jan 2024 paper). Same job as vLLM, different scheduler. The differentiator is **RadixAttention** — a radix tree keyed on token sequences for prefix-aware KV reuse.

In 2026, SGLang is the engine behind:
- xAI Grok 3
- Microsoft Azure's DeepSeek R1 deployment on AMD
- Workloads totaling 400k+ GPUs in production (per their own blog)

## RadixAttention — the data structure

Two requests share a prefix only if their *token sequences* match exactly. A radix tree (compressed trie) over token sequences finds the longest matching prefix in O(prefix length) and lets the KV blocks for that prefix be shared by both requests.

```
Tree state after three requests with the same system prompt + different user turns:

         [system tokens 0..63]               <- shared by all 3 requests
                  |
         [system tokens 64..127]             <- shared by all 3
                  |
        ┌─────────┼─────────┐
        |         |         |
   [user-A]   [user-B]  [user-C]             <- per-request branches
        |         |         |
   [decode-A] [decode-B] [decode-C]

Each node holds: (token_ids, KV block ids, ref_count, last_access_time)
LRU eviction works on the tree edges that have ref_count == 0.
```

Compare to vLLM's prefix cache: vLLM hashes per-block, looks up by hash. The hash chain encodes the prefix, so the effect on cache hits is similar. The structural difference matters for:

1. **Eviction granularity.** Tree-aware eviction can preserve a long shared prefix while evicting per-branch tails. Hash-based eviction needs more bookkeeping to do the same.
2. **Observability.** A radix tree is directly inspectable — "this node is shared by 17 requests" is one read. Hash buckets need traversal.
3. **Frontend integration.** SGLang's frontend DSL (the `sgl.gen` / `sgl.fork` API) maps cleanly onto the tree — fork creates a branch, gen extends a node.

## When SGLang wins, when it doesn't

```
Workload                              Likely winner
──────────────────────────────────    ─────────────
Chatbot with shared system prompt     SGLang (clean prefix tree win)
Multi-turn conversations              SGLang (turn-by-turn prefix accumulation)
Agentic tool-use loops                SGLang (same tool definitions reused)
RAG with shared retrieval header      SGLang
Tree-of-thought / branching search    SGLang (fork is first-class)
Generic OpenAI-compat completions     vLLM (small edge from C++ router)
Heavy structured output (XGrammar)    SGLang (FSM compilation tighter)
Multi-LoRA serving                    vLLM (Punica kernels more mature)
```

Published numbers from SGLang's own blog: up to 6.4× over vLLM on prefix-heavy benchmarks; ~29% on generic H100 SharedGPT workloads. Treat both as upper bounds — your bake-off (Topic 07) will show smaller gaps.

## The overlap scheduler

SGLang's other architectural idea: **while the GPU runs step N, the CPU prepares step N+1 concurrently.**

```
GPU timeline:    [step N forward         ][step N+1 forward      ]
CPU timeline:    [prep step N+1 inputs  ][prep step N+2 inputs  ]
                  └─ tokenize new reqs
                  └─ assemble sampling params
                  └─ allocate KV slots
                  └─ build block tables
```

The "zero-overhead scheduler" claim. vLLM V1 has converged on similar techniques (the diff-based update + multi-step lookahead in the 2026 roadmap), so this is no longer a clean SGLang-only advantage in 2026.

## Disaggregated serving

SGLang supports prefill/decode disaggregation natively, including KV transport via NIXL or the engine's own RDMA path. Topic 08 covers the architecture; SGLang is one of the production implementations.

## Frontend DSL — when it matters

```python
import sglang as sgl

@sgl.function
def two_step(s, question):
    s += sgl.system("You are a helpful assistant.")
    s += sgl.user(question)
    s += sgl.assistant(sgl.gen("draft", max_tokens=128))
    s += sgl.user("Now make it concise.")
    s += sgl.assistant(sgl.gen("final", max_tokens=64))
```

This isn't just a wrapper. The runtime knows the `system` prefix is shared across calls, so it stays in the radix tree. For agentic workloads with reused system prompts and tool definitions, the DSL cuts effective latency dramatically.

If you only hit the OpenAI-compatible endpoint, you don't get DSL benefits — the runtime falls back to standard prefix caching, which is similar to vLLM's. To benchmark RadixAttention's edge fairly in Topic 07, send the prefix-heavy workload through both engines' OpenAI endpoints with shared system prompts.

## Pitfalls

1. **`--chunked-prefill-size` is batch-wide, not per-request.** Common confusion — see vllm-project/vllm#20018 for the SGLang-vs-vLLM difference.
2. **Python router GIL.** SGLang's router is Python-bound; for very-high-RPS edge cases, vLLM's C++ router wins. Mostly irrelevant under 1000 RPS per replica.
3. **Comparing without prefix-heavy traffic.** Without the workload that exercises RadixAttention, you'll conclude "SGLang and vLLM are basically the same." That conclusion is workload-conditional and your bake-off must say so.
4. **Forgetting the DSL.** If your real workload looks like the DSL pattern (multi-turn, branching, shared tools), benchmarking only the OpenAI endpoint understates SGLang's win.

## What to do this topic

1. `pip install sglang[all]`. Serve the same model as Topic 01 on a different port.
2. Run the prefix-heavy workload (`prefix_workload.py`).
3. Run the same workload against vLLM (Topic 01's setup).
4. Compare TTFT and throughput. The gap on prefix-heavy traffic is the headline.

## References

- SGLang docs — https://docs.sglang.ai/
- SGLang paper (RadixAttention) — https://arxiv.org/abs/2312.07104
- SGLang GitHub — https://github.com/sgl-project/sglang
- vLLM vs SGLang chunked-prefill semantics — https://github.com/vllm-project/vllm/issues/20018
- xAI Grok 3 / SGLang note — https://lmsys.org/blog/
