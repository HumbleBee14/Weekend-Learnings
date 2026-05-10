# 12 — Long-Context Stress Testing

## What "long context" means in 2026

In 2024: 8K-32K was "long."
In 2025: 128K became standard.
In 2026: 1M-token contexts are mainstream for top-tier models. Practical workloads regularly serve 32K-128K prompts (full codebases, long documents, agentic conversations).

This topic stress-tests `mini-vllm` against long context. Naive cache (Topic 09) collapses immediately. Paged + prefix-shared (Topics 10-11) handles it but introduces new bottlenecks.

## What changes at long context

### 1. KV cache memory dominates

For Qwen2.5-7B at 128K context with batch=1:

```
Model weights (BF16):           14 GB
KV cache (BF16, 32 layers):     ~12 GB
Activations (peak, prefill):    ~8 GB
─────────────────────────────────
Total:                          34 GB
```

KV cache is *almost the size of the model*. At batch=4, KV alone is 48 GB. Many GPUs can't fit even batch=2 at 128K.

### 2. Prefill becomes the bottleneck

Prefill is O(N²) in attention compute (the QK^T matrix is N × N). For N=128K, that's 16 billion attention scores per layer. Without chunked prefill, prefill of a 128K prompt can take 30-60 seconds before the first token is generated.

### 3. Decode stays O(N) per step but cache reads dominate

Each decode step reads the full N-token KV cache. For N=128K, that's 12 GB of HBM traffic per step per request. Memory-bound by miles.

### 4. KV cache compression matters more

At 128K context, FP8 KV cache halves memory. FP4 KV quarters it. Quality cost is small; the alternative is "doesn't fit at all."

## The 2026 long-context production recipe

Three techniques, each tackling one bottleneck:

### Chunked prefill — solves prefill latency

Split the prefill into chunks of 4K-8K tokens. Process them sequentially, interleaving with ongoing decode work for other requests. Decode latency for those other requests stays low.

vLLM has chunked prefill on by default. SGLang exposes `--chunked-prefill-size` (note: this is *batch-wide*, not per-request — common gotcha).

### Pipeline parallelism — solves prefill throughput at scale

When chunked prefill isn't enough, pipeline-parallelize the model across GPUs. Each GPU holds a subset of layers. Tokens stream through.

SGLang's January 2026 blog showed DeepSeek-V3.1 at 128K with pipeline parallelism + chunked prefill: 3.31× prefill throughput improvement, 67.9% TTFT reduction, 82.8% strong scaling. The 2026 long-context production recipe.

### MLA — KV cache architectural compression (not just quantization)

DeepSeek's Multi-head Latent Attention. Instead of storing per-head K and V, store a *compressed latent* of dimension `kv_lora_rank` (~128) and *re-derive* K and V from it during attention.

Result: **93% KV cache reduction** with *better* perplexity than standard MHA.

In 2026: MLA is the architectural KV-compression story. DeepSeek-V3 popularized it; "Enabling MLA in Any Transformer" (arXiv 2502.14837) shows how to retrofit MLA onto existing models post-training.

For the curriculum: MLA is a first-class concept, not a footnote. If you see a model spec with "kv_lora_rank" in the config, that's MLA.

## LMCache + Mooncake — multi-tier KV offload

When the GPU pool is full, where do old KV blocks go?

```
GPU HBM (engine native)
   ↓ async push on eviction
CPU DRAM (pinned, hot tier)
   ↓ async push when DRAM full
Local NVMe (warm tier, large capacity)
   ↓ async push when NVMe full
Remote backend (Redis / Mooncake / Ceph; persistent, slowest)
```

Reported numbers: TTFT 11s → 1.5s for 128K system prompt on H100. ~15× throughput on multi-turn QA / doc analysis.

**LMCache 2026 status**: production. Default chunk size 256 tokens (note: bigger than vLLM's 16-token blocks). Position-independent matching. Supports CPU offload, Redis distributed tier, Mooncake backend.

**Mooncake (Feb 2026 PyTorch Ecosystem)**: started as Kimi's KV cache backend (Moonshot AI). Now mainstream. Backs production at thousands-of-nodes scale, 100B+ tokens/day.

Multimodal angle: SGLang merged Mooncake-powered Encoder Global Cache (Feb 2026) for cross-instance ViT embedding sharing — for VLM serving, this is the equivalent of prefix sharing for images.

These are Level 7 topics in depth; this topic surfaces them as the long-context-fix toolbox.

## What you'll do

Three stress tests:

### Test 1 — Naive cache at long context

Take your Topic 09 naive cache. Try a single 32K-token request. Note the OOM or extreme slowdown.

### Test 2 — Paged cache at long context

Same workload on Topic 10/11's paged cache. Should work but expose new issues:
- KV memory dominates
- Prefill latency is brutal without chunking

Add chunked prefill (process tokens in 4K-token chunks during prefill). Measure TTFT delta.

### Test 3 — Multi-request long context

Five concurrent requests, each with 32K context. Without prefix sharing, you allocate 5 × 32K worth of blocks. With shared system prompt at the start, much less.

Implement chunked prefill from scratch in your `mini-vllm`:

```python
def chunked_prefill(model, request, chunk_size=4096):
    """Process the prompt in chunks, updating KV cache as we go."""
    tokens = request.prompt_tokens
    for chunk_start in range(0, len(tokens), chunk_size):
        chunk = tokens[chunk_start:chunk_start + chunk_size]
        # Forward pass on this chunk; KV cache grows per chunk
        model.forward_with_kv_update(chunk, kv_cache, position=chunk_start)
        # Yield control to scheduler — let other requests' decode steps run
```

This pattern *interleaves* prefill chunks with decode work, hiding prefill latency from other requests' decode-step latency.

### Test 4 — Compare to vLLM

Same workload on vLLM with chunked prefill on. You'll be slower (production-tuned vs hand-rolled) but the *shape* of the curve should match.

## Pitfalls

1. **Mistaking memory pressure for a quality issue.** OOM looks like the model "broke." Check memory metrics first.
2. **Comparing FP16 KV cache to FP8 KV cache without measuring quality.** FP8 KV cache is well-validated for most models, but verify on your domain.
3. **Skipping chunked prefill at 32K+.** TTFT is unusable without it.
4. **Stress testing without a quality harness.** Long-context quality degrades subtly; always run lm-eval-harness with long-context tasks (RULER, LongBench).
5. **Treating MLA as "exotic."** It's standard for any model with `kv_lora_rank` in its config. DeepSeek-V3 is the headline example but several Chinese models adopted it.

## What you should walk away with

- A working `mini-vllm` that handles 32K+ context cleanly
- Chunked prefill implemented and measured
- Understanding of when LMCache, Mooncake, and MLA enter the picture
- The "before paged → after paged → after chunked → after offload" trajectory

## References

- SGLang Pipeline Parallelism for million-token contexts — https://www.lmsys.org/blog/2026-01-15-chunked-pipeline/
- DeepSeek MLA explained — https://planetbanatt.net/articles/mla.html
- "Enabling MLA in any transformer" — https://arxiv.org/abs/2502.14837
- LMCache — https://docs.lmcache.ai/
- Mooncake (PyTorch Ecosystem post) — https://pytorch.org/blog/mooncake-joins-pytorch-ecosystem/
- LMCache + Redis distributed KV — https://barrahome.org/2026/02/08/lmcache-redis-distributed-kv-cache.md
- vLLM chunked prefill docs — https://docs.vllm.ai/en/latest/usage/optimization.html#chunked-prefill
