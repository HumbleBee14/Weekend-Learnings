# 12 — Speculative Decoding in Production

Level 4 Topic 13 covered the algorithm. This topic is about the systems story: what it costs operationally, how it interacts with batching and KV cache, and whether the speedup survives contact with a real workload.

## The 30-second refresher

Speculative decoding generates K candidate next tokens cheaply (small "draft" model or a tiny head), then has the big "target" model verify all K in **one forward pass**. If the first M tokens are accepted, you got M tokens for the cost of one big forward pass + K draft passes. Net: 1.5-3× speedup *when acceptance rate is high*.

```
Without spec:     T B T B T B T B   (8 big-model forwards for 8 tokens)
With spec K=4:    DDDD T DDDD T     (2 verify passes; 8 draft tokens; 6-8 accepted)
```

D = draft pass (cheap), T = target verify pass (one big forward pass).

## The 2026 spec-decode landscape

Three families that matter:

```
n-gram                       EAGLE-3                   MTP / Multi-Token Prediction
───────                      ───────                   ────────────────────────────
Look up next K tokens in     Tiny adapter on top of   Trained into the base model
recent context history.      base; sees hidden state  itself (DeepSeek-V3 pattern,
Cheapest, no extra weights.  + last token; trained     Qwen3 next-token-N head).
Acceptance: workload-        on base outputs.          Acceptance: ~70-85%.
sensitive (high on chat).    Acceptance: ~70-85%.      No separate draft model.
```

**P-EAGLE** (Feb 2026) — the biggest delta of the year. Parallel-drafting extension of EAGLE-3 that generates K draft tokens in a single forward pass via a learnable shared hidden state. In vLLM v0.16+. Reported 1.10×-1.36× speedups over autoregressive EAGLE-3.

**MTP heads** trained into the base have become the high-end default — DeepSeek-V3, Qwen3, Llama-4-Scout all ship MTP heads. No separate draft model; the spec is "free" architecturally.

## Acceptance rate is the metric

```
speedup ≈ (n_accepted_per_step + 1) / (1 + draft_overhead_factor)

If draft overhead is 10% of target cost and acceptance is 3.0 / 4 drafts:
    speedup ≈ 4.0 / 1.10 ≈ 3.6× ... but only on accepted-heavy steps.
    Steady-state with 70% acceptance ≈ 1.6-1.8×
```

Acceptance varies wildly:

```
Workload                          Acceptance (n-gram / EAGLE-3 / P-EAGLE)
──────────                        ──────────────────────────────────────
Casual chat                       45% / 75% / 80%
Code (fill-in-middle)             60% / 80% / 85%
Long-form writing                 50% / 78% / 82%
Hard reasoning / math             20% / 55% / 60%
Tool-use / structured output      30% / 65% / 70%
```

If your workload is reasoning-heavy, spec decode helps less than the published benchmark numbers suggest. Measure on *your* workload.

## Systems-level interactions

```
Spec decode interacts with:

1. Continuous batching
   The scheduler must handle variable token outputs per step (some
   requests accept 4, some 1). vLLM V1 handles this; older schedulers
   don't.

2. Prefix caching
   The draft pass uses the same KV cache as the target. Prefix cache
   hits help both. No conflict.

3. Multi-LoRA
   Draft and target need to use the same adapter, mostly. Mismatched
   adapters tank acceptance.

4. Quantization
   Target FP8 + draft FP16 is fine. Target NVFP4 + draft FP16 is fine.
   The draft just needs to be cheap; precision freedom is yours.

5. CUDA graphs
   K-token verify passes are a different shape than 1-token decode.
   vLLM captures both shape buckets; first request at each pays the
   capture cost. Don't measure on cold starts.
```

## When spec decode hurts

It can be net-negative under:

```
- Very small batch (concurrency 1) — verify pass overhead dominates
- High batch + low acceptance — wasted verify slots eat throughput
- Workload with tight TTFT SLA — verify pass is bigger and slower
- Quantization mismatch where draft-vs-target outputs diverge
```

Production teams typically run with spec decode on for chat / code, off for hard reasoning, conditional for batch.

## Configuring spec decode in vLLM

n-gram (always available, no extra weights):

```bash
vllm serve <model> \
    --speculative-config '{"method":"ngram","prompt_lookup_max":4,"num_speculative_tokens":4}'
```

EAGLE-3 (needs an EAGLE-3 head trained for your model):

```bash
vllm serve <model> \
    --speculative-config '{
        "method":"eagle3",
        "model":"<eagle3-head-repo>",
        "num_speculative_tokens":5
    }'
```

MTP (model has built-in next-token-N head):

```bash
vllm serve <model> \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

Server emits per-step acceptance rate via `/metrics`:

```
vllm:spec_decode_num_accepted_tokens_total
vllm:spec_decode_num_emitted_tokens_total
vllm:spec_decode_num_draft_tokens_total
```

`accepted / emitted` is your acceptance rate. Monitor it in production; if it drops, your workload mix shifted.

## Quality must be unchanged

Spec decode is **lossless by construction** when implemented correctly — the verify step uses the target model's distribution to accept/reject drafts, so output distribution matches greedy or sampled-from-target. Test it anyway:

```
1. Run lm-eval-harness with spec decode off → baseline scores
2. Run lm-eval-harness with spec decode on → should be within noise
3. If they differ by more than ±0.5%, bug in the implementation
```

This check is non-negotiable. A spec implementation with a subtle sampling-distribution bug looks like "free speedup" until accuracy regresses on a benchmark.

## What to do this topic

1. Enable n-gram spec decode in vLLM. Run the same workloads as Topic 07.
2. Measure speedup, acceptance rate (`/metrics`), and quality (`lm-eval-harness`).
3. Try EAGLE-3 if a trained head exists for your model (the Qwen2.5 series has one).
4. Workload-sensitivity: chat (high acceptance) vs code vs hard reasoning. Three distinct numbers.
5. Cold-start: the first 50 requests will look slower because of CUDA graph capture for the new shapes. Discard them.

## Pitfalls

1. **Comparing cold spec-decode to warm baseline.** Both must be fully warmed.
2. **Reporting acceptance rate without QPS.** Acceptance at concurrency 1 is not the acceptance you'll see at concurrency 32.
3. **Skipping the quality check.** This is the only thing that catches a broken sampling implementation.
4. **Forgetting the base-model match.** The draft head must match the target model version. Mismatched checkpoints silently tank acceptance.
5. **Treating MTP as plug-in.** MTP heads need to be trained alongside the base. You can't bolt one onto a model that wasn't built for it.

## References

- EAGLE-3 paper — https://arxiv.org/abs/2503.01840
- P-EAGLE (Feb 2026) — vLLM blog: https://vllm.ai/blog/p-eagle · AWS blog: https://aws.amazon.com/blogs/machine-learning/p-eagle-faster-llm-inference-with-parallel-speculative-decoding-in-vllm/
- Speculative decoding original (Leviathan et al.) — https://arxiv.org/abs/2211.17192
- DeepSeek-V3 MTP head — https://arxiv.org/abs/2412.19437
- vLLM spec-decode docs — https://docs.vllm.ai/en/stable/features/spec_decode.html
- vLLM Prometheus metrics — https://docs.vllm.ai/en/stable/serving/metrics.html
