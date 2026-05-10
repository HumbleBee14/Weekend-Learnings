# 13 — Speculative Decoding

## What it is

Decode is sequential and memory-bound. Each step processes 1 token. You read the entire model from HBM to compute that token. Tons of bandwidth, very little compute used.

Speculative decoding's insight: if you can *guess* the next K tokens and verify all K in parallel, you've used the model's compute capacity (which was sitting idle) to amortize the bandwidth cost across multiple tokens.

```
Standard decode:
  Step 1: read full model, compute token 1
  Step 2: read full model, compute token 2
  Step 3: read full model, compute token 3
  → 3 model reads for 3 tokens

Spec decode (verify K=4 candidates):
  Step 1: read full model ONCE, verify all 4 candidate tokens in parallel
  → On accept: 1 model read for 4 tokens
  → On reject after token 2: 1 model read for 2 tokens (still 2× win)
```

When K candidates are accepted, you got 4 tokens at the bandwidth cost of 1. Decode throughput jumps 2-4× on average.

The catch: you need a way to *guess* the candidates. The methods differ on how.

## Three method families

### Draft model

Train (or use an off-the-shelf) small model alongside your big one. Draft model proposes K tokens; big model verifies.

- Pro: works on any architecture
- Con: need a separate draft model that's good enough; coordination overhead

Less common in 2026 since EAGLE-class methods replaced it.

### EAGLE (and successors)

Trains a *small head* on top of the big model's hidden states to predict the next K tokens. Single model, single forward pass for verification.

EAGLE-3 (production in vLLM) is the 2026 standard. Up to 2.5× speedup in real workloads.

### n-gram

No model at all. Use the previously-generated text as a lookup: if "the quick brown" was followed by "fox" earlier in the same response, predict "fox" next time you see "the quick brown."

- Pro: zero training, works on any model
- Con: only useful for repetitive text (code, structured output, repeated phrases)

In vLLM/SGLang as `--speculative-config '{"method": "ngram", ...}'`.

## EAGLE-3 — the 2026 standard

EAGLE-3 trains a small "draft head" on top of the target model's hidden states. The head proposes K candidate tokens; verification passes them all through the target model in a single forward pass.

Key 2026 facts:

- **Production in vLLM** since v0.9.1 (CUDA-graph captured for low overhead)
- **2.5× speedup** typical on chat workloads; less on hard reasoning (acceptance rate drops)
- **Per-position acceptance rate, mean acceptance length** exposed as Prometheus metrics
- **gpt-oss perf improvements** April 2026 pushed speedup higher on that family

## Tree-spec verification

EAGLE-3 doesn't propose K tokens linearly. It proposes a **tree** of K candidate continuations:

```
       <past tokens>
            ↓
        token A           ← top-1 candidate
       /   |   \
     B1    B2   B3        ← top-3 candidates following A
    /  \    |    |
   C1  C2   D1   E1       ← further continuations
```

Verification computes attention over the *whole tree* in one forward pass, with a custom mask that ensures each path is causal within itself. The verifier picks the longest accepted prefix.

Result: higher acceptance rates than linear speculation, because branches give the model more chances to "stay on a likely path."

## P-EAGLE — 2026's biggest spec-decode delta

**P-EAGLE** (Parallel-Drafting EAGLE), introduced Feb 2026, is the largest improvement to spec decoding in years.

Standard EAGLE-3 generates K draft tokens *sequentially* — even the draft head runs K times. P-EAGLE generates all K in **one forward pass**:

- Up to **1.69× over vanilla EAGLE-3** on B200
- Integrated in vLLM v0.16.0 as `"parallel_drafting": true`
- Pre-trained heads available on HF for GPT-OSS 120B/20B, Qwen3-Coder 30B
- SGLang has an open issue tracking integration

Status as of May 2026: production-ready in vLLM, rolling out to other engines.

## Acceptance rate — the metric that matters

Acceptance rate = fraction of speculated tokens that the verifier accepts. Higher = bigger speedup.

Typical values:
```
Workload type       Acceptance rate    Speedup
─────────────────────────────────────────────────
Chat / general       60-80%             2.0-2.5×
Code generation     70-90%             2.5-3.5× (more predictable)
Hard reasoning       40-60%             1.3-1.8×
Translation          50-70%             1.7-2.2×
```

Spec decode pays off whenever acceptance rate × candidate count > 1 (you're getting more tokens per model forward than without spec). For most chat-shaped workloads, that's true at K=4.

## Quality is preserved exactly

Spec decode is **mathematically equivalent** to standard sampling, *if* implemented correctly. The output distribution is identical — same temperature, same top-p, same seed produces the same tokens.

When this fails: bugs in the verification mask, RNG state inconsistency between draft and verify, off-by-one in the rollback. These are subtle and easy to miss in chat eyeballing. Always run lm-eval-harness with-and-without spec decode to catch them.

## The systems integration story

Topic 17 covers how spec decode interacts with continuous batching, scheduler decisions, and KV cache rollback semantics. This topic is the *algorithmic* side; Topic 17 is the *systems* side.

## Pitfalls

1. **Trusting "spec decode is on" without measuring acceptance rate.** Easy to enable; easy for it to silently produce 0% acceptance because of config mismatch.
2. **Using a poorly-trained draft model / EAGLE head.** Acceptance rate craters.
3. **Forgetting that K=1 spec is always slower.** Need K≥3 for typical workloads.
4. **Comparing tokens/sec without accounting for acceptance.** "Tokens/sec" should be *accepted* tokens, not proposed.
5. **Mismatched draft and verifier sampling.** Same temperature, top-p; otherwise math doesn't work out.

## What you'll do

Enable n-gram spec decode in vLLM (no draft head needed):

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --speculative-config '{
        "method": "ngram",
        "num_speculative_tokens": 5,
        "prompt_lookup_max": 5,
        "prompt_lookup_min": 2
    }'
```

Run a code-generation workload (where n-gram excels). Measure:

- Tokens/sec with vs without spec decode
- Acceptance rate
- TTFT (should not regress)
- Quality (lm-eval HumanEval)

Then if you have a model with a pre-trained EAGLE head (Llama-3.1-8B, gpt-oss, Qwen3-Coder), enable EAGLE-3 and compare.

## References

- EAGLE-3 paper — https://arxiv.org/html/2503.01840v1
- P-EAGLE (Feb 2026) — vLLM blog: https://vllm.ai/blog/p-eagle · AWS ML blog (Mar 2026): https://aws.amazon.com/blogs/machine-learning/p-eagle-faster-llm-inference-with-parallel-speculative-decoding-in-vllm/
- vLLM spec decode for gpt-oss (April 2026 perf improvements) — https://developers.redhat.com/articles/2026/04/16/performance-improvements-speculative-decoding-vllm-gpt-oss
- vLLM spec-decode docs — https://docs.vllm.ai/en/latest/features/spec_decode.html
- Original speculative decoding paper (Leviathan et al.) — https://arxiv.org/abs/2211.17192
