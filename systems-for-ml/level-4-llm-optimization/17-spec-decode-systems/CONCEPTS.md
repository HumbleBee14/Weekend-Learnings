# 17 — Speculative Decoding Systems Integration

## What this topic covers

Topic 13 explained spec decode as an algorithm — what EAGLE, n-gram, and tree-spec do mathematically.

This topic covers the *systems* problem: how spec decode fits into a continuous-batching scheduler, how the KV cache handles rollback when verification rejects tokens, how tree-spec verification masks work, and the quality-regression risks of subtle implementation bugs.

These are the things production teams spend serious engineering on.

## Problem 1 — Variable accept-rate breaks the scheduler

Standard continuous batching (Topic 14) assumes one new token per step per request. The scheduler picks the next batch composition assuming this fixed step.

Spec decode breaks this:
- Sometimes a step accepts 4 tokens (all candidates passed verification)
- Sometimes only 1 token (only the first candidate passed)
- Sometimes 0 tokens — wait, no — at least the first candidate is always evaluated

The scheduler must:
- Track how many tokens *each request* accepted this step
- Update KV cache positions accordingly per request
- Reconcile this with batch admission/eviction decisions

Reconciliation strategies:
1. **Decouple at finer granularity.** Schedule on per-token boundaries; let spec decode advance variable amounts.
2. **Accept that batches advance at variable rate.** Some batch slots advance 4 tokens, others 1; the scheduler handles this in its accounting.

vLLM V1 uses (2). The persistent batch from Topic 14 is well-suited because batch state is updated by *diff*, not rebuilt — variable advances are just bigger diffs.

## Problem 2 — KV cache rollback on rejection

Spec decode proposes K candidate tokens. The verifier passes all K through the model. If the verifier rejects token `j`, all subsequent candidates (`j+1`, ..., `K`) are discarded.

But the model already wrote KV cache entries for *all K candidates* during verification (the forward pass populates K and V tensors per position). When tokens get rejected, those KV entries are wrong — they correspond to candidates that won't be the actual sequence.

Solutions:

### Option A — tentative writes, commit on accept

Write KV entries as you go, but don't move the per-request "committed length" pointer until verification finishes. On accept-2-of-4: bump the pointer by 2; the entries for positions 3-4 are leftover and will be overwritten by the next step's writes.

### Option B — atomic commit via paged KV

Allocate KV blocks for the K candidates. After verification, only commit the blocks corresponding to accepted tokens. Free the blocks for rejected tokens.

Both work. (A) is simpler and more common; (B) is cleaner conceptually but adds bookkeeping.

vLLM uses a variant of (A) — the same block manager that handles prefix caching is used for spec decode rollback. Not a separate code path.

## Problem 3 — Tree-spec verification masks

EAGLE-2/3 propose a *tree* of candidate continuations, not a sequence. Verification computes attention over the whole tree in *one* forward pass.

The trick: a custom attention mask that ensures each path through the tree is causal within itself but unaware of sibling paths.

```
Tree (each node is a candidate token):
       root (accepted so far)
      /  |  \
     A   B   C       ← three first-position candidates
    /\   |   |
   D E   F   G       ← second-position candidates per branch
   |
   H                 ← third-position
```

Attention mask (rows are query positions, columns are key positions):

```
        root  A  B  C  D  E  F  G  H
root  [  1   .  .  .  .  .  .  .  . ]   root attends only to itself
A     [  1   1  .  .  .  .  .  .  . ]   A attends to root + A
B     [  1   .  1  .  .  .  .  .  . ]   B attends to root + B
C     [  1   .  .  1  .  .  .  .  . ]
D     [  1   1  .  .  1  .  .  .  . ]   D attends to root + A + D
E     [  1   1  .  .  .  1  .  .  . ]   E attends to root + A + E
F     [  1   .  1  .  .  .  1  .  . ]   F attends to root + B + F
G     [  1   .  .  1  .  .  .  1  . ]
H     [  1   1  .  .  1  .  .  .  1 ]   H attends to root + A + D + H
```

Building this mask correctly is the implementation detail that bites. Off-by-one in mask construction → wrong attention scores → quality regression invisible until benchmarks.

## Problem 4 — Multi-model coordination

If you use a separate draft model (not EAGLE), the draft is potentially on different hardware. Coordination becomes a network/IPC problem:

```
Draft model on GPU 1 ──→ propose K tokens ──→ verifier on GPU 0
                                                    ↓
                                                  reject 2
                                                    ↓
                          new context ←── update draft model state
```

The round-trip latency must be hidden by the win. If draft+verify takes longer than K direct decode steps, you've made things worse.

Practical: keep draft and verifier on the same host. Better: use EAGLE/P-EAGLE which avoids the coordination problem entirely.

## Problem 5 — Quality regression from subtle bugs

Spec decode is *mathematically equivalent* to standard decoding *if implemented correctly*. The output distribution is identical.

The "if" is doing a lot of work. Subtle bugs that change the distribution:

- Off-by-one in the verification mask
- RNG state inconsistency between draft proposal and verifier sampling
- Forgot to check probability ratios for "stochastic accept" (the rejection-sampling step that ensures equivalence)
- Tree-spec node order mismatch between mask and actual tokens

Each of these silently changes the distribution. "Feels fine" testing won't catch it. **Always run lm-eval-harness with-and-without spec decode** to detect distribution drift.

## P-EAGLE and the parallel-drafting interaction

P-EAGLE (Topic 13) generates all K draft tokens in one forward pass instead of K sequential. This changes the systems story:

- Draft cost is amortized → spec decode is worth it at higher acceptance rates
- Tree structure is fixed at draft time (not refined as draft progresses) → tree shape is more important
- The "stochastic accept" step still applies, just on the K parallel candidates

Integration in vLLM v0.16.0 as `"parallel_drafting": true`. SGLang catching up.

## What you'll do

Add cancellation-aware spec decode handling to your `mini-vllm`:

1. After verification, free KV blocks for rejected tokens (or leave-and-overwrite via tentative-writes)
2. Update each request's "committed length" by the number of accepted tokens, not by 1
3. Test: send a request with spec decode enabled. Confirm:
   - Output is identical to non-spec decode (same seed, same temperature)
   - Throughput improved
   - lm-eval-harness scores match within noise

For a real implementation: easier to use vLLM's spec decode directly than to build it from scratch in your `mini-vllm`. The point of this topic is *understanding the systems integration*, not implementing it.

## Pitfalls

1. **Treating spec decode as a "drop-in optimization."** It interacts with everything: scheduler, KV cache, sampling, even backpressure. Roll it out carefully.
2. **Skipping the with/without quality test.** Distribution drift is invisible without it.
3. **Using a draft model on different hardware without measuring.** Network round-trip can dwarf the win.
4. **Hardcoding K=5.** Optimal K depends on workload acceptance rate. Use the engine's auto-tuning.
5. **Ignoring the warmup.** First few spec-decode steps include compilation; not representative.

## What you walk away with

- Understanding of why spec decode is hard at the systems level (variable advance per request, rollback semantics, tree-spec mask)
- Awareness of where bugs hide (mask construction, RNG state, probability ratios)
- The framework for testing: throughput-improvement-AND-distribution-equivalence
- Closing the Level 4 sub-arc

## References

- EAGLE-3 paper — https://arxiv.org/html/2503.01840v1
- P-EAGLE (Feb 2026) — vLLM blog: https://vllm.ai/blog/p-eagle · AWS blog: https://aws.amazon.com/blogs/machine-learning/p-eagle-faster-llm-inference-with-parallel-speculative-decoding-in-vllm/
- vLLM spec decode source: `vllm/spec_decode/`
- "Verify once, sample many" trick (Leviathan et al.) — https://arxiv.org/abs/2211.17192
- vLLM v0.9.1 spec decode docs — https://docs.vllm.ai/en/latest/features/spec_decode.html
