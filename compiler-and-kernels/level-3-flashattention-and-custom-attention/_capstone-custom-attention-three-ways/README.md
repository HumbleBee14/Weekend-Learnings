# Capstone — Custom attention three ways

> Prereq: every sub-module above. Hardware: A100 minimum for meaningful numbers.

A real attention variant, implemented three ways, benchmarked head-to-head. By the end you can answer: **which would I actually ship in production, and why?**

## The variant: sliding-window + sink-tokens + ALiBi

A StreamingLLM-shaped attention pattern with a per-head ALiBi bias. Specifically:

- **Sliding window** of size `W = 512`: queries attend to KV positions `[q - W, q]`.
- **Sink tokens**: queries always attend to the first `S = 4` KV positions, regardless of window distance. This is the [StreamingLLM (arXiv 2309.17453)](https://arxiv.org/abs/2309.17453) fix for long-context decode without retraining.
- **ALiBi bias**: `score -= slopes[h] * |q_idx - kv_idx|`, with the standard geometric-progression slopes per head.
- **Causal**: `q_idx >= kv_idx`.

This is a real pattern. StreamingLLM's KV cache eviction policy uses exactly this mask shape. Production engines that serve very long contexts (vLLM, SGLang) have first-class support for sink + sliding-window because it stabilizes the model's attention sink behavior without retraining.

## The three implementations

### 1. Hand-Triton (`impl_triton.py`)

Build on your sub-module 03 FA2 forward kernel. Add:

- ALiBi bias inside the inner loop: `s = s - slopes[h] * |q_offs[:, None] - n_offs[None, :]|` after the QK^T.
- Sink+window mask: `mask = (q_offs[:, None] >= n_offs[None, :]) & ((q_offs[:, None] - n_offs[None, :] <= W) | (n_offs[None, :] < S))`. Apply as `s = tl.where(mask, s, -inf)`.

Forward only. Match your sub-module 03 numerical tolerance.

The mask shape means most of the KV iteration is wasted on -inf scores. You don't skip blocks. ~6% of `S` is useful at N=8192. This is the lesson — hand-Triton can compute custom variants but can't trivially exploit block sparsity. (Yes, you can hand-roll block skipping. That is a week of work.)

### 2. FlexAttention (`impl_flex.py`)

```python
def alibi_score(score, b, h, q_idx, kv_idx):
    return score - alibi_slopes[h] * (q_idx - kv_idx).abs()

def sink_window_causal(b, h, q_idx, kv_idx):
    causal = q_idx >= kv_idx
    window = (q_idx - kv_idx) <= W
    sink = kv_idx < S_SINK
    return causal & (window | sink)

block_mask = create_block_mask(sink_window_causal, B=None, H=None, Q_LEN=N, KV_LEN=N)
flex = torch.compile(flex_attention, dynamic=False)
out = flex(q, k, v, score_mod=alibi_score, block_mask=block_mask)
```

~10 lines. Block-sparsity automatic — sliding window + sinks at N=8192, W=512, S=4 produces a BlockMask with ~7% blocks kept. Speedup should be roughly proportional to sparsity.

### 3. FlashInfer (`impl_flashinfer.py`)

The production-shaped path. Use FlashInfer's custom-mask plumbing (block-sparse row matrix specifying which K positions each query can see) plus the JIT attention-variant template to inject the ALiBi bias.

```python
# Build BSR mask in advance.
mask_indptr, mask_indices = build_bsr_mask_for_sink_window(N, W=W, S=S_SINK)

wrap = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(workspace, kv_layout="NHD")
wrap.plan(qo_indptr, kv_indptr, num_heads, num_heads, head_dim,
          causal=False,  # we encode causal in the custom mask
          custom_mask=mask_indices, mask_indptr=mask_indptr,
          # ALiBi via the attention bias / soft-cap mechanism, or via a JIT variant
          )
out = wrap.run(q, k, v)
```

FlashInfer is the only one of the three that handles ragged batching out of the box. Even for a single-sequence benchmark, this is the implementation that's closest to what vLLM actually calls.

## What you build

```
_capstone-custom-attention-three-ways/
├── README.md           (this file)
├── reference.py        (NumPy reference; the ground truth)
├── impl_triton.py      (hand-Triton FA2 with sink+window+ALiBi)
├── impl_flex.py        (FlexAttention)
├── impl_flashinfer.py  (FlashInfer with custom mask)
├── bench.py            (head-to-head with do_bench)
└── report.md           (the table + writeup)
```

The reference matters. All three implementations must bit-equal-check (within bf16 tolerance) against `reference.py` on a small shape before you run benchmarks. If they don't agree, the benchmark is meaningless.

## The benchmark table

Fill in `report.md`:

| Impl | N=2048 ms | N=4096 ms | N=8192 ms | N=16384 ms | TFLOPs/s @ 8192 | LoC | Notes |
|---|---|---|---|---|---|---|---|
| F.sdpa (full causal, no ALiBi) | | | | | | | OOM at 16384 likely |
| Hand-Triton (this capstone) | | | | | | | |
| FlexAttention | | | | | | | |
| FlashInfer + custom mask | | | | | | | |

A100, bf16, B=2, H=8, D=64, causal+sliding-window(W=512)+sinks(S=4)+ALiBi. Same warmup, `triton.testing.do_bench` for everything.

## The writeup

Three paragraphs in `report.md`. Not bullets. Not a table summary. Sentences.

**Paragraph 1: which one was fastest, and why.** Honest assessment. Probably FlexAttention or FlashInfer is fastest at N>=4096 because of block sparsity. Hand-Triton without block-skip logic computes all the masked blocks. State the numbers.

**Paragraph 2: which one was easiest to write.** Including: how many lines of new code, how many debugging hours, how confident you'd be modifying it next quarter. FlexAttention should win this by a wide margin.

**Paragraph 3: which would you ship.** Depends on the deployment context:
- Training a new model? FlexAttention (forward + backward, easy variants).
- Serving an existing model in a ragged-batch context? FlashInfer.
- Need a variant FlexAttention can't express? Hand-Triton, accepting the cost.

Write the actual answer for *your* hypothetical project. Justify it.

## The report template (create `report.md` yourself with this skeleton)

```
# Capstone report — sink + sliding-window + ALiBi causal attention

Hardware: ___ (A100 / H100 / B200)
Dtype: bf16. Shape: B=2, H=8, D=64.

## Benchmark
| Impl | N=2048 | N=4096 | N=8192 | N=16384 | eff TFLOPs/s @ 8192 | LoC |
|---|---|---|---|---|---|---|
| F.sdpa (full causal, baseline) | | | | | | — |
| Hand-Triton (this capstone) | | | | | | ~80 |
| FlexAttention + BlockMask | | | | | | ~20 |
| FlashInfer + custom mask | | | | | | ~30 |

BlockMask sparsity at N=8192, W=512, S=4: ~7%.

## Which won and why
(paragraph 1)

## Which was easiest
(paragraph 2)

## Which would I ship
(paragraph 3)
```

## Definition of done

- [ ] All three implementations agree with the NumPy reference to bf16 tolerance.
- [ ] All three benchmark numbers (any "did not run on my hardware" marked clearly).
- [ ] `report.md` written with the three paragraphs (you create it; the skeleton is above).
- [ ] You can defend each implementation choice to a teammate.

## Why this capstone matters

Three months from now, someone will ask you "we want attention sinks for our long-context model — how should we ship it?" The right answer is not "look it up." The right answer is "I built this once; here's the tradeoff."
