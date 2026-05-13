# 01 — Attention from scratch

> Prereq: linear algebra, NumPy. No GPU required.

Goal: produce a NumPy reference for scaled dot-product attention you will use to bit-exactly verify every kernel for the rest of the level, and feel the `O(N²)` memory wall in your own measurements.

## What you build

1. `attention_ref.py` — naive SDPA in NumPy. ~30 lines.
2. `memory_wall.py` — measure the memory of the score matrix `S` for `N ∈ {512, 1024, 2048, 4096, 8192}`; print a table.
3. `masks.py` — write causal, sliding-window, and document masks as boolean arrays. Apply them as `score.masked_fill(~mask, -inf)` style. No efficiency tricks yet.

## The formula

For one head:

```
S = Q @ K.T * (1 / sqrt(d_head))   # (N, N)
P = softmax(S, axis=-1)             # (N, N), row-stochastic
O = P @ V                           # (N, d)
```

Multi-head is the same with a leading head dimension; batched is the same with leading batch + head dims. Get the single-head case right and the rest is reshape gymnastics.

The `1/sqrt(d_head)` scale exists because at random init `E[<q, k>] = 0` and `Var[<q, k>] = d_head`. Without the scale, dot products grow with `d_head`, softmax peaks, gradients vanish. Always grep for this scale first when a kernel "works" but the loss is nan.

## The memory wall (do this, don't just read it)

For `N=8192`, `d_head=128`, bf16, one head:
- `Q, K, V`: each `8192 × 128 × 2 B = 2 MB`. Total inputs: ~6 MB.
- `S`: `8192 × 8192 × 2 B = 128 MB`. Twenty-one times the size of the inputs.
- `P`: another 128 MB.
- `O`: 2 MB.

Naive PyTorch attention writes `S` to HBM, reads it back to compute row maxes, reads it back to compute row sums, reads it back to multiply by `V`. That is *four* round-trips of a 128 MB tensor per head. Multiply by `num_heads × batch` and HBM bandwidth becomes the entire story.

Now compare to a tiled implementation: `S` and `P` never leave SRAM. Total HBM traffic is `Q + K + V + O ≈ 8 MB` — roughly `4 × N × d_head × 2B`. That is `O(N · d)` instead of `O(N²)`. At N=8192 it's a 60× reduction in HBM traffic. **This is the whole point of FlashAttention.**

Your `memory_wall.py` should print:

```
N        S size (MB)  Inputs+O (MB)  Ratio (S / inputs)
512      0.5           0.5            1.0x
1024     2.0           1.0            2.0x
2048     8.0           2.0            4.0x
4096     32.0          4.0            8.0x
8192     128.0         8.0            16.0x
```

Write three sentences in `notes.md` describing what this table means for HBM bandwidth in a real kernel.

## The mask vocabulary

You will use these throughout. Build them now as boolean arrays of shape `(N, N)` where `True` means "attend":

- **Causal:** `q_idx >= kv_idx`. The lower triangle.
- **Sliding-window causal:** `(q_idx >= kv_idx) & (q_idx - kv_idx <= W)`. A band of width `W` along the diagonal.
- **Document mask:** given `doc_id` of shape `(N,)`, `doc_id[q] == doc_id[kv]`. Block-diagonal.
- **Sink + sliding window:** `(kv_idx < S) | (q_idx - kv_idx <= W)`. The first `S` columns plus a band — this is the StreamingLLM pattern and what the capstone uses.

For each mask, compute the fraction of `True` entries. Sliding-window at `W=512, N=8192` is ~6% true. This is the sparsity FlexAttention's BlockMask will exploit.

What the four masks look like on an 8×8 toy (`█` = attended, `░` = masked, rows = Q, cols = K):

```
   causal               sliding window W=2     document (2 docs)     sink S=2 + window W=2
   k0 k1 k2 k3 k4 k5 k6 k7    same axes              same axes              same axes
q0 █  ░  ░  ░  ░  ░  ░  ░    █  ░  ░  ░  ░  ░  ░  ░  █  █  █  █  ░  ░  ░  ░  █  ░  ░  ░  ░  ░  ░  ░
q1 █  █  ░  ░  ░  ░  ░  ░    █  █  ░  ░  ░  ░  ░  ░  █  █  █  █  ░  ░  ░  ░  █  █  ░  ░  ░  ░  ░  ░
q2 █  █  █  ░  ░  ░  ░  ░    █  █  █  ░  ░  ░  ░  ░  █  █  █  █  ░  ░  ░  ░  █  █  █  ░  ░  ░  ░  ░
q3 █  █  █  █  ░  ░  ░  ░    ░  █  █  █  ░  ░  ░  ░  █  █  █  █  ░  ░  ░  ░  █  █  █  █  ░  ░  ░  ░
q4 █  █  █  █  █  ░  ░  ░    ░  ░  █  █  █  ░  ░  ░  ░  ░  ░  ░  █  █  █  █  █  █  ░  █  █  ░  ░  ░
q5 █  █  █  █  █  █  ░  ░    ░  ░  ░  █  █  █  ░  ░  ░  ░  ░  ░  █  █  █  █  █  █  ░  ░  █  █  ░  ░
q6 █  █  █  █  █  █  █  ░    ░  ░  ░  ░  █  █  █  ░  ░  ░  ░  ░  █  █  █  █  █  █  ░  ░  ░  █  █  ░
q7 █  █  █  █  █  █  █  █    ░  ░  ░  ░  ░  █  █  █  ░  ░  ░  ░  █  █  █  █  █  █  ░  ░  ░  ░  █  █
```

Read each grid by row: q_i's row of `█` cells is the set of keys it sums over. The sink+window pattern is the StreamingLLM shape — first columns always visible, plus a moving causal band — and the one your capstone implements.

## Definition of done

- [ ] `attention_ref.py` matches `F.scaled_dot_product_attention` to `1e-5` in fp32.
- [ ] `memory_wall.py` prints the table above for your hardware.
- [ ] `masks.py` builds all four masks and prints sparsity fractions.
- [ ] `notes.md` has the three sentences on HBM traffic.

## What you can do after this sub-module

Read any FlashAttention paper introduction without flinching, and feel exactly why `O(N²)` HBM traffic is the enemy.
