# 02 — Online softmax, derived

> Prereq: sub-module 01. No GPU.

This is the algebra at the center of FlashAttention. The goal is not to recognize the formula — it is to derive it on a whiteboard from "I want to softmax a row in pieces" and have the worked numbers in your head. If you can do that, the FA2 Triton kernel in sub-module 03 is a transcription, not a translation.

## The problem

Softmax is row-wise. Tiling processes the row in column-chunks of, say, 64. We want a single-pass algorithm: process tile 1, then tile 2, ..., then tile T, maintaining a small running state, and at the end the answer must equal `softmax([x_1 ... x_N])` exactly (in real arithmetic; bf16-tight in practice).

Naively, softmax needs *three* passes over the row:
1. Pass 1: find `m = max(x)`.
2. Pass 2: compute `e_i = exp(x_i - m)`, sum to `ℓ`.
3. Pass 3: divide `e_i / ℓ`.

Online softmax (Milakov & Gimelshein 2018) collapses passes 1 and 2 into one. Combined with the attention output accumulator update, the whole attention forward becomes one pass over K/V.

## The derivation (do this on paper before reading)

Let `x = [x_1, ..., x_N]`. Define the *tile-local* state after seeing tile `t`:

- `m_t = max(x_1, ..., x_{end of tile t})` — running max.
- `ℓ_t = sum_{i <= end of tile t} exp(x_i - m_t)` — running denominator, *renormalized to the current max*.

The trick: when a new tile reveals a larger max `m_{t+1} > m_t`, the old `ℓ_t` was computed relative to the wrong (smaller) max. To fix it, multiply by `exp(m_t - m_{t+1})`. Because:

```
ℓ_t = sum exp(x_i - m_t)
ℓ_t * exp(m_t - m_{t+1}) = sum exp(x_i - m_t) * exp(m_t - m_{t+1})
                         = sum exp(x_i - m_{t+1})    <-- renormalized to new max
```

Then add the new tile's contributions:

```
m_{t+1} = max(m_t, max(x_tile))
ℓ_{t+1} = ℓ_t * exp(m_t - m_{t+1})  +  sum_{i in tile} exp(x_i - m_{t+1})
```

At the end, `softmax(x)_i = exp(x_i - m_T) / ℓ_T`. Bit-exactly equal to the three-pass version under IEEE rounding.

**The two things to internalize:**

1. The factor `exp(m_t - m_{t+1})` is always `<= 1`. When the new tile doesn't raise the max it equals `1` and the rescale is free. **This is the property FA4 exploits to skip rescales when the max didn't change** (Mar 2026). On typical attention rows the running max stabilizes early, and ~90% of tiles after the first few don't change `m`. FA4 reports ~10× fewer correction ops.
2. The rescale is also what you apply to the *output accumulator* `O`. In attention, `O = sum_t P_t @ V_t` where `P_t = exp(S_t - m_T) / ℓ_T`. While streaming, you compute partial `O_t` with the wrong max; when the max updates you rescale `O` by the same `exp(m_t - m_{t+1})` factor. Same math, different tensor.

## Worked example (run this with pencil first, then verify with `online_softmax_walk.py`)

Row: `x = [1.0, 2.0, 4.0, 3.5, 0.5, 5.0, 4.8, 2.2]`. Tile size 4. Two tiles.

**Three-pass reference.**
- `m = 5.0`.
- `exp(x - m) = [0.0183, 0.0498, 0.3679, 0.2231, 0.0111, 1.0, 0.8187, 0.0608]`.
- `ℓ = 2.5497`.
- `softmax(x) = [0.00718, 0.01953, 0.14430, 0.08750, 0.00434, 0.39220, 0.32109, 0.02384]`.

**Online, tile 1 = [1.0, 2.0, 4.0, 3.5]:**
- `m_1 = 4.0`.
- `ℓ_1 = exp(1-4) + exp(2-4) + exp(4-4) + exp(3.5-4) = 0.0498 + 0.1353 + 1.0 + 0.6065 = 1.7916`.

**Online, tile 2 = [0.5, 5.0, 4.8, 2.2]:**
- `tile_max = 5.0`.
- `m_2 = max(4.0, 5.0) = 5.0`.
- Rescale factor: `exp(m_1 - m_2) = exp(-1) = 0.3679`.
- `ℓ_1 rescaled = 1.7916 * 0.3679 = 0.6591`.
- Tile-2 contribution: `exp(0.5-5) + exp(5-5) + exp(4.8-5) + exp(2.2-5) = 0.0111 + 1.0 + 0.8187 + 0.0608 = 1.8906`.
- `ℓ_2 = 0.6591 + 1.8906 = 2.5497`.

`ℓ_2 == ℓ_reference`. The denominators match. Now divide `exp(x_i - m_2)` by `ℓ_2` for each i — same result as the three-pass.

State evolution at a glance (this is exactly what your `online_softmax_walk.py` should print):

```
 step │ tile values         │ tile_max │  m_t  │ rescale=exp(m_prev-m_t) │   ℓ_t (running)
──────┼─────────────────────┼──────────┼───────┼─────────────────────────┼────────────────────
 init │ —                   │    —     │ -inf  │           —             │   0.0000
   1  │ [1.0, 2.0, 4.0,3.5] │   4.0    │  4.0  │  exp(-inf-4) = 0  (×)   │   0 + 1.7916 = 1.7916
   2  │ [0.5, 5.0, 4.8,2.2] │   5.0    │  5.0  │  exp(4-5) = 0.3679      │   1.7916·0.3679 + 1.8906 = 2.5497
──────┴─────────────────────┴──────────┴───────┴─────────────────────────┴────────────────────
                                                       3-pass reference ℓ = 2.5497  ✓
```

In the attention version the same row picks up a third state `O_t` (a `(d,)` accumulator) that is rescaled by the same `exp(m_prev - m_t)` factor before adding the new tile's `P_t @ V_t`. The recurrence is the headline FA2 update: rescale-then-add applies to `ℓ` and `O` together.

If you got 2.5497 by hand, the FA2 kernel will be straightforward.

## What to build

- `online_softmax_walk.py` — pencil version. Steps through the recursion and prints `m_t`, `ℓ_t`, the rescale factor at each tile. Compares to the three-pass reference. **Run this; modify the row; build intuition.**
- `online_attention_np.py` — extend the same idea to a one-row attention computation: given Q[0], iterate over K/V tiles, maintain `(m, ℓ, O)` per the recursion. Verify bit-equal against `attention_ref` from sub-module 01.

These two scripts are 30–50 lines each. They are the smallest reproductions of FA2 you can have and they run on a laptop.

## Definition of done

- [ ] You can derive the recursion on paper without notes.
- [ ] `online_softmax_walk.py` matches `scipy.special.softmax` to `1e-12` in fp64 for at least three different rows you make up.
- [ ] `online_attention_np.py` matches `attention_ref` to `1e-10` in fp64.
- [ ] You wrote in `notes.md`: what happens when `m_t == m_{t+1}`? (Answer: rescale factor = 1, free; FA4 skips this case.)

## References

- [Milakov & Gimelshein, "Online normalizer calculation for softmax" (2018)](https://arxiv.org/abs/1805.02867) — the original two-page paper.
- [Zihao Ye, "From Online Softmax to FlashAttention" notes](https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf) — the cleanest exposition of the math you'll find.
- [Damek Davis, "Basic idea behind flash attention (V1)"](https://damek.github.io/random/basic-idea-behind-flash-attention/) — short and visual.
