# notes — fa2 tiling

## Pitfalls you will hit (and notes for when you do)

1. **Rescale order.** `O_i = O_i * rescale + P @ V_j`, not `(O_i + P @ V_j) * rescale`. The smoke test in `fa2_numpy.py` and `fa2_triton.py` will fail loudly. The bug also shows up visually: output is right when m never changes (homogeneous Q,K) and drifts otherwise.
2. **`exp(-inf - -inf)` = nan.** Happens on the first tile and on tiles where every position is masked. Guard with `tl.where(m_i == -inf, 0.0, exp(m_i - m_new))`.
3. **bf16 in the inner accumulator.** Don't. Keep `m_i, l_i, O_i, S, P` in fp32. The final cast to bf16 happens at the write to `O`. If you accumulate in bf16 you'll lose 2–3 digits of precision and the test against fp32 SDPA will fail.
4. **Boundary masking on N.** When `N` is not a multiple of `BLOCK_M` or `BLOCK_N`, you need `mask=offs<N, other=0` on every `tl.load` and you need to mask `S` to `-inf` for OOB KV positions, not 0 (zero would let them participate in the softmax with weight `exp(0)/ell = 1/ell`).
5. **Don't broadcast `rescale` wrong.** `rescale` is `(BLOCK_M,)`; for `O_i` you want `rescale[:, None]`; for `l_i` you don't index.
6. **`tl.dot(p, v)` dtype.** Triton's `tl.dot` is picky about input dtypes — if `p` is fp32 and `v` is bf16, cast `p` to `v.dtype` first.

## What surprises learners

- The kernel is only ~50 lines and it really does include the entire FlashAttention algorithm. Most of the "complexity" of FA2 is in the C++ kernel-launch boilerplate and the backward pass.
- The forward kernel produces `L = m + log(ℓ)` per row, not just `O`. That's the only intermediate the backward needs — at `8N` bytes (fp32) it's tiny compared to the `N^2` score matrix.

## Where to go from here

- Add `make_tensor_descriptor` loads. On T4 no win; on H100 unlocks TMA.
- Try `num_warps in [4, 8]`, `num_stages in [2, 3, 4]`, `BLOCK_M in [64, 128]`, `BLOCK_N in [32, 64, 128]`. Build a small autotune. Be careful with the pruning function from Level 1 — D=64 vs D=128 has very different register pressure.
- Sub-module 04 turns on warp specialization.
