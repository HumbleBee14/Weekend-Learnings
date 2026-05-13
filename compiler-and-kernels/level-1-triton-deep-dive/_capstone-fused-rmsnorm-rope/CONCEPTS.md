# The fusion: math, kernel pattern, and what Liger-Kernel did

## The unfused version

In a LLaMA-style attention block, the input goes through:

```
h_norm = rmsnorm(h, w_norm)
q, k, v = qkv_proj(h_norm)
q = rope(q, cos, sin, position_ids)
k = rope(k, cos, sin, position_ids)
# attention(q, k, v)...
```

In eager mode each of those is a separate kernel:

| Step | Reads | Writes | HBM bytes (B=32, S=2048, H=4096, fp16) |
|---|---|---|---|
| rmsnorm | h, w_norm | h_norm | 32·2048·4096·2 + 4096·2 + same = 1.07 GB |
| qkv_proj | h_norm, W_qkv | q, k, v | (matmul; ignored for this analysis) |
| rope_q | q, cos, sin | q | 1.07 GB + 2·2048·128·2 |
| rope_k | k, cos, sin | k | 1.07 GB + ... |

The HBM traffic for RMSNorm + RoPE alone (just these elementwise+rotation steps, not counting the GEMM) is ~3 GB per layer per call. At 32 layers × ~100 calls/sec at modest decode rates that's a real bandwidth bill.

## What fusion buys

The simplest fusion combines `rmsnorm` with the RoPE that immediately follows for `q` and `k` in attention. (We can't easily fuse with the GEMM in between in pure Triton — that's covered by `torch.compile` epilogue fusion in Level 2.) The fused RMSNorm+RoPE kernel:

```
read x once (the normed-and-then-rotated tensor, before the GEMM in this design, fused after)
compute rms over x
multiply x by inv_rms * w  (this is the RMSNorm normalize step)
apply RoPE rotation using precomputed cos, sin
write output once
```

Note: in production this fusion usually goes `qkv_proj → fused_rmsnorm_rope_for_q_and_k → attention`. The norm-before-projection pattern is the one we'd fuse with the projection's epilogue (a `torch.compile` job in Level 2). For *this* capstone, to keep things teachable, we'll fuse `RMSNorm` over the residual stream with a `RoPE` that takes the same shape — the algorithmic pattern is identical to the production version.

HBM bytes for the fused version: read x once (1.07 GB), read w_norm once (8 KB, amortized in persistent kernel), read cos+sin once (1 MB, amortized), write output once (1.07 GB). Total: ~2.14 GB. We've cut HBM traffic by ~30% on this fragment of the model and we've cut kernel launches by 3×.

## The fusion pattern in code

```python
@triton.jit
def fused_kernel(...):
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)

    # ONE pass over the row.
    x = tl.load(x_row + cols, mask=mask).to(tl.float32)
    w = tl.load(w_norm + cols, mask=mask).to(tl.float32)

    # RMSNorm in registers.
    sum_sq = tl.sum(x * x, axis=0)
    inv_rms = 1.0 / tl.sqrt(sum_sq / n_cols + eps)
    x = x * inv_rms * w   # x is still in registers, just transformed

    # RoPE in registers. cos/sin tables loaded once for this row.
    pair_idx = cols // 2  # which sin/cos pair
    is_first_of_pair = (cols % 2) == 0
    cos_val = tl.load(cos_table + position * head_dim + pair_idx * 2 + 0, mask=mask)
    sin_val = tl.load(sin_table + position * head_dim + pair_idx * 2 + 0, mask=mask)

    # ... rotation math, all in registers ...

    tl.store(out_row + cols, result.to(tl.float16), mask=mask)
```

That's the pattern. Three components:
1. Load input(s) into registers.
2. Compute all stages of the fused operation on the in-register tile.
3. Write output back.

## What Liger-Kernel does that you might miss

Read their actual code, but a few things to look for:

- **Backward pass via custom `torch.autograd.Function`.** Forward is the easy half; the backward (computing `dx`, `dw`) requires recomputing the inverse RMS and re-applying the rotation transpose. They handle it.
- **Compile-time choices via `tl.constexpr`.** Whether to use a separate `casting_mode` (BF16 forward + FP32 reduce, FP32 throughout, etc.) is `constexpr`-gated. The compiler emits different code for each.
- **One row per program, full row in one tile.** Same as our sub-module 03 — no inner tiling within the row.
- **`num_warps` not hand-picked.** They set up `num_warps = min(MAX_FUSED_SIZE // BLOCK_SIZE, ...)` — derived from the actual tile size.
- **They don't fuse with the matmul.** That's left to `torch.compile` Inductor's epilogue fusion. They draw the line between hand-fused-Triton (norm + rotation) and compile-fused (norm + proj epilogue) deliberately — both are valid; they picked the side that's stable across hardware.

## Persistent extension

If your benchmark batch is large (many rows), make the kernel persistent (sub-module 06 pattern):

```python
@triton.jit
def kernel(...):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    # ... load w, cos, sin ONCE here, before the loop, into registers ...
    for row in range(pid, n_rows, num_pids):
        # ... per-row work using already-loaded w/cos/sin ...
```

This saves redundant loads of w and the rotation tables. On a 64K-row batch with H=4096, the saving is real (5-10% additional bandwidth).

## Warp specialization (Hopper+ only, optional)

On Hopper or Blackwell, add `tl.range(..., warp_specialize=True)` to the row loop. Modest win (5-10%) for this kernel because it's memory-bound; the bigger wins from warp specialization are on compute-bound kernels (matmul, attention with tensor cores). Try it, see what you get. Don't be surprised if the win is small.

## What you'll measure in the profiler

After running `profile.py` on your fused kernel, the `proton` trace should show:

- `dram__bytes_read` per call ≈ `(N_rows * N_cols + N_cols + 2 * max_seqlen * head_dim) * 2` (input + weight + cos+sin tables, fp16)
- `dram__bytes_write` per call ≈ `N_rows * N_cols * 2`
- For persistent: weight + cos/sin reads should be ~equal across SM counts (not per-row scaling)
- `dram__throughput` should be > 80% of peak

If the byte count is roughly 2× what you expect, you're loading something twice. Find the second load and remove it.

## Why this kernel matters as a teaching artifact

It's the smallest non-trivial production fusion: two real operators, one fused kernel, reproducible benchmarking against a real production reference. If you can make this match Liger, you have demonstrated the full Triton workflow end-to-end:
- The mental model (sub-module 01)
- The language (sub-module 02)
- The bandwidth template (sub-module 03)
- Autotune discipline (sub-module 04)
- TMA + warp spec where it earns its keep (sub-module 05)
- Persistence + CUDA graph compatibility (sub-module 06)

Everything else in the rest of this track is a deeper application of these ideas.
