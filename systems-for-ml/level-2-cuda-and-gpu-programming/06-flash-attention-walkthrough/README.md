# 06 — FlashAttention Walkthrough

## Files

- `CONCEPTS.md` — naive attention's HBM problem, the FA tiling idea, FA2 algorithm pseudocode, FA1→FA2→FA3→FA4 progression, FlashInfer
- `online_softmax_numpy.py` — the online softmax recursion, in pure NumPy. The heart of FA, isolated.
- `flash_attention_minimal.py` — minimal FA2 in Triton (~80 lines). Forward pass only, no backward, no causal masking. Read every line.
- **`READING-FLASH-ATTENTION.md`** — guided reading of the production FA stack: dao-ailab's FA2 in CUDA C++ (with CUTLASS), FlashInfer's Python dispatcher + C++ kernels, vLLM's Triton FA backend, and a note on FA4 in CuTe-DSL. Same algorithm, four polish levels.

## Quickstart

```bash
# CPU is enough to understand the algorithm
python online_softmax_numpy.py

# GPU needed for the actual kernel
pip install triton torch
python flash_attention_minimal.py
```

## Expected output

`online_softmax_numpy.py`:
```
case                         tile_size  max_err
--------------------------------------------------
simple                       16         0.00e+00
simple                       64         0.00e+00
simple                       256        0.00e+00
with negatives               16         5.55e-17
needs subtract-max           16         1.42e-14
long with one big spike      16         3.12e-15
...
```

The online softmax matches the naive (two-pass) softmax to within FP64 precision. It's mathematically equivalent — only the order of operations differs.

`flash_attention_minimal.py`:
```
max abs error vs reference: 0.0050  (small fp16 noise expected)

B=1, H=8, N=1024, D=64, fp16:
  reference (torch)         3.451 ms  (1530 GFLOPS)
  minimal flash attn        0.821 ms  (6430 GFLOPS)
  torch SDPA (FA2 internal) 0.534 ms  (9890 GFLOPS)
```

Three takeaways:
- Our minimal FA is ~4× faster than the reference (which materializes the (N, N) matrix).
- Our minimal FA is ~50% slower than torch's SDPA (which calls into the production FA2/FA3 from dao-ailab). That's expected — the production version has warp specialization, double buffering, and tuned block sizes.
- Numerically the FA path matches the reference within fp16 noise tolerances.

## Try

- **Increase N to 4096 or 8192.** The reference will get *much* slower (it's quadratic in HBM); FA stays close to linear. The gap widens.
- **Profile with `ncu`.** Compare HBM bytes moved between reference and FA. The reference will move 10×+ more.
- **Read the [Triton FA tutorial](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html).** It's the canonical version with causal masking, varying head_dim, and proper handling of N not divisible by BLOCK_N. Our `flash_attention_minimal.py` is a stripped-down learning version of the same.
- **Add causal masking.** Mask out `s` where `offs_n > offs_m` (positions in the future) before the online softmax update. Verify against `reference_attention` with `is_causal=True`.

## Don't try this

- Don't try to write FA3 or FA4 yourself. Read the papers and the Tri Dao / Modal blogs. They're months of work even for experts.
- Don't try to add backward pass to your kernel. The backward is a different algorithm with different layout constraints. Use `torch.autograd.Function` and call `flash_attn` from dao-ailab's repo if you need the backward.

## What you should be able to do after this topic

- Derive the online softmax recursion from scratch on a whiteboard
- Trace one Q-tile's iteration through the FA2 kernel (which K, V tiles it sees, how the running m/l/O state evolves)
- Explain why FA's win is *bandwidth* (no quadratic intermediate in HBM), not *compute* (same FLOPs)
- Name the three changes FA3 makes (warp spec, ping-pong, FP8) and the four changes FA4 makes (5-stage pipeline, software exp, conditional rescale, role-specialized warps)
- Know what FlashInfer does that raw FA doesn't (ragged batching, paged KV, JIT dispatch)

## What you'll measure for the writeup

The 200-word writeup test from CONCEPTS.md. Write it. If it's hard, re-read FA2 paper Section 3 + the Tri Dao FA3 blog.

## Where this goes

This is the last topic of Level 2. Level 3 (profiling) takes everything you've built and learns to measure it: Nsight Systems, Nsight Compute, the roofline model. Level 4 (optimization) replaces your `mini-serve` batcher with a paged KV cache and continuous batching — using all the kernel knowledge from this level.
