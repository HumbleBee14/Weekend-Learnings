# 03 — Matrix Multiply

## Files

- `CONCEPTS.md` — the 7-step Boehm progression, why each step wins, where Hopper/Blackwell change the picture
- `matmul.cu` — three kernels (naive, coalesced, shared-memory tiled) + cuBLAS reference, all benchmarked

## Quickstart

```bash
nvcc -O3 -arch=sm_80 matmul.cu -lcublas -o matmul
./matmul
```

Use the right `-arch=` for your GPU (T4 = sm_75, A100 = sm_80, H100 = sm_90).

## Expected output (A100, FP32, 4096×4096×4096)

```
M=N=K=4096  (137 GFLOPs of work per call)

kernel        time       TFLOPS
------        ----       ------
naive         570 ms     0.24
coalesced      82 ms     1.66
smem_tiled     54 ms     2.54
cuBLAS         12 ms    11.40

max abs error vs cuBLAS: 4.2e-04  OK
```

That's ~22% of cuBLAS at step 3. Steps 4–7 (read Boehm) get you to ~95%.

## What you should see

- **Naive → coalesced: ~7× faster.** Pure win from fixing the memory access pattern. No algorithm change.
- **Coalesced → SMEM tiled: ~2× faster.** Reuse via shared memory cuts HBM traffic.
- **SMEM tiled → cuBLAS: ~5× gap.** Steps 4–7 (thread tiling, vec loads, double buffering) close it.
- **Correctness check.** All three kernels agree with cuBLAS within FP32 precision.

## Try

- **Profile each kernel with `ncu`:** `ncu --set full ./matmul`. Look at "Memory Throughput" and "Compute (SM) Throughput." Naive is bandwidth-starved; SMEM-tiled is much closer to compute-bound.
- **Implement Step 4** (1D thread tiling — each thread computes 8 output rows). Should give another 2× over Step 3.
- **Run on smaller sizes** (M=N=K=512). cuBLAS overhead becomes visible. Your kernels might briefly look more competitive.
- **Switch to FP16/BF16.** You'd need to use tensor cores via inline PTX (`mma.sync`) or — much easier — Triton (Topic 4) or CUTLASS. Skip this in raw CUDA C++.

## What you should *not* try

Don't try to write a Hopper TMA + WGMMA + warp-specialized matmul in raw CUDA C++. The Pranjal Shankhdhar worklog ([cudaforfun.substack.com](https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog)) shows what that looks like — it's the right reference but not the right exercise. For modern hardware, the practical paths are:

- **Triton** (Topic 4) — Python, the compiler handles WGMMA + TMA + warp spec
- **CUTLASS / CuTe-DSL** (compiler-and-kernels Level 4) — when you need maximum control over a custom epilogue or layout

## Where this goes

Topic 4 writes the same matmul in Triton, in 1/4 the lines, often with better performance. The point of doing it in CUDA C++ first is to internalize what the compiler is doing for you.
