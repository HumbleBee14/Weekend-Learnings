# 03 — Matrix Multiply

## Why matmul matters

Matmul is the kernel underneath every transformer. Every Q/K/V projection, every attention `QK^T` and attention `attn @ V`, every MLP `xW + b`, every output projection — it's all matmul. If you understand how a high-performance matmul is structured, the rest of LLM kernel work stops being mysterious.

This is the longest topic in Level 2 and the most worth the time.

## The problem

Compute `C = A · B` where:
- `A` is `M × K`
- `B` is `K × N`
- `C` is `M × N`

Each output element `C[i,j] = sum_k(A[i,k] * B[k,j])` — a dot product of one row of A with one column of B.

Total work: `2 · M · N · K` floating point operations (one multiply + one add per `k` step, MNK output elements).

## The 7-step Boehm progression (still canonical in 2026)

[Simon Boehm's 2022 worklog](https://siboehm.com/articles/22/CUDA-MMM) is the standard learning path. Read it. The 7 kernels go from a naive ~1 TFLOPS implementation to ~95% of cuBLAS.

```
Step 1  Naive                                  ~250 GFLOPS    (1×)
Step 2  Global memory coalescing               ~1.7 TFLOPS    (~7×)
Step 3  Shared memory tiling                   ~2.5 TFLOPS    (~10×)
Step 4  1D thread tiling (TM rows/thread)      ~5 TFLOPS      (~20×)
Step 5  2D thread tiling (TM × TN/thread)      ~10 TFLOPS     (~40×)
Step 6  Vectorized loads (float4)              ~14 TFLOPS     (~55×)
Step 7  Warp tiling + double-buffered SMEM     ~17 TFLOPS     (~95% of cuBLAS)
```

Numbers above are A100 FP32. On Hopper FP16/BF16 with tensor cores it's 10× higher again.

We won't reproduce all 7 here — read Boehm's writeup and implement them yourself. We'll build steps 1, 2, and 3 in this folder. The remaining four are an exercise.

## Step 1 — Naive

One thread per output element. Each thread reads one row of A and one column of B from HBM.

```cuda
__global__ void matmul_naive(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
```

Why is this slow? Look at the memory access pattern.

Within a warp (32 threads, consecutive `col` from 0 to 31, all sharing the same `row`):
- A reads: every thread reads `A[row * K + k]` — same address! All 32 reads coalesce into 1.
- B reads: thread 0 reads `B[k * N + 0]`, thread 1 reads `B[k * N + 1]`, ..., thread 31 reads `B[k * N + 31]`. Consecutive — coalesced.

That part is actually fine. The real problem: **no data reuse**. Each `A[row, k]` is read 32 times (once per `col` in the warp), but those reads come from different threads, not cached in shared memory.

For `M = N = K = 4096`: total HBM traffic is roughly `K * (M*N + M*N) = 2 * M*N*K` bytes. That's an enormous read amplification. The hardware can't compute fast enough to keep up with the bandwidth demand.

## Step 2 — Coalescing the loop order

Subtle bug in Step 1: the way `threadIdx.x` maps to output is row-major *backwards*. Boehm's article shows the swap that fixes it. The fix often gives 7× because uncoalesced B-column reads were silently happening before.

The lesson: **coalescing is the first thing to verify**. Run with Nsight Compute (`ncu --set full`) and look at "Achieved Bandwidth" vs "Peak Bandwidth." If you're below 50% of peak, you're not coalescing.

## Step 3 — Shared memory tiling

Key insight: we want each chunk of A and B to be loaded from HBM **once** and reused for many output elements. Shared memory is the buffer for that reuse.

The pattern:

```
                    K dim
       ┌───────────┬─────────────────────┐
       │           │                     │
   M   │  A tile   │     A (rest)        │
       │  BM × BK  │                     │
       │           │                     │
       └───────────┴─────────────────────┘

       ┌───────────────┐
       │   B tile      │
   K   │   BK × BN     │       N dim
       │               │
       ├───────────────┤
       │  B (rest)     │
       │               │
       └───────────────┘

The block computes:
       ┌───────────┐
   M   │  C tile   │
       │  BM × BN  │
       └───────────┘
        N dim
```

Algorithm:

```
For each output tile (BM × BN):
    Initialize accumulator C_tile to 0 (in registers / shared)
    For each chunk along K (BK at a time):
        Load A's BM × BK chunk into shared memory
        Load B's BK × BN chunk into shared memory
        __syncthreads()
        Compute partial: C_tile += A_smem @ B_smem
        __syncthreads()
    Write C_tile back to global memory
```

Each chunk of A is loaded **once** per output tile — not once per thread. Massive reduction in HBM traffic.

Visualization for BM = BN = BK = 32, with one block computing a 32×32 tile of C:

```
For k_chunk in [0, K, BK):
    All 32×32 = 1024 threads in the block cooperate to load:
        - A[block_row*32 : (block_row+1)*32, k_chunk : k_chunk+32]   into A_smem
        - B[k_chunk : k_chunk+32, block_col*32 : (block_col+1)*32]   into B_smem
    __syncthreads()
    Each thread computes its C[i,j] partial:
        for kk in [0, BK):
            C_acc += A_smem[ty, kk] * B_smem[kk, tx]
    __syncthreads()
```

After the K loop, each thread writes its `C_acc` to global memory.

Bandwidth math: with BM=BN=BK=32, we read each of A and B from HBM `K/BK = K/32` times per block, but only once per element across the K dimension. The total HBM traffic drops by ~32×.

## Steps 4–7 (read Boehm)

- **Step 4 — 1D thread tiling**: each thread computes TM rows of C (e.g., 8 outputs). More register reuse.
- **Step 5 — 2D thread tiling**: each thread computes TM×TN outputs. Even more register reuse.
- **Step 6 — Vectorized loads**: use `float4` to load A and B into shared memory in 16-byte chunks.
- **Step 7 — Warp tiling + double-buffered SMEM**: organize threads into "warp tiles" and prefetch the next K chunk while computing the current one.

By Step 7 you're at ~95% of cuBLAS on Ampere FP32.

## The Hopper / Blackwell story (read, don't reproduce)

For modern hardware the pattern changes:

- **TMA** (Tensor Memory Accelerator) replaces the threads-cooperatively-load pattern with a single instruction. You set up a TMA descriptor on the host, then issue one `cp.async.bulk.tensor` and the hardware does the load asynchronously.
- **WGMMA** (Warp-Group MMA) replaces per-thread MMA with a per-warp-group instruction. Operands can come *directly from shared memory* — no register staging.
- **Warp specialization**: split threads into producer warps (issue TMA) and consumer warps (issue WGMMA). They run concurrently — true compute/memory overlap.
- **Thread block clusters**: groups of 2–16 blocks share data via DSMEM. FA3 and the fastest GEMMs use this.
- **Blackwell tcgen05 + TMEM**: results live in a new on-SM scratchpad called Tensor Memory (256 KB/SM). MMA instructions read/write TMEM rather than registers.

Two excellent 2025-2026 worklogs walk through this on real hardware:
- [Pranjal Shankhdhar — Outperforming cuBLAS on H100](https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog) — 7% faster than cuBLAS, code at https://github.com/pranjalssh/fast.cu
- [Hamza Elshafie — Optimising GEMM on H100 (Jan 2026)](https://hamzaelshafie.bearblog.dev/worklog-optimising-gemm-on-nvidia-h100-for-cublas-like-performance-wip/) — pedagogical
- [Emmanuel Alo — Anatomy of a CUDA GEMM on Blackwell (Apr 2026)](https://medium.com/@emmanuelalo52/anatomy-of-a-cuda-gemm-from-naive-kernels-to-outperforming-cublas-on-blackwell-c394b04b5995)

For learning: do steps 1-3 yourself on whatever GPU you have. Read steps 4-7 in Boehm and the Hopper/Blackwell extensions in the worklogs above. **Don't try to write WGMMA + TMA + warp-specialized kernels in raw CUDA C++ in 2026.** That's CUTLASS/CuTe-DSL territory, covered in the `compiler-and-kernels` track.

## Realistic learner target

- Ampere (A100, T4): hit ~80% of cuBLAS following Boehm's first 5 steps. ~1 day of work.
- Hopper (H100): read the worklogs. Don't try to match cuBLAS in raw CUDA C++ — that's a multi-week project even for experts.
- Blackwell (B200): read Modular's matmul series. Stay in the reading-only mode.

## Pitfalls

1. **Wrong loop order in step 1.** Causes uncoalesced reads on B. Run `ncu` and look at memory throughput before assuming your kernel is "naive but correct."
2. **Forgetting `__syncthreads()` after the load and before the compute.** Some threads start computing while others are still loading. Result is garbage.
3. **Skipping correctness checks.** Compare against `torch.matmul` (or a CPU reference for small sizes). Wrong matmul can produce plausible-looking outputs that aren't correct — don't trust eyeballing.
4. **Reporting FP32 numbers as if they're FP16 numbers.** Tensor cores at FP16/BF16 are ~16× FP32 throughput on Ampere, ~32× on Hopper. State the precision.
5. **Comparing your matmul to cuBLAS at small sizes.** cuBLAS has overhead for very small matmul sizes; you might "beat" it at M=N=K=128 because cuBLAS isn't trying. Compare at M=N=K=4096 or larger.

## What you'll measure

For each kernel: TFLOPS = `2 * M * N * K / time / 1e12`. Plot vs cuBLAS. Boehm-style learning curve.

## References

- **Simon Boehm — How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance** — https://siboehm.com/articles/22/CUDA-MMM (the canonical learning path)
- **Pranjal Shankhdhar — Outperforming cuBLAS on H100** — https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog (code: https://github.com/pranjalssh/fast.cu)
- **Hamza Elshafie — Optimising GEMM on H100 (Jan 2026)** — https://hamzaelshafie.bearblog.dev/worklog-optimising-gemm-on-nvidia-h100-for-cublas-like-performance-wip/
- **Emmanuel Alo — Anatomy of a CUDA GEMM on Blackwell** — https://medium.com/@emmanuelalo52/anatomy-of-a-cuda-gemm-from-naive-kernels-to-outperforming-cublas-on-blackwell-c394b04b5995
- **Colfax — WGMMA on Hopper** — https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/
- **Colfax — Mastering the TMA** — https://research.colfax-intl.com/tutorial-hopper-tma/
- **Colfax — GEMM with Thread Block Clusters on Blackwell** — https://research.colfax-intl.com/cutlass-tutorial-gemm-with-thread-block-clusters-on-nvidia-blackwell-gpus/
- **Modular — Matrix Multiplication on Blackwell** — https://www.modular.com/blog/matrix-multiplication-on-nvidias-blackwell-part-1-introduction
- **Dissecting the NVIDIA Hopper Architecture (arXiv 2501.12084)** — https://arxiv.org/abs/2501.12084
