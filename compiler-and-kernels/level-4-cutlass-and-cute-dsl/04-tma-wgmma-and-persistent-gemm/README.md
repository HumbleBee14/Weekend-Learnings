# 04 — TMA, WGMMA, and the persistent BF16 GEMM on SM90

> Outer: [`../README.md`](../README.md) · Hardware: H100 ideal. A100 reaches stage 3. T4 too small for these instructions.

This is the level's hands-on centerpiece. You build a BF16 persistent GEMM on Hopper in five stages, watching achieved TFLOPS climb from ~30% of cuBLAS to ~85–90%. Same operation, same dtypes, same M=N=K=4096 shape. Each stage adds one hardware-aware idea.

The structure mirrors Level 1's RMSNorm bandwidth journey: the *number* moves stage by stage, and at every stage you can point to the hardware mechanism that earned the gain.

## Stages

| Stage | Adds | Target | File |
|---|---|---|---|
| 1 | Plain tiled GEMM, `cute.copy`, WGMMA atom on registers | ~30% cuBLAS | [`stage1_naive.py`](stage1_naive.py) |
| 2 | TMA (`cp.async.bulk.tensor`) for A and B loads | ~55% cuBLAS | [`stage2_tma.py`](stage2_tma.py) |
| 3 | Multi-stage SMEM pipeline (3 buffers, overlap load + MMA) | ~70% cuBLAS | [`stage3_multistage.py`](stage3_multistage.py) |
| 4 | Warp specialization (producer + consumer warp groups) | ~80% cuBLAS | [`stage4_warpspec.py`](stage4_warpspec.py) |
| 5 | Persistent grid (one CTA per SM, loop over tiles internally) | ~85–90% cuBLAS | [`stage5_persistent.py`](stage5_persistent.py) |

```
% of cuBLAS  (M=N=K=4096 BF16 on H100)

100% ┤                                                  ── cuBLAS reference
 90% ┤                                          ██████  stage 5  persistent grid
 85% ┤                                  ██████          stage 4  warp-spec
 80% ┤                          ██████                  
 70% ┤                  ██████                          stage 3  multi-stage
 60% ┤                                                  
 55% ┤          ██████                                  stage 2  TMA descriptors
 40% ┤                                                  
 30% ┤  ██████                                          stage 1  naive tiled
     └──┴──────┴──────┴──────┴──────┴──────┴──────
        1      2      3      4      5
              one hardware idea added per stage
```

*Each stage adds exactly one hardware-aware mechanism; the percent-of-cuBLAS number is what you watch climb.*

The persistent stage-5 kernel is also the starting point for the capstone in `_capstone-bf16-persistent-gemm/`.

## Stage 1 — naive tiled GEMM

Plain BLOCK_M × BLOCK_N output tile per CTA. Each CTA: loop over K, load BLOCK_K slabs of A and B into SMEM with `cute.copy` (synchronous), call `cute.gemm(tiled_mma, sA, sB, accumulator)`, repeat. After the K loop, write accumulator to GMEM.

Key choices:
- `BLOCK_M=128`, `BLOCK_N=128`, `BLOCK_K=64`. These are the natural tile sizes for the `SM90_64x128x16_F32BF16BF16_SS` WGMMA atom (one warpgroup, 64×128 output per MMA).
- `num_warps=4` per CTA. One warpgroup. WGMMA needs full warpgroup participation.
- SMEM: 2 tiles (one A, one B), no double buffering. `BLOCK_M*BLOCK_K*2 + BLOCK_N*BLOCK_K*2 = 16384 + 16384 = 32KB`.

Why it's slow: the WGMMA stalls waiting for the SMEM load to complete. There's no overlap. You're using a Hopper tensor core like an Ampere one.

Read the file. Note the structure of the K loop. This is what the next four stages improve.

## Stage 2 — TMA descriptors

Replace the synchronous `cute.copy` with a TMA descriptor and `cp.async.bulk.tensor`. The descriptor encodes shape, stride, box size, and swizzle. One instruction issues a multi-KB tile copy. The issuing thread does not wait.

```python
# Build descriptors once (host-side).
tma_a = cute.create_tma_atom(
    SM90_TMA_LOAD,
    gA,                                  # global tensor
    sA_layout,                           # SMEM layout (with swizzle)
    box_shape=(BLOCK_M, BLOCK_K),
)
tma_b = cute.create_tma_atom(
    SM90_TMA_LOAD,
    gB,
    sB_layout,
    box_shape=(BLOCK_K, BLOCK_N),
)

# Inside the K loop:
cute.copy(tma_a, gA_tile, sA, mbar)
cute.copy(tma_b, gB_tile, sB, mbar)
cute.arch.mbarrier_wait(mbar, phase)     # wait for both tiles
cute.gemm(tiled_mma, sA, sB, acc)
```

The `mbar` is an mbarrier — a hardware-backed barrier that the TMA signals on completion. You wait on the mbarrier before reading the SMEM.

What you should see:
- DRAM traffic per tile drops (one big transaction instead of many small ones).
- Bank conflicts disappear if your SMEM layout has the right swizzle for the box (use `Swizzle(3,4,3)` for 64-wide BF16 rows).
- The kernel still serializes load → wait → MMA per K iteration. Stage 3 fixes that.

## Stage 3 — multi-stage SMEM pipeline

Allocate 3 SMEM buffers per operand (configurable; 3 is the H100 sweet spot). While the MMA computes on tile *k*, the TMA loads tile *k+1*, and tile *k+2* is staged behind it. Three barriers track which buffer is loaded vs consumed.

```python
NUM_STAGES = 3

sA = cute.make_smem_tensor(shape=(NUM_STAGES, BLOCK_M, BLOCK_K), ...)
sB = cute.make_smem_tensor(shape=(NUM_STAGES, BLOCK_K, BLOCK_N), ...)

# Prologue: load first NUM_STAGES tiles.
for s in range(NUM_STAGES):
    issue_tma(s)

# Steady state: each iteration issues a new load and waits for the oldest.
for k in range(NUM_STAGES, num_k_tiles + NUM_STAGES):
    if k < num_k_tiles:
        issue_tma(k % NUM_STAGES)
    wait_for_stage((k - NUM_STAGES) % NUM_STAGES)
    cute.gemm(tiled_mma, sA[(k - NUM_STAGES) % NUM_STAGES], sB[...], acc)
```

This is the standard pipelined mainloop. The same pattern is in every Hopper GEMM in CUTLASS. The win: TMA and WGMMA run concurrently, limited only by the slower of the two.

What you should see:
- Tensor core utilization (visible in `ncu`) climbs noticeably.
- The kernel is now compute-bound for large-K shapes.
- For small-K shapes (decode), the prologue/epilogue dominate and the gain is smaller. Stage 5 helps here.

## Stage 4 — warp specialization

Stages 1–3 used one warpgroup. Stage 4 uses two: a **producer** warpgroup (1 warp, 32 threads) that *only* issues TMA, and a **consumer** warpgroup (4 warps, 128 threads) that *only* runs WGMMA. The producer's TMA throughput and the consumer's WGMMA throughput overlap perfectly, with mbarriers coordinating.

In CuTe-DSL this is a warp-role pattern inside the kernel:

```python
@cute.kernel
def gemm_warpspec_kernel(...):
    warp_idx = cute.arch.warp_idx()
    if warp_idx == 0:
        # Producer: issue TMAs all K iterations
        for k in range(num_k_tiles):
            stage = k % NUM_STAGES
            wait_for_empty(stage)
            issue_tma(stage)
    else:
        # Consumer: warps 1..4, run WGMMA
        for k in range(num_k_tiles):
            stage = k % NUM_STAGES
            wait_for_full(stage)
            cute.gemm(tiled_mma, sA[stage], sB[stage], acc)
            signal_empty(stage)
    # Epilogue: consumer warpgroup writes
```

The producer is 32 threads doing nothing but issuing TMAs and waiting on empty-buffer mbarriers. The consumer is 128 threads running WGMMA back-to-back. Each warpgroup is full at all times.

What you should see:
- WGMMA tensor-core utilization approaches peak.
- Producer warpgroup occupancy is low (it spends most of its time waiting) — that's correct.
- The kernel is now compute-bound except for the prologue/epilogue.

You can also do **ping-pong** (two consumer warpgroups alternating on output tiles) — this is the [PyTorch CUTLASS ping-pong blog post](https://pytorch.org/blog/cutlass-ping-pong-gemm-kernel/). For BF16 dense GEMM the gain over single-consumer + persistent is small; the pattern matters more for FP8 where the MMA is faster.

## Stage 5 — persistent grid

Launch `grid = (num_SMs,)` instead of `(M/BLOCK_M, N/BLOCK_N)`. Each persistent CTA loops over multiple output tiles internally, picking the next tile from a precomputed schedule or via an atomic counter. The hardware scheduler never has to dispatch a new CTA — the kernel does its own scheduling.

```python
@cute.kernel
def gemm_persistent_kernel(...):
    pid = cute.arch.block_idx()[0]                  # 0..num_SMs-1
    num_tiles = (M // BLOCK_M) * (N // BLOCK_N)
    for t in range(pid, num_tiles, num_SMs):
        m_tile = t // (N // BLOCK_N)
        n_tile = t % (N // BLOCK_N)
        # ... K loop, same as stage 4 ...
        # Write output tile (m_tile, n_tile).
```

Wins:
- **CUDA-graph compatibility.** Grid size is fixed for the device; you capture once and replay for any M, N (the K loop bound is dynamic, K loop is internal).
- **No launch overhead between output tiles** — the CTA stays resident, the K loop and accumulator allocation are re-initialized cheaply.
- **Better L2 cache utilization** if you schedule tiles in a Hilbert curve or row-major order rather than the hardware's default raster.

What you should see:
- 4096³ BF16 GEMM at >85% of cuBLAS on H100.
- Decode-shape GEMMs (M=1, M=8) at 2–5× the non-persistent version because launch overhead disappears.

## Build steps

```bash
# Stage by stage. Each file is self-contained.
python stage1_naive.py        # measures TFLOPS, prints alongside cuBLAS
python stage2_tma.py
python stage3_multistage.py
python stage4_warpspec.py
python stage5_persistent.py
```

Each file:
1. Builds the kernel for one specific shape (M=N=K=4096, BF16) so it compiles fast.
2. Runs correctness check against `torch.matmul`.
3. Runs `do_bench`-style measurement (25 warmups, 100 iters).
4. Prints achieved TFLOPS and percent-of-cuBLAS.

The kernels share a `_helpers.py` with cuBLAS reference and benchmarking. The actual kernel code in each stage is the part that changes.

## What you should read alongside the code

- [`cutlass/examples/python/CuTeDSL/hopper/dense_gemm.py`](https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/hopper/dense_gemm.py) — the basic CuTe-DSL Hopper GEMM. Maps to stages 1–2.
- [`cutlass/examples/python/CuTeDSL/hopper/dense_gemm_persistent.py`](https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/hopper/dense_gemm_persistent.py) — the persistent version. Maps to stage 5.
- [Colfax: GEMM Kernel Design and Pipelining](https://research.colfax-intl.com/cutlass-tutorial-design-of-a-gemm-kernel/) — the conceptual walkthrough, written in C++ but the patterns transfer.
- [Colfax: WGMMA on Hopper](https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/) — the WGMMA atom in detail.
- [Colfax: Hopper TMA](https://research.colfax-intl.com/tutorial-hopper-tma/) — TMA descriptor construction.
- [PyTorch: CUTLASS Ping-Pong GEMM Kernel](https://pytorch.org/blog/cutlass-ping-pong-gemm-kernel/) — the two-consumer pattern.

## What you should be able to do next

- Explain why each stage moved the number, in one sentence per stage.
- Identify the corresponding line in `cutlass/examples/python/CuTeDSL/hopper/dense_gemm_persistent.py` for each technique.
- Predict (within a factor of 2) what the speedup will be for a different shape or dtype.
- Write your own kernel for a non-square shape (e.g. M=8192, K=4096, N=12288 — LLaMA FFN-1) that hits >70% of cuBLAS without changes to the kernel beyond shape constants.

The capstone takes stage 5 and tunes it.

## Hardware-by-hardware reality

- **H100 SXM/PCIe.** All five stages work. Stage 5 should land 85–90% of cuBLAS at 4096³ BF16.
- **A100.** Stages 1–3 work (no WGMMA, no TMA). The kernel uses `cp.async` and HMMA atoms. Stages 4–5 in their pure form depend on Hopper features. There is an A100 fallback path that uses `cp.async` and HMMA — the [`ampere/`](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL) examples cover it; the conceptual progression is the same.
- **B200.** All stages work in concept but use `tcgen05.mma` and TMEM accumulators. See submodule 05. The Blackwell version of [`dense_gemm_persistent.py`](https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/blackwell/dense_gemm_persistent.py) is the reference.
- **T4 / V100 / Ampere SM86 (3090, 4090).** Out of scope; these tensor cores predate cp.async or have very different scheduling. Use Triton.
