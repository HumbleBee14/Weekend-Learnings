# Walkthrough 01 — Single-SM tcgen05.mma

> Source: [`cutlass/examples/cute/tutorial/blackwell/01_mma_sm100.cu`](https://github.com/NVIDIA/cutlass/blob/main/examples/cute/tutorial/blackwell/01_mma_sm100.cu)

The minimal Blackwell MMA kernel. One CTA, one MMA tile, no pipelining, no cluster. Read for the pattern; skim the boilerplate.

## What the kernel does

`C = A @ B` where A is `(M, K)` BF16, B is `(K, N)` BF16, C is `(M, N)` FP32. M=128, N=256, K=16 — one MMA tile, exactly. No loop over K. This isolates the MMA mechanics.

## The shape of the kernel

```cpp
__global__ void mma_kernel(...) {
  // 1. Allocate TMEM for the accumulator
  __shared__ struct { ... uint32_t tmem_base; ... } shared_storage;
  cute::TMEM::Allocator1Sm tmem_allocator{};
  if (elect_one_warp_in_cta()) {
    tmem_allocator.allocate(NUM_COLUMNS, &shared_storage.tmem_base);
  }
  __syncthreads();

  // 2. Build accumulator tensor view over TMEM
  Tensor tCtAcc = make_tensor(
    make_tmem_ptr(shared_storage.tmem_base),
    Layout<Shape<...>, Stride<...>>{}    // the (lane<<16 | col) layout
  );

  // 3. Load A and B into SMEM via TMA (omitted in 01; uses synchronous copy)
  // ...

  // 4. Issue the MMA — one thread, on behalf of the CTA
  TiledMMA tiled_mma = make_tiled_mma(SM100_MMA_F16BF16_SS{});
  if (elect_one_warp_in_cta() && elect_one_thread()) {
    gemm(tiled_mma, tCrA, tCrB, tCtAcc);   // lowers to tcgen05.mma
  }

  // 5. Signal MMA completion to an mbarrier
  cutlass::arch::umma_arrive(&shared_storage.mma_barrier);

  // 6. Wait for the MMA to complete (warpgroup-wide)
  cutlass::arch::umma_wait(&shared_storage.mma_barrier);

  // 7. Copy TMEM → registers (warpgroup-wide)
  TiledCopy tiled_t2r = make_tmem_copy(SM100_TMEM_LOAD_32dp32b1x{}, tCtAcc);
  copy(tiled_t2r, tCtAcc, tDrAcc);

  // 8. Convert FP32 → BF16, store to GMEM
  // ...

  // 9. Deallocate TMEM
  if (elect_one_warp_in_cta()) {
    tmem_allocator.free(shared_storage.tmem_base, NUM_COLUMNS);
  }
}
```

## Things to internalize

**The TMEM allocation is column-granular and CTA-local.** One warp elects to call `allocate`; the resulting base pointer is in shared memory and broadcast to all warps via `__syncthreads()`. The minimum allocation is 32 columns.

**The accumulator's CuTe layout is the address layout.** Look at the layout stride: `((128, 256), 1, 1) : ((65536, 1), 0, 0)`. The `65536 = 1 << 16` is the lane-to-lane jump in TMEM's `(lane << 16 | col)` address space. The CuTe layout describes the address pattern, not an array index pattern — same algebra, hardware-specific stride.

**`tcgen05.mma` is single-thread.** `elect_one_thread()` picks one thread in the elected warp. That thread issues the MMA. Every other thread does nothing at that line. This is the central simplification vs WGMMA.

**`umma_arrive` and `umma_wait` are the MMA's barrier protocol.** The MMA is asynchronous — issue returns immediately. `umma_arrive` is the completion signal; `umma_wait` blocks until the signal arrives. In a real kernel you'd do other work between issue and wait (load the next tile, run an unrelated warp's job).

**`tcgen05.ld` requires a warpgroup.** `make_tmem_copy` is hardcoded to 4 warps because TMEM has 128 lanes and one warp can only access 32 of them. The `SM100_TMEM_LOAD_32dp32b1x` atom selects the per-thread fragment layout — there are multiple variants for different epilogue patterns; this one matches the canonical 32-column-per-warp slice.

**Deallocate, always.** `tmem_allocator.free()` is mandatory. CuTe-DSL's allocator helpers do this automatically in normal control flow; if you go through inline PTX you do it by hand.

## Things to ignore for now

The `umma_arrive_to_mma_complete` / `umma_arrive_to_smem_release` distinction (matters for pipelined kernels with cluster, irrelevant here). The `ScaleOut::Zero` / `ScaleOut::One` toggle (first MMA initializes accumulator, subsequent MMAs accumulate; this kernel only has one).

## After reading

You should be able to point to the 6 lines that are Blackwell-specific (TMEM alloc, TMEM accumulator tensor view, single-thread MMA issue, umma_arrive, tcgen05.ld via TiledCopy, TMEM dealloc). The rest is the same shape as any CuTe GEMM.
