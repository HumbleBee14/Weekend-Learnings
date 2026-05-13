# Walkthrough 03 — Persistent dense GEMM on SM100 (CuTe-DSL Python)

> Source: [`cutlass/examples/python/CuTeDSL/blackwell/dense_gemm_persistent.py`](https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/blackwell/dense_gemm_persistent.py)

The Python equivalent of submodule 04's stage 5, but on SM100. Open both files side-by-side; the deltas are illuminating.

## What stays the same

- The persistent grid (one CTA per SM, internal tile loop).
- Warp specialization (producer warp(s) for TMA, consumer warps for MMA + epilogue).
- The multi-stage SMEM pipeline (3 or more stages) for A and B operand loads.
- The mbarrier protocol between producer and consumer.
- The CuTe layout algebra for SMEM tiles, descriptors, and accumulator views.

## What changes (SM90 → SM100)

### 1. The MMA atom

```python
# SM90:
from cutlass.cute.nvgpu.warpgroup import SM90_64x128x16_F32BF16BF16_SS
tiled_mma = make_tiled_mma(SM90_64x128x16_F32BF16BF16_SS, atom_layout=(2, 1, 1))

# SM100:
from cutlass.cute.nvgpu.tcgen05 import SM100_MMA_F16BF16_SS
tiled_mma = make_tiled_mma(SM100_MMA_F16BF16_SS, ...)
```

The SM100 atom's accumulator handle is a TMEM pointer, not a register fragment.

### 2. TMEM allocation in the kernel prologue

```python
# Only on SM100:
tmem_base = cute.tcgen05.alloc(num_columns=NUM_ACC_COLS)
acc = cute.make_tmem_tensor(tmem_base, acc_layout)   # the (lane<<16 | col) layout
acc.fill_(0.0)   # this writes zeros via tcgen05.st
```

There's no equivalent on SM90 — the accumulator was just a register fragment, no allocation needed.

### 3. MMA issue is warp-elected

```python
# SM90 consumer warpgroup:
cute.gemm(tiled_mma, sA[s], sB[s], acc)   # all 128 threads participate

# SM100 MMA warp (one warp out of the consumer warps):
if cute.arch.warp_idx() == MMA_WARP:
    if cute.arch.thread_idx()[0] % 32 == 0:    # elect one thread
        cute.gemm(tiled_mma, sA[s], sB[s], acc)
```

The single-thread issue means the MMA warp is mostly idle. That's fine — it's free for other work (often: managing the next iteration's bookkeeping).

### 4. The epilogue brings TMEM into registers

```python
# SM100 epilogue (warpgroup-wide):
cute.arch.tcgen05_wait_for_mma_complete()         # umma_wait
acc_regs = cute.make_fragment_like(acc_register_layout, dtype=cutlass.Float32)
cute.copy(tmem_to_reg_atom, acc, acc_regs)        # tcgen05.ld
# convert + store
acc_bf16 = acc_regs.to(cutlass.BFloat16)
cute.copy(acc_bf16, gC_tile)
```

The `tmem_to_reg_atom` is built with `make_tmem_copy(SM100_TMEM_LOAD_32dp32b1x{}, acc)` and partitions the 128-row TMEM slab across the 128 threads of the consumer warpgroup.

### 5. TMEM deallocate before exit

```python
cute.tcgen05.dealloc(tmem_base, num_columns=NUM_ACC_COLS)
```

If you forget this, subsequent kernel launches on the same SM will fail with a TMEM allocation error.

### 6. Optional: 2-SM cluster

```python
@cute.jit
def gemm_persistent_sm100(...):
    # ...
    gemm_persistent_kernel(...).launch(
        grid=(num_sms,),
        block=(TOTAL_WARPS * 32,),
        cluster=(2, 1, 1) if use_2sm else (1, 1, 1),
    )
```

With `cluster=(2,1,1)` the MMA atom is the `_2SM_` variant; the kernel uses `block_rank_in_cluster()` to distinguish leader from peer.

## Numbers to expect on B200

- Single-SM persistent BF16 at M=N=K=4096: ~75–80% of cuBLAS.
- 2-SM cooperative BF16 at the same shape: ~85–90% of cuBLAS.
- NVFP4 (different kernel — see [`dense_blockscaled_gemm_persistent.py`](https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/blackwell/dense_blockscaled_gemm_persistent.py)): ~85% of cuBLAS NVFP4, which itself is ~2× BF16 cuBLAS TFLOPS on Blackwell.

## Reading order

1. Read `walkthrough_01` and the matching `01_mma_sm100.cu`. Internalize: TMEM alloc, single-thread MMA, warpgroup-wide tcgen05.ld, dealloc.
2. Read `walkthrough_02` and `04_mma_tma_2sm_sm100.cu`. Internalize: cluster rank, TMA multicast, leader-only MMA issue.
3. Read this walkthrough and the Python `dense_gemm_persistent.py`. Open submodule 04's `stage5_persistent.py` next to it and diff in your head.

After all three, open FlashAttention-4's CuTe-DSL source and Modal's writeup. Most of it will look familiar — TMA, mbarriers, warp specialization, TMEM. The novel parts (cubic-polynomial softmax, five warp roles instead of two, correction-rescale pattern) are the FA4-specific contributions.
