"""
stage2_tma.py — add TMA descriptors for A and B loads.

Difference from stage 1: cute.copy(gA, sA) -> TMA load via cp.async.bulk.tensor.
The issuing thread does not wait; an mbarrier signals completion.

Target: ~55% of cuBLAS on H100 at M=N=K=4096.

The TMA descriptor packs:
  - global tensor base + strides
  - element type (BF16)
  - box shape (BLOCK_M x BLOCK_K)
  - swizzle (128B for 64 BF16 wide)
into a 128-byte structure created once on the host, then issued by one
device-side instruction per tile load.

This file is a delta over stage 1 — only the K loop changes.
"""

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu.warpgroup import (
    make_tiled_mma, SM90_64x128x16_F32BF16BF16_SS,
)
from cutlass.cute.nvgpu.cp_async import SM90_TMA_LOAD
from _helpers import bench, verify, report


BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 64
NUM_WARPS = 4


@cute.kernel
def gemm_tma_kernel(
    mA, mB, mC,
    tma_a, tma_b,
    M: cutlass.Int32, N: cutlass.Int32, K: cutlass.Int32,
):
    bm, bn, _ = cute.arch.block_idx()

    sA_layout = cute.make_layout(
        (BLOCK_M, BLOCK_K),
        swizzle=cute.Swizzle(3, 4, 3),
    )
    sB_layout = cute.make_layout(
        (BLOCK_K, BLOCK_N),
        swizzle=cute.Swizzle(3, 4, 3),
    )
    sA = cute.make_smem_tensor(layout=sA_layout, dtype=cutlass.BFloat16)
    sB = cute.make_smem_tensor(layout=sB_layout, dtype=cutlass.BFloat16)

    # One mbarrier per operand. Arrival count = 1 (the TMA itself).
    mbar = cute.make_mbarrier(count=1)

    tiled_mma = make_tiled_mma(
        SM90_64x128x16_F32BF16BF16_SS, atom_layout=(2, 1, 1),
    )
    acc = cute.make_fragment_like(tiled_mma.partition_C(...), dtype=cutlass.Float32)
    acc.fill_(0.0)

    num_k_tiles = K // BLOCK_K
    phase = 0
    for k in range(num_k_tiles):
        # Issue both TMAs from a single thread; bytes-expect on the mbarrier.
        if cute.arch.thread_idx()[0] == 0:
            cute.arch.mbarrier_expect_tx(mbar, BLOCK_M * BLOCK_K * 2 + BLOCK_K * BLOCK_N * 2)
            cute.copy(tma_a, (bm, k), sA, mbar)
            cute.copy(tma_b, (k, bn), sB, mbar)
        cute.arch.mbarrier_wait(mbar, phase)
        phase ^= 1

        cute.gemm(tiled_mma, sA, sB, acc)

    gC_tile = mC[bm*BLOCK_M:(bm+1)*BLOCK_M, bn*BLOCK_N:(bn+1)*BLOCK_N]
    cute.copy(acc.to(cutlass.BFloat16), gC_tile)


@cute.jit
def gemm_tma(mA, mB, mC, M, N, K):
    tma_a = cute.create_tma_atom(SM90_TMA_LOAD, mA, box_shape=(BLOCK_M, BLOCK_K))
    tma_b = cute.create_tma_atom(SM90_TMA_LOAD, mB, box_shape=(BLOCK_K, BLOCK_N))
    grid = (M // BLOCK_M, N // BLOCK_N, 1)
    block = (NUM_WARPS * 32, 1, 1)
    gemm_tma_kernel(mA, mB, mC, tma_a, tma_b, M, N, K).launch(grid=grid, block=block)


def main():
    M = N = K = 4096
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    c = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)

    def run():
        gemm_tma(
            cute.make_tensor_from_torch(a),
            cute.make_tensor_from_torch(b),
            cute.make_tensor_from_torch(c),
            M, N, K,
        )
    run()
    verify(c, a, b)
    ms = bench(run)
    report("stage2 +TMA", M, N, K, ms)


if __name__ == "__main__":
    main()
