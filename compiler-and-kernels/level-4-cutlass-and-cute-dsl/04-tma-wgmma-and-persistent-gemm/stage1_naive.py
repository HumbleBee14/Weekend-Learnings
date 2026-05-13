"""
stage1_naive.py — naive tiled BF16 GEMM in CuTe-DSL on Hopper (SM90).

Structure:
  - One CTA per output tile (BLOCK_M x BLOCK_N).
  - One warpgroup (4 warps, 128 threads) — WGMMA needs the full group.
  - Synchronous cute.copy GMEM -> SMEM each K iteration. No TMA, no pipelining.
  - WGMMA atom: SM90_64x128x16_F32BF16BF16_SS.

Target: ~30% of cuBLAS on H100 at M=N=K=4096.

Bottleneck: the WGMMA stalls waiting for SMEM load to finish. No overlap.
Stage 2 fixes that with TMA; stage 3 with multi-stage pipelining.

This file is annotated heavily because it's where you meet the WGMMA atom,
the SMEM layout, and the cute.gemm call for the first time.
"""

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu.warpgroup import (
    make_tiled_mma,
    OperandMajorMode,
    SM90_64x128x16_F32BF16BF16_SS,
)
from _helpers import bench, verify, report


BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 64
NUM_WARPS = 4         # one warpgroup


@cute.kernel
def gemm_naive_kernel(
    mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor,
    M: cutlass.Int32, N: cutlass.Int32, K: cutlass.Int32,
):
    bm, bn, _ = cute.arch.block_idx()

    # SMEM tiles. Swizzle for BF16 64-wide rows.
    sA = cute.make_smem_tensor(
        shape=(BLOCK_M, BLOCK_K), dtype=cutlass.BFloat16,
        swizzle=cute.Swizzle(3, 4, 3),
    )
    sB = cute.make_smem_tensor(
        shape=(BLOCK_K, BLOCK_N), dtype=cutlass.BFloat16,
        swizzle=cute.Swizzle(3, 4, 3),
    )

    # Accumulator in registers; FP32.
    tiled_mma = make_tiled_mma(
        SM90_64x128x16_F32BF16BF16_SS,
        atom_layout=(2, 1, 1),  # 2 atoms in M, 1 in N: 128x128 output per CTA
    )
    acc = cute.make_fragment_like(tiled_mma.partition_C(...), dtype=cutlass.Float32)
    acc.fill_(0.0)

    num_k_tiles = K // BLOCK_K
    for k in range(num_k_tiles):
        # GMEM -> SMEM (synchronous; the warpgroup pulls the tile in)
        gA_tile = mA[bm*BLOCK_M:(bm+1)*BLOCK_M, k*BLOCK_K:(k+1)*BLOCK_K]
        gB_tile = mB[k*BLOCK_K:(k+1)*BLOCK_K, bn*BLOCK_N:(bn+1)*BLOCK_N]
        cute.copy(gA_tile, sA)
        cute.copy(gB_tile, sB)
        cute.arch.barrier()

        # WGMMA: acc += sA @ sB
        cute.gemm(tiled_mma, sA, sB, acc)

    # Epilogue: write FP32 accumulator -> BF16 GMEM
    gC_tile = mC[bm*BLOCK_M:(bm+1)*BLOCK_M, bn*BLOCK_N:(bn+1)*BLOCK_N]
    cute.copy(acc.to(cutlass.BFloat16), gC_tile)


@cute.jit
def gemm_naive(mA, mB, mC, M, N, K):
    grid = (M // BLOCK_M, N // BLOCK_N, 1)
    block = (NUM_WARPS * 32, 1, 1)
    gemm_naive_kernel(mA, mB, mC, M, N, K).launch(grid=grid, block=block)


def main():
    M = N = K = 4096
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    c = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)

    def run():
        gemm_naive(
            cute.make_tensor_from_torch(a),
            cute.make_tensor_from_torch(b),
            cute.make_tensor_from_torch(c),
            M, N, K,
        )
    run()
    verify(c, a, b)
    ms = bench(run)
    report("stage1 naive", M, N, K, ms)


if __name__ == "__main__":
    main()
