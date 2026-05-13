"""
stage3_multistage.py — add a 3-stage SMEM pipeline.

While the WGMMA consumes tile k, the TMA loads tile k+1, and tile k+2 is
staged behind it. Three mbarriers track per-stage load completion.

Target: ~70% of cuBLAS on H100 at M=N=K=4096.

The pattern (every Hopper GEMM uses this):

    NUM_STAGES = 3
    for s in range(NUM_STAGES):
        issue_load(s)                   # prologue: fill the pipe

    for k in range(NUM_STAGES, num_k):
        wait(k - NUM_STAGES)            # tile k-NUM_STAGES is ready
        compute(k - NUM_STAGES)
        if k < num_k: issue_load(k)     # keep the pipe full

    for k in range(num_k, num_k+NUM_STAGES):
        wait(k - NUM_STAGES)
        compute(k - NUM_STAGES)          # drain
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
NUM_STAGES = 3


@cute.kernel
def gemm_multistage_kernel(mA, mB, mC, tma_a, tma_b,
                           M: cutlass.Int32, N: cutlass.Int32, K: cutlass.Int32):
    bm, bn, _ = cute.arch.block_idx()

    sA = cute.make_smem_tensor(
        shape=(NUM_STAGES, BLOCK_M, BLOCK_K),
        dtype=cutlass.BFloat16,
        swizzle_per_stage=cute.Swizzle(3, 4, 3),
    )
    sB = cute.make_smem_tensor(
        shape=(NUM_STAGES, BLOCK_K, BLOCK_N),
        dtype=cutlass.BFloat16,
        swizzle_per_stage=cute.Swizzle(3, 4, 3),
    )

    mbars = [cute.make_mbarrier(count=1) for _ in range(NUM_STAGES)]

    tiled_mma = make_tiled_mma(SM90_64x128x16_F32BF16BF16_SS, atom_layout=(2, 1, 1))
    acc = cute.make_fragment_like(tiled_mma.partition_C(...), dtype=cutlass.Float32)
    acc.fill_(0.0)

    num_k = K // BLOCK_K

    # Prologue.
    for s in range(NUM_STAGES):
        if cute.arch.thread_idx()[0] == 0:
            cute.arch.mbarrier_expect_tx(mbars[s], BLOCK_M*BLOCK_K*2 + BLOCK_K*BLOCK_N*2)
            cute.copy(tma_a, (bm, s), sA[s], mbars[s])
            cute.copy(tma_b, (s, bn), sB[s], mbars[s])

    phase = [0] * NUM_STAGES

    # Steady state + drain (single loop).
    for k in range(num_k):
        s_consume = k % NUM_STAGES
        cute.arch.mbarrier_wait(mbars[s_consume], phase[s_consume])
        cute.gemm(tiled_mma, sA[s_consume], sB[s_consume], acc)
        phase[s_consume] ^= 1

        # Issue next load if any remain.
        k_next = k + NUM_STAGES
        if k_next < num_k and cute.arch.thread_idx()[0] == 0:
            cute.arch.mbarrier_expect_tx(mbars[s_consume],
                                         BLOCK_M*BLOCK_K*2 + BLOCK_K*BLOCK_N*2)
            cute.copy(tma_a, (bm, k_next), sA[s_consume], mbars[s_consume])
            cute.copy(tma_b, (k_next, bn), sB[s_consume], mbars[s_consume])

    gC_tile = mC[bm*BLOCK_M:(bm+1)*BLOCK_M, bn*BLOCK_N:(bn+1)*BLOCK_N]
    cute.copy(acc.to(cutlass.BFloat16), gC_tile)


@cute.jit
def gemm_multistage(mA, mB, mC, M, N, K):
    tma_a = cute.create_tma_atom(SM90_TMA_LOAD, mA, box_shape=(BLOCK_M, BLOCK_K))
    tma_b = cute.create_tma_atom(SM90_TMA_LOAD, mB, box_shape=(BLOCK_K, BLOCK_N))
    grid = (M // BLOCK_M, N // BLOCK_N, 1)
    block = (NUM_WARPS * 32, 1, 1)
    gemm_multistage_kernel(mA, mB, mC, tma_a, tma_b, M, N, K).launch(grid=grid, block=block)


def main():
    M = N = K = 4096
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    c = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)

    def run():
        gemm_multistage(
            cute.make_tensor_from_torch(a),
            cute.make_tensor_from_torch(b),
            cute.make_tensor_from_torch(c),
            M, N, K,
        )
    run()
    verify(c, a, b)
    ms = bench(run)
    report("stage3 +multi-stage", M, N, K, ms)


if __name__ == "__main__":
    main()
