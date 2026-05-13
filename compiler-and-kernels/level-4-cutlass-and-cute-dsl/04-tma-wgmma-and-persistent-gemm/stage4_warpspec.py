"""
stage4_warpspec.py — warp specialization.

One CTA = 5 warps = 160 threads.
  - Warp 0 (producer warpgroup of size 32 — exception to the standard
    "warpgroup is 4 warps" because the producer only issues TMAs which
    don't need full warpgroup width). Reads "empty" mbarriers, issues
    TMA, signals "full".
  - Warps 1..4 (one consumer warpgroup of 128 threads). Waits on "full",
    runs WGMMA, signals "empty".

Two sets of mbarriers per stage: full[s] (load done) and empty[s] (consumer
released the buffer).

Target: ~80% of cuBLAS on H100 at M=N=K=4096.
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
NUM_STAGES = 3
PRODUCER_WARPS = 1
CONSUMER_WARPS = 4
TOTAL_WARPS = PRODUCER_WARPS + CONSUMER_WARPS


@cute.kernel
def gemm_ws_kernel(mA, mB, mC, tma_a, tma_b,
                   M: cutlass.Int32, N: cutlass.Int32, K: cutlass.Int32):
    bm, bn, _ = cute.arch.block_idx()
    warp_idx = cute.arch.warp_idx()

    sA = cute.make_smem_tensor(shape=(NUM_STAGES, BLOCK_M, BLOCK_K),
                               dtype=cutlass.BFloat16,
                               swizzle_per_stage=cute.Swizzle(3, 4, 3))
    sB = cute.make_smem_tensor(shape=(NUM_STAGES, BLOCK_K, BLOCK_N),
                               dtype=cutlass.BFloat16,
                               swizzle_per_stage=cute.Swizzle(3, 4, 3))

    # full[s]: signaled by producer when load is complete.
    # empty[s]: signaled by consumer when MMA has consumed the buffer.
    full = [cute.make_mbarrier(count=1) for _ in range(NUM_STAGES)]
    empty = [cute.make_mbarrier(count=CONSUMER_WARPS * 32) for _ in range(NUM_STAGES)]

    num_k = K // BLOCK_K

    if warp_idx == 0:
        # Producer.
        phase = [0] * NUM_STAGES
        for k in range(num_k):
            s = k % NUM_STAGES
            if k >= NUM_STAGES:
                cute.arch.mbarrier_wait(empty[s], phase[s])
                phase[s] ^= 1
            if cute.arch.thread_idx()[0] == 0:
                cute.arch.mbarrier_expect_tx(full[s],
                                             BLOCK_M*BLOCK_K*2 + BLOCK_K*BLOCK_N*2)
                cute.copy(tma_a, (bm, k), sA[s], full[s])
                cute.copy(tma_b, (k, bn), sB[s], full[s])
    else:
        # Consumer warpgroup.
        tiled_mma = make_tiled_mma(SM90_64x128x16_F32BF16BF16_SS, atom_layout=(2, 1, 1))
        acc = cute.make_fragment_like(tiled_mma.partition_C(...), dtype=cutlass.Float32)
        acc.fill_(0.0)

        phase = [0] * NUM_STAGES
        for k in range(num_k):
            s = k % NUM_STAGES
            cute.arch.mbarrier_wait(full[s], phase[s])
            phase[s] ^= 1
            cute.gemm(tiled_mma, sA[s], sB[s], acc)
            cute.arch.mbarrier_arrive(empty[s])

        gC_tile = mC[bm*BLOCK_M:(bm+1)*BLOCK_M, bn*BLOCK_N:(bn+1)*BLOCK_N]
        cute.copy(acc.to(cutlass.BFloat16), gC_tile)


@cute.jit
def gemm_ws(mA, mB, mC, M, N, K):
    tma_a = cute.create_tma_atom(SM90_TMA_LOAD, mA, box_shape=(BLOCK_M, BLOCK_K))
    tma_b = cute.create_tma_atom(SM90_TMA_LOAD, mB, box_shape=(BLOCK_K, BLOCK_N))
    grid = (M // BLOCK_M, N // BLOCK_N, 1)
    block = (TOTAL_WARPS * 32, 1, 1)
    gemm_ws_kernel(mA, mB, mC, tma_a, tma_b, M, N, K).launch(grid=grid, block=block)


def main():
    M = N = K = 4096
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    c = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)

    def run():
        gemm_ws(
            cute.make_tensor_from_torch(a),
            cute.make_tensor_from_torch(b),
            cute.make_tensor_from_torch(c),
            M, N, K,
        )
    run()
    verify(c, a, b)
    ms = bench(run)
    report("stage4 +warpspec", M, N, K, ms)


if __name__ == "__main__":
    main()
