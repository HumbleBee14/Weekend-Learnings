"""
stage5_persistent.py — persistent grid.

Launch grid = (num_SMs,). Each CTA loops over multiple (m_tile, n_tile)
output tiles internally, picking the next via a precomputed schedule.
The CTA is "persistent" — it stays resident on its SM for the entire
kernel.

Wins:
  - CUDA-graph compatible (grid size is fixed).
  - No launch overhead between output tiles.
  - Better L2 utilization via custom tile-order (rasterized vs Hilbert).

Target: ~85-90% of cuBLAS on H100 at M=N=K=4096.

For decode-shape (M=1..8) GEMMs the win vs non-persistent is much larger
(~2-5x) because launch overhead disappears.
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
def gemm_persistent_kernel(mA, mB, mC, tma_a, tma_b,
                           M: cutlass.Int32, N: cutlass.Int32, K: cutlass.Int32,
                           num_sms: cutlass.Constexpr[int]):
    pid = cute.arch.block_idx()[0]
    warp_idx = cute.arch.warp_idx()

    sA = cute.make_smem_tensor(shape=(NUM_STAGES, BLOCK_M, BLOCK_K),
                               dtype=cutlass.BFloat16,
                               swizzle_per_stage=cute.Swizzle(3, 4, 3))
    sB = cute.make_smem_tensor(shape=(NUM_STAGES, BLOCK_K, BLOCK_N),
                               dtype=cutlass.BFloat16,
                               swizzle_per_stage=cute.Swizzle(3, 4, 3))
    full = [cute.make_mbarrier(count=1) for _ in range(NUM_STAGES)]
    empty = [cute.make_mbarrier(count=CONSUMER_WARPS * 32) for _ in range(NUM_STAGES)]

    tiles_m = M // BLOCK_M
    tiles_n = N // BLOCK_N
    num_tiles = tiles_m * tiles_n
    num_k = K // BLOCK_K

    if warp_idx == 0:
        # Producer.
        phase = [0] * NUM_STAGES
        for t in range(pid, num_tiles, num_sms):
            bm = t // tiles_n
            bn = t % tiles_n
            for k in range(num_k):
                s = (t * num_k + k) % NUM_STAGES
                if (t * num_k + k) >= NUM_STAGES:
                    cute.arch.mbarrier_wait(empty[s], phase[s])
                    phase[s] ^= 1
                if cute.arch.thread_idx()[0] == 0:
                    cute.arch.mbarrier_expect_tx(
                        full[s], BLOCK_M*BLOCK_K*2 + BLOCK_K*BLOCK_N*2
                    )
                    cute.copy(tma_a, (bm, k), sA[s], full[s])
                    cute.copy(tma_b, (k, bn), sB[s], full[s])
    else:
        # Consumer.
        tiled_mma = make_tiled_mma(SM90_64x128x16_F32BF16BF16_SS, atom_layout=(2, 1, 1))
        phase = [0] * NUM_STAGES
        for t in range(pid, num_tiles, num_sms):
            bm = t // tiles_n
            bn = t % tiles_n
            acc = cute.make_fragment_like(tiled_mma.partition_C(...), dtype=cutlass.Float32)
            acc.fill_(0.0)
            for k in range(num_k):
                s = (t * num_k + k) % NUM_STAGES
                cute.arch.mbarrier_wait(full[s], phase[s])
                phase[s] ^= 1
                cute.gemm(tiled_mma, sA[s], sB[s], acc)
                cute.arch.mbarrier_arrive(empty[s])
            gC_tile = mC[bm*BLOCK_M:(bm+1)*BLOCK_M, bn*BLOCK_N:(bn+1)*BLOCK_N]
            cute.copy(acc.to(cutlass.BFloat16), gC_tile)


@cute.jit
def gemm_persistent(mA, mB, mC, M, N, K, num_sms: cutlass.Constexpr[int]):
    tma_a = cute.create_tma_atom(SM90_TMA_LOAD, mA, box_shape=(BLOCK_M, BLOCK_K))
    tma_b = cute.create_tma_atom(SM90_TMA_LOAD, mB, box_shape=(BLOCK_K, BLOCK_N))
    grid = (num_sms, 1, 1)
    block = (TOTAL_WARPS * 32, 1, 1)
    gemm_persistent_kernel(mA, mB, mC, tma_a, tma_b, M, N, K, num_sms).launch(
        grid=grid, block=block
    )


def main():
    M = N = K = 4096
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count

    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    c = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)

    def run():
        gemm_persistent(
            cute.make_tensor_from_torch(a),
            cute.make_tensor_from_torch(b),
            cute.make_tensor_from_torch(c),
            M, N, K, num_sms,
        )
    run()
    verify(c, a, b)
    ms = bench(run)
    report(f"stage5 persistent (SMs={num_sms})", M, N, K, ms)


if __name__ == "__main__":
    main()
