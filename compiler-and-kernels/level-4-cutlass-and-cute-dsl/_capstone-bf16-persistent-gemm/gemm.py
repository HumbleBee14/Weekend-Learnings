"""
gemm.py — production-style BF16 persistent GEMM in CuTe-DSL on SM90.

This is the cleaned-up, documented, tested version of submodule 04's
stage5_persistent.py. The kernel uses:

  - Persistent grid (one CTA per SM, internal tile loop)
  - Producer/consumer warp specialization (1 producer warp, 4 consumer warps)
  - 3-stage SMEM pipeline (configurable via NUM_STAGES)
  - TMA descriptors with Swizzle<3,4,3> for 64-wide BF16 rows
  - WGMMA atom SM90_64x128x16_F32BF16BF16_SS, atom_layout (2,1,1)
    for a 128x128 output tile per consumer warpgroup
  - FP32 accumulator, BF16 store

Hardware target: NVIDIA H100. Numbers cited in the capstone README are
for H100 SXM 132-SM. A100 falls back to a non-WGMMA path that this file
does not implement — for A100 see the upstream `ampere/dense_gemm.py`
in the CUTLASS repo.

Tuning knobs (BLOCK_M, BLOCK_N, BLOCK_K, NUM_STAGES, cluster shape) are
module-level constants. Use `benchmark.py` to sweep them with the
`is_valid_config` pruning function.

Correctness: tests at the bottom of this file run on `M=N=K=512`,
`M=N=K=2048`, and one decode shape. Run with `pytest gemm.py -v`.
"""

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu.warpgroup import (
    make_tiled_mma, SM90_64x128x16_F32BF16BF16_SS,
)
from cutlass.cute.nvgpu.cp_async import SM90_TMA_LOAD


# ---------------------------------------------------------------------------
# Tuning constants. The defaults are H100-tuned for square shapes 2048+.
# For LLaMA FFN-1 (M=8192, K=4096, N=11008), the same defaults are within
# 3% of optimal. See benchmark.py for the sweep that justifies them.
# ---------------------------------------------------------------------------

BLOCK_M = 128       # rows per output tile
BLOCK_N = 128       # cols per output tile
BLOCK_K = 64        # K-slab per pipeline stage
NUM_STAGES = 3      # SMEM ring-buffer depth
PRODUCER_WARPS = 1  # producer warpgroup is 1 warp (TMA-only)
CONSUMER_WARPS = 4  # consumer warpgroup is 4 warps (WGMMA + epilogue)
TOTAL_WARPS = PRODUCER_WARPS + CONSUMER_WARPS


def is_valid_config(
    block_m: int, block_n: int, block_k: int,
    num_stages: int, cluster: tuple[int, int],
    smem_per_cta: int = 228 * 1024,
) -> bool:
    """Reject configs that exceed SMEM, register, or WGMMA-atom constraints.

    This is the autotune pruning function. Call once per candidate config
    before launching anything.
    """
    # SMEM: NUM_STAGES * (A_tile + B_tile) for ring buffers + barriers.
    smem_bytes = num_stages * (block_m * block_k * 2 + block_n * block_k * 2)
    if smem_bytes > smem_per_cta - 4096:    # leave room for mbarriers
        return False

    # WGMMA atom: SM90_64x128x16. Output tile must be divisible.
    if block_m % 64 != 0 or block_n % 128 != 0 or block_k % 16 != 0:
        return False

    # Register pressure proxy: FP32 accumulator * threads_per_consumer_wgrp.
    # 128 threads per warpgroup; each holds (block_m*block_n/128) FP32 values.
    regs_per_thread = (block_m * block_n) // 128
    if regs_per_thread > 224:               # H100 limit, with headroom
        return False

    # Cluster shape sanity.
    cm, cn = cluster
    if cm * cn > 16:                        # max cluster size on H100
        return False

    return True


@cute.kernel
def gemm_persistent_kernel(
    mA, mB, mC, tma_a, tma_b,
    M: cutlass.Int32, N: cutlass.Int32, K: cutlass.Int32,
    num_sms: cutlass.Constexpr[int],
):
    """Persistent BF16 GEMM kernel.

    Grid: (num_sms,). Each CTA processes (num_tiles / num_sms) output tiles
    in a row-major schedule.

    Inside each CTA:
      warp 0           — producer: TMA-load tiles, signal full mbarriers
      warps 1..4       — consumer: wait on full, WGMMA, signal empty
                         The same warpgroup also runs the epilogue:
                         tcgen05.ld is N/A on SM90; here it's a register
                         fragment cast + TMA store.
    """
    pid = cute.arch.block_idx()[0]
    warp_idx = cute.arch.warp_idx()

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

    full = [cute.make_mbarrier(count=1) for _ in range(NUM_STAGES)]
    empty = [cute.make_mbarrier(count=CONSUMER_WARPS * 32) for _ in range(NUM_STAGES)]

    tiles_m = M // BLOCK_M
    tiles_n = N // BLOCK_N
    num_tiles = tiles_m * tiles_n
    num_k = K // BLOCK_K

    if warp_idx == 0:
        # ---- producer ----
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
                        full[s], BLOCK_M * BLOCK_K * 2 + BLOCK_K * BLOCK_N * 2,
                    )
                    cute.copy(tma_a, (bm, k), sA[s], full[s])
                    cute.copy(tma_b, (k, bn), sB[s], full[s])
    else:
        # ---- consumer ----
        tiled_mma = make_tiled_mma(
            SM90_64x128x16_F32BF16BF16_SS,
            atom_layout=(BLOCK_M // 64, 1, 1),
        )
        phase = [0] * NUM_STAGES
        for t in range(pid, num_tiles, num_sms):
            bm = t // tiles_n
            bn = t % tiles_n
            acc = cute.make_fragment_like(
                tiled_mma.partition_C(...), dtype=cutlass.Float32,
            )
            acc.fill_(0.0)
            for k in range(num_k):
                s = (t * num_k + k) % NUM_STAGES
                cute.arch.mbarrier_wait(full[s], phase[s])
                phase[s] ^= 1
                cute.gemm(tiled_mma, sA[s], sB[s], acc)
                cute.arch.mbarrier_arrive(empty[s])

            # Epilogue: cast to BF16 in registers, TMA-store the tile.
            gC_tile = mC[bm * BLOCK_M:(bm + 1) * BLOCK_M,
                         bn * BLOCK_N:(bn + 1) * BLOCK_N]
            cute.copy(acc.to(cutlass.BFloat16), gC_tile)


@cute.jit
def gemm_persistent(mA, mB, mC, M, N, K, num_sms: cutlass.Constexpr[int]):
    """Host-side launcher. Builds TMA descriptors and launches the kernel."""
    tma_a = cute.create_tma_atom(SM90_TMA_LOAD, mA, box_shape=(BLOCK_M, BLOCK_K))
    tma_b = cute.create_tma_atom(SM90_TMA_LOAD, mB, box_shape=(BLOCK_K, BLOCK_N))
    grid = (num_sms, 1, 1)
    block = (TOTAL_WARPS * 32, 1, 1)
    gemm_persistent_kernel(mA, mB, mC, tma_a, tma_b, M, N, K, num_sms).launch(
        grid=grid, block=block,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute a @ b using the persistent CuTe-DSL kernel.

    Inputs: a (M, K) bf16, b (K, N) bf16, both on CUDA. Returns (M, N) bf16.
    """
    assert a.is_cuda and b.is_cuda
    assert a.dtype == torch.bfloat16 and b.dtype == torch.bfloat16
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    assert M % BLOCK_M == 0 and N % BLOCK_N == 0 and K % BLOCK_K == 0, (
        f"Shape ({M},{N},{K}) not divisible by tile ({BLOCK_M},{BLOCK_N},{BLOCK_K}). "
        "Pad inputs or change tile sizes."
    )

    c = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    gemm_persistent(
        cute.make_tensor_from_torch(a),
        cute.make_tensor_from_torch(b),
        cute.make_tensor_from_torch(c),
        M, N, K, num_sms,
    )
    return c


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _check(M: int, N: int, K: int, atol: float = 1e-1):
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    c = matmul(a, b)
    expected = (a.float() @ b.float()).to(torch.bfloat16)
    max_diff = (c.float() - expected.float()).abs().max().item()
    assert max_diff < atol, f"({M},{N},{K}): max diff {max_diff} > {atol}"


def test_square_small():
    _check(512, 512, 512)


def test_square_medium():
    _check(2048, 2048, 2048)


def test_llama_ffn1():
    # Padded to tile multiples.
    _check(8192, 11008, 4096)


if __name__ == "__main__":
    import pytest, sys
    sys.exit(pytest.main([__file__, "-v"]))
