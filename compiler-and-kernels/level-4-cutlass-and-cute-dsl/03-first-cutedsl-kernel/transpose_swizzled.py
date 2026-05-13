"""
transpose_swizzled.py — 64×64 transpose in CuTe-DSL, with and without
shared-memory swizzling. The bank-conflict pattern in unswizzled transpose
is the smallest didactic case where Swizzle<3,4,3> matters.

Run:   python transpose_swizzled.py
Expect on H100: ~2-4x speedup with swizzle.
Expect on T4:   ~1.5-2x speedup with swizzle.
"""

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import Swizzle


TILE = 64


@cute.kernel
def transpose_kernel(
    src: cute.Tensor,
    dst: cute.Tensor,
    swizzled: cutlass.Constexpr[bool],
):
    bm, bn, _ = cute.arch.block_idx()
    tm, tn, _ = cute.arch.thread_idx()

    # SMEM tile, optionally swizzled.
    if swizzled:
        sA = cute.make_smem_tensor(
            shape=(TILE, TILE),
            dtype=cutlass.Float32,
            swizzle=Swizzle(3, 4, 3),
        )
    else:
        sA = cute.make_smem_tensor(shape=(TILE, TILE), dtype=cutlass.Float32)

    # Coalesced GMEM → SMEM (each row of SMEM filled by one row of threads).
    g_row = bm * TILE + tm
    g_col = bn * TILE + tn
    sA[tm, tn] = src[g_row, g_col]
    cute.arch.barrier()

    # Transposed read from SMEM → coalesced store to GMEM.
    out_row = bn * TILE + tm
    out_col = bm * TILE + tn
    dst[out_row, out_col] = sA[tn, tm]


@cute.jit
def transpose(
    src: cute.Tensor,
    dst: cute.Tensor,
    swizzled: cutlass.Constexpr[bool],
):
    M, N = src.shape
    transpose_kernel(src, dst, swizzled).launch(
        grid=(M // TILE, N // TILE, 1),
        block=(TILE, TILE, 1),
    )


def benchmark(swizzled: bool, n_iter: int = 100):
    M = N = 4096
    src = torch.randn(M, N, device="cuda", dtype=torch.float32)
    dst = torch.empty(N, M, device="cuda", dtype=torch.float32)

    for _ in range(5):
        transpose(
            cute.make_tensor_from_torch(src),
            cute.make_tensor_from_torch(dst),
            swizzled,
        )
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iter):
        transpose(
            cute.make_tensor_from_torch(src),
            cute.make_tensor_from_torch(dst),
            swizzled,
        )
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / n_iter
    gbps = (2 * M * N * 4) / (ms * 1e6)

    # Correctness check.
    expected = src.t().contiguous()
    assert torch.allclose(dst, expected), "transpose mismatch"
    return ms, gbps


if __name__ == "__main__":
    ms_no, gbps_no = benchmark(swizzled=False)
    ms_yes, gbps_yes = benchmark(swizzled=True)
    print(f"unswizzled: {ms_no:7.3f} ms  {gbps_no:7.1f} GB/s")
    print(f"swizzled:   {ms_yes:7.3f} ms  {gbps_yes:7.1f} GB/s")
    print(f"speedup:    {ms_no / ms_yes:5.2f}x")
