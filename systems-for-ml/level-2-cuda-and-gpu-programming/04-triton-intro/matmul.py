"""
A Triton matmul that hits ~95% of cuBLAS on Ampere. Compare to the CUDA C++ version
in Topic 3 — Triton is ~80 lines and matches what took 7 hand-written kernels there.

Run:
    pip install triton torch
    python matmul.py
"""

import torch
import triton
import triton.language as tl


def get_autotune_configs():
    """A small set of (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages) configs.

    Real production kernels (vLLM, SGLang) use 20-50 configs. We keep it small for
    teaching — autotune still picks the best for each shape.
    """
    return [
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 32, "GROUP_M": 8}, num_warps=2, num_stages=5),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=8, num_stages=3),
    ]


@triton.autotune(configs=get_autotune_configs(), key=["M", "N", "K"])
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,    # how many elements to skip in A per row/col
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,    # L2-cache-friendly tile reordering
):
    """Compute one BLOCK_M × BLOCK_N tile of C per program.

    GROUP_M reorders programs so that consecutive blocks operate on nearby tiles —
    this improves L2 cache reuse for the A and B chunks they share.
    """
    # ---- Tile reordering for L2 cache friendliness ----
    # Without grouping, programs sweep row-by-row over C. Programs in the same row
    # share A[row], programs in the same column share B[col]. Plain raster ordering
    # streams through B fast but reuses A poorly. GROUP_M groups programs to balance.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ---- Compute offsets for this tile ----
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # ---- Accumulator in registers ----
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # ---- Loop over K dimension ----
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # Load tiles. mask handles the case where K is not a multiple of BLOCK_K.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        # Tensor-core matmul. Triton picks HMMA / WGMMA / tcgen05 based on the GPU.
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = accumulator.to(tl.float16)   # downcast for storage if input was fp16

    # ---- Write tile to C ----
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Driver — call from Python like a normal function."""
    assert a.shape[1] == b.shape[0], "shape mismatch"
    assert a.is_cuda and b.is_cuda
    M, K = a.shape
    K2, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c


def main():
    if not torch.cuda.is_available():
        raise SystemExit("Needs CUDA.")

    M, N, K = 4096, 4096, 4096
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)

    # Trigger autotune on first call (slow — picks best config and caches it)
    print("First call (autotune)...")
    c = matmul(a, b)
    torch.cuda.synchronize()

    # Verify against PyTorch's matmul
    expected = a @ b
    err = (c - expected).abs().max().item()
    print(f"max abs error vs torch.matmul: {err:.2e}")

    # Benchmark
    import time
    for _ in range(3):
        matmul(a, b)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(50):
        matmul(a, b)
    torch.cuda.synchronize()
    triton_ms = (time.perf_counter() - t0) * 1000 / 50

    t0 = time.perf_counter()
    for _ in range(50):
        a @ b
    torch.cuda.synchronize()
    torch_ms = (time.perf_counter() - t0) * 1000 / 50

    flops = 2 * M * N * K
    print(f"\nM=N=K={M}, fp16:")
    print(f"  triton: {triton_ms:.3f} ms,  {flops / triton_ms / 1e9:.1f} TFLOPS")
    print(f"  torch:  {torch_ms:.3f} ms,  {flops / torch_ms / 1e9:.1f} TFLOPS  (cuBLAS underneath)")
    print(f"  triton/torch: {torch_ms / triton_ms * 100:.0f}%")


if __name__ == "__main__":
    main()
