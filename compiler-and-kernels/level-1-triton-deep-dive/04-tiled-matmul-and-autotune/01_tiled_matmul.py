"""
01 — Tiled matmul, the standard form.

The kernel is the textbook tiled GEMM:
  - 2D grid over output tiles
  - inner K-loop with tl.dot accumulating into a register tile
  - regular tl.load with explicit pointer arithmetic

No autotune yet. BLOCK_M, BLOCK_N, BLOCK_K are hand-picked.

Expected: ~25-40% of torch.matmul (which dispatches to cuBLAS) on most GPUs.
The number is unimpressive. We're not using tensor descriptors yet, and the
config is fixed. The next two steps fix each in turn.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        # Masks for the K boundary on the last iteration.
        a_mask = (offs_m[:, None] < M) & ((k + offs_k)[None, :] < K)
        b_mask = ((k + offs_k)[:, None] < K) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=c_mask)


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.is_cuda and b.is_cuda and a.shape[1] == b.shape[0]
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )
    return c


def main():
    torch.manual_seed(0)
    for M, N, K in [(256, 256, 256), (1024, 1024, 1024), (4097, 4096, 4096)]:
        a = torch.randn(M, K, device="cuda", dtype=torch.float16)
        b = torch.randn(K, N, device="cuda", dtype=torch.float16)
        c_t = matmul(a, b)
        c_r = a @ b
        diff = (c_t - c_r).abs().max().item()
        print(f"({M},{N},{K})  max diff = {diff:.2e}  {'OK' if diff < 1.0 else 'WRONG'}")
        assert diff < 1.0  # fp16 matmul accumulation precision is loose

    # Benchmark
    M = N = K = 4096
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    ms_triton = triton.testing.do_bench(lambda: matmul(a, b))
    ms_torch = triton.testing.do_bench(lambda: a @ b)
    flops = 2 * M * N * K  # 1 mul + 1 add per output element per K
    tflops_triton = flops / (ms_triton * 1e-3) / 1e12
    tflops_torch = flops / (ms_torch * 1e-3) / 1e12

    print(f"\n4096^3 fp16 matmul:")
    print(f"  triton (basic tiled): {ms_triton:.2f} ms   {tflops_triton:.1f} TFLOPS   {tflops_triton/tflops_torch*100:.0f}% of torch")
    print(f"  torch.matmul (cuBLAS): {ms_torch:.2f} ms   {tflops_torch:.1f} TFLOPS")
    print()
    print("If this is ~30% of torch, you're on track. The next two steps close most of the gap.")


if __name__ == "__main__":
    main()
