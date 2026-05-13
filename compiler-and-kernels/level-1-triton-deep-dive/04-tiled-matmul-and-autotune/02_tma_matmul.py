"""
02 — Tiled matmul using tl.make_tensor_descriptor.

Same tile loop as 01, but each load is a tensor-descriptor load. On Hopper+
this lowers to TMA: a single async bulk copy from HBM to SRAM. The descriptor
encodes the tensor shape, strides, and tile shape; the compiler emits one
instruction per tile load instead of per-element pointer math.

On pre-Hopper (T4, A100, RTX 4090), the descriptor falls back to regular
loads. You'll see roughly the same speed as 01. That's expected — the win
shows up on H100 / B200.

Requires: Triton >= 3.4.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def matmul_tma_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Tensor descriptors. On Hopper+ these lower to TMA loads; on pre-Hopper
    # they're regular tile loads with the same semantics.
    a_desc = tl.make_tensor_descriptor(
        a_ptr, shape=[M, K], strides=[K, 1], block_shape=[BLOCK_M, BLOCK_K]
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr, shape=[K, N], strides=[N, 1], block_shape=[BLOCK_K, BLOCK_N]
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr, shape=[M, N], strides=[N, 1], block_shape=[BLOCK_M, BLOCK_N]
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    offs_m = pid_m * BLOCK_M
    offs_n = pid_n * BLOCK_N

    for k in range(0, K, BLOCK_K):
        a = a_desc.load([offs_m, k])
        b = b_desc.load([k, offs_n])
        acc += tl.dot(a, b)

    c_desc.store([offs_m, offs_n], acc.to(c_ptr.dtype.element_ty))


def matmul_tma(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.is_cuda and b.is_cuda
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 64
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    matmul_tma_kernel[grid](
        a, b, c, M, N, K,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=8, num_stages=3,
    )
    return c


def main():
    torch.manual_seed(0)
    for M, N, K in [(256, 256, 256), (1024, 1024, 1024), (4096, 4096, 4096)]:
        a = torch.randn(M, K, device="cuda", dtype=torch.float16)
        b = torch.randn(K, N, device="cuda", dtype=torch.float16)
        c_t = matmul_tma(a, b)
        c_r = a @ b
        diff = (c_t - c_r).abs().max().item()
        print(f"({M},{N},{K})  max diff = {diff:.2e}")
        assert diff < 1.0

    M = N = K = 4096
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    ms_triton = triton.testing.do_bench(lambda: matmul_tma(a, b))
    ms_torch = triton.testing.do_bench(lambda: a @ b)
    flops = 2 * M * N * K
    tflops_triton = flops / (ms_triton * 1e-3) / 1e12
    tflops_torch = flops / (ms_torch * 1e-3) / 1e12

    print(f"\n4096^3 fp16 matmul (TMA descriptors):")
    print(f"  triton: {ms_triton:.2f} ms   {tflops_triton:.1f} TFLOPS   "
          f"{tflops_triton/tflops_torch*100:.0f}% of torch")
    print(f"  torch : {ms_torch:.2f} ms   {tflops_torch:.1f} TFLOPS")
    print()
    print("On H100/B200: expect 50-80% of cuBLAS. On pre-Hopper: ~same as step 01.")
    print("(The TMA path falls back to regular loads on pre-Hopper hardware.)")


if __name__ == "__main__":
    main()
