"""
01 — Static-persistent matmul.

Grid is exactly (num_SMs,). Each program owns a precomputed contiguous range
of output tiles and walks them in order. Compare against the non-persistent
form (grid = num_tiles, hardware schedules) on two shapes:

  - Decode (M=1, N=4096, K=4096): the non-persistent grid is tiny, so
    launch overhead dominates. Persistent should win by 1.3-2x on T4.
  - Square (M=N=K=2048): both forms saturate the device; expect a near-tie.

The lesson is in the gap, not the absolute numbers. Persistence is a structural
choice that buys you (a) graph-friendliness (see file 03) and (b) launch
amortization on small grids. It is not a free speedup on every shape.

Reference for the canonical persistent matmul shape:
  https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html
"""

import torch
import triton
import triton.language as tl


# ----------------------------------------------------------------------------
# Non-persistent baseline. Standard tiled matmul from sub-module 04.
# ----------------------------------------------------------------------------

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    # Grouped tile ordering for L2 locality (standard Triton tutorial pattern).
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] + k * BLOCK_K < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] + k * BLOCK_K < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=mask)


def matmul_nonpersistent(a, b, BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, GROUP_M=8,
                        num_warps=4, num_stages=2):
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    matmul_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_M=GROUP_M,
        num_warps=num_warps, num_stages=num_stages,
    )
    return c


# ----------------------------------------------------------------------------
# Static-persistent: grid = (num_SMs,). Each program walks a slice of tile space.
# ----------------------------------------------------------------------------

@triton.jit
def persistent_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    NUM_SMS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    total_tiles = num_pid_m * num_pid_n

    # Static schedule: this pid owns tiles [pid, pid + NUM_SMS, pid + 2*NUM_SMS, ...].
    # Round-robin distribution gives better L2 reuse than a contiguous chunking
    # when adjacent tiles share input rows/columns. See the PyTorch grouped-GEMM
    # blog for a richer discussion.
    for tile_id in range(pid, total_tiles, NUM_SMS):
        # Grouped (super-tile) re-ordering inside the persistent walk.
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] + k * BLOCK_K < K), other=0.0)
            b = tl.load(b_ptrs, mask=(offs_k[:, None] + k * BLOCK_K < K) & (offs_n[None, :] < N), other=0.0)
            acc += tl.dot(a, b)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=mask)


def matmul_persistent(a, b, BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, GROUP_M=8,
                     num_warps=4, num_stages=2):
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    num_sms = torch.cuda.get_device_properties(a.device).multi_processor_count
    # Cap grid so we don't launch more programs than tiles (wasted SMs sit in
    # the loop with zero iterations, harmless but cheap to avoid).
    total_tiles = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    grid = (min(num_sms, total_tiles),)
    persistent_matmul_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        NUM_SMS=grid[0],
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_M=GROUP_M,
        num_warps=num_warps, num_stages=num_stages,
    )
    return c


# ----------------------------------------------------------------------------
# Benchmark
# ----------------------------------------------------------------------------

def bench_shape(M, N, K, dtype=torch.float16, label=""):
    torch.manual_seed(0)
    a = torch.randn((M, K), device="cuda", dtype=dtype)
    b = torch.randn((K, N), device="cuda", dtype=dtype)

    # Tile config tuned for T4. On H100 use larger blocks.
    BLOCK_M = 16 if M < 32 else 64
    cfg = dict(BLOCK_M=BLOCK_M, BLOCK_N=64, BLOCK_K=32, GROUP_M=8, num_warps=4, num_stages=2)

    # Correctness.
    c_np = matmul_nonpersistent(a, b, **cfg)
    c_p = matmul_persistent(a, b, **cfg)
    c_ref = a.float() @ b.float()
    err_np = (c_np.float() - c_ref).abs().max().item()
    err_p = (c_p.float() - c_ref).abs().max().item()

    ms_np = triton.testing.do_bench(lambda: matmul_nonpersistent(a, b, **cfg))
    ms_p = triton.testing.do_bench(lambda: matmul_persistent(a, b, **cfg))

    num_tiles_np = triton.cdiv(M, cfg["BLOCK_M"]) * triton.cdiv(N, cfg["BLOCK_N"])
    num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    print(f"\n{label}  M={M} N={N} K={K} dtype={dtype}")
    print(f"  err non-persistent={err_np:.2e}  err persistent={err_p:.2e}")
    print(f"  non-persistent grid={num_tiles_np} tiles  |  persistent grid={min(num_sms, num_tiles_np)}")
    print(f"  non-persistent: {ms_np*1000:.2f} us")
    print(f"  persistent    : {ms_p*1000:.2f} us  ({ms_np/ms_p:.2f}x vs non-persistent)")


def main():
    assert torch.cuda.is_available()
    dev = torch.cuda.get_device_properties("cuda")
    print(f"GPU: {dev.name}, {dev.multi_processor_count} SMs")

    # Decode shape: tiny grid, launch overhead dominates.
    bench_shape(M=1, N=4096, K=4096, label="[decode M=1]")

    # Small batch: still few tiles relative to SM count.
    bench_shape(M=8, N=4096, K=4096, label="[decode M=8]")

    # Square: large enough grid that both forms saturate the device.
    bench_shape(M=2048, N=2048, K=2048, label="[square 2k]")

    print("\nExpected: persistent wins on M=1 and M=8 (1.3-2x on T4), ties on the square shape.")
    print("If persistent loses on the square, your tile config is suboptimal — that's fine, the")
    print("structural point is that persistence enables the graph capture in file 03.")


if __name__ == "__main__":
    main()
