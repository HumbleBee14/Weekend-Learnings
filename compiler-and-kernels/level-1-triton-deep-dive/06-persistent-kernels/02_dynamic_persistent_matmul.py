"""
02 — Dynamic-persistent matmul via atomic tile claiming.

The static schedule in file 01 deals tiles round-robin. That's optimal when
every tile costs the same. It is suboptimal when tiles cost wildly different
amounts — split-K, variable-seqlen attention, grouped-GEMM-with-skewed-experts.
The fix: dynamic work-stealing. Every program loops claiming the next tile
from a global atomic counter, so fast programs grab more tiles than slow ones.

This is the same scheme the vLLM Triton paged-attention kernel uses for
variable-length decode batches; see
  https://blog.vllm.ai/2026/03/04/vllm-triton-backend-deep-dive.html
and the formal description in arXiv:2511.11581 ("Anatomy of a Triton
Attention Kernel"). The PyTorch grouped-GEMM MoE blog
  https://pytorch.org/blog/accelerating-moes-with-a-triton-persistent-cache-aware-grouped-gemm-kernel/
uses an equivalent pattern.

We simulate ragged work by giving each tile a different K-loop length, so
some tiles do 10x the FLOPs of others. Static schedule's tail latency is set
by the unlucky program that drew the heaviest tiles; dynamic schedule
amortizes the imbalance across all programs.
"""

import torch
import triton
import triton.language as tl


# ----------------------------------------------------------------------------
# Static-persistent reference. Identical structure to file 01 but takes a
# per-tile K-length vector so we can simulate ragged work.
# ----------------------------------------------------------------------------

@triton.jit
def static_ragged_kernel(
    a_ptr, b_ptr, c_ptr, k_per_tile_ptr,
    M, N, K_MAX,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    NUM_SMS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    total_tiles = num_pid_m * num_pid_n

    for tile_id in range(pid, total_tiles, NUM_SMS):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n

        k_this = tl.load(k_per_tile_ptr + tile_id)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(k_this, BLOCK_K)):
            mask_k = offs_k[None, :] + k * BLOCK_K < k_this
            a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & mask_k, other=0.0)
            b = tl.load(b_ptrs, mask=(offs_k[:, None] + k * BLOCK_K < k_this) & (offs_n[None, :] < N), other=0.0)
            acc += tl.dot(a, b)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=mask)


# ----------------------------------------------------------------------------
# Dynamic-persistent: atomic counter for tile claiming.
# ----------------------------------------------------------------------------

@triton.jit
def dynamic_ragged_kernel(
    a_ptr, b_ptr, c_ptr, k_per_tile_ptr, tile_counter_ptr,
    M, N, K_MAX, total_tiles,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    num_pid_n = tl.cdiv(N, BLOCK_N)

    # Initial tile: race on the counter. Returns the OLD value, then increments.
    tile_id = tl.atomic_add(tile_counter_ptr, 1)

    while tile_id < total_tiles:
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n

        k_this = tl.load(k_per_tile_ptr + tile_id)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(k_this, BLOCK_K)):
            mask_k = offs_k[None, :] + k * BLOCK_K < k_this
            a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & mask_k, other=0.0)
            b = tl.load(b_ptrs, mask=(offs_k[:, None] + k * BLOCK_K < k_this) & (offs_n[None, :] < N), other=0.0)
            acc += tl.dot(a, b)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=mask)

        # Claim next tile. The atomic serializes through L2, but it's a single
        # 32-bit add per tile — cheap relative to a BLOCK_M x BLOCK_N matmul step.
        tile_id = tl.atomic_add(tile_counter_ptr, 1)


# ----------------------------------------------------------------------------
# Drivers
# ----------------------------------------------------------------------------

def matmul_static_ragged(a, b, k_per_tile, BLOCK_M=64, BLOCK_N=64, BLOCK_K=32):
    M, K_MAX = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    num_sms = torch.cuda.get_device_properties(a.device).multi_processor_count
    total_tiles = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    assert k_per_tile.numel() == total_tiles
    grid = (min(num_sms, total_tiles),)
    static_ragged_kernel[grid](
        a, b, c, k_per_tile, M, N, K_MAX,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        NUM_SMS=grid[0],
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )
    return c


def matmul_dynamic_ragged(a, b, k_per_tile, BLOCK_M=64, BLOCK_N=64, BLOCK_K=32):
    M, K_MAX = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    num_sms = torch.cuda.get_device_properties(a.device).multi_processor_count
    total_tiles = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    assert k_per_tile.numel() == total_tiles
    # Reset the atomic counter every launch. This single int32 lives in HBM.
    tile_counter = torch.zeros(1, device=a.device, dtype=torch.int32)
    grid = (min(num_sms, total_tiles),)
    dynamic_ragged_kernel[grid](
        a, b, c, k_per_tile, tile_counter, M, N, K_MAX, total_tiles,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )
    return c


def reference_ragged(a, b, k_per_tile, BLOCK_M=64, BLOCK_N=64):
    """Slow reference: each output tile is a @ b but truncated to k_per_tile[tile]."""
    M, K_MAX = a.shape
    _, N = b.shape
    num_pid_m = triton.cdiv(M, BLOCK_M)
    num_pid_n = triton.cdiv(N, BLOCK_N)
    c = torch.zeros((M, N), device=a.device, dtype=a.dtype)
    for tile_id in range(num_pid_m * num_pid_n):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n
        m0, m1 = pid_m * BLOCK_M, min(M, (pid_m + 1) * BLOCK_M)
        n0, n1 = pid_n * BLOCK_N, min(N, (pid_n + 1) * BLOCK_N)
        kk = int(k_per_tile[tile_id].item())
        c[m0:m1, n0:n1] = (a[m0:m1, :kk].float() @ b[:kk, n0:n1].float()).to(a.dtype)
    return c


def build_ragged_workload(M, N, K_max, BLOCK_M, BLOCK_N, skew=0.7, seed=0):
    """Construct a per-tile K vector with high coefficient of variation.

    `skew` controls how many tiles get the full K_max vs the short K. Higher
    skew => more imbalance => more headroom for dynamic scheduling.
    """
    num_pid_m = triton.cdiv(M, BLOCK_M)
    num_pid_n = triton.cdiv(N, BLOCK_N)
    total = num_pid_m * num_pid_n
    g = torch.Generator(device="cpu").manual_seed(seed)
    r = torch.rand(total, generator=g)
    # Roughly `skew` fraction of tiles get short K (K_max // 8), the rest full.
    k_per_tile = torch.where(r < skew,
                             torch.full((total,), K_max // 8),
                             torch.full((total,), K_max)).to(torch.int32)
    return k_per_tile.cuda()


def main():
    assert torch.cuda.is_available()
    dev = torch.cuda.get_device_properties("cuda")
    print(f"GPU: {dev.name}, {dev.multi_processor_count} SMs")

    torch.manual_seed(0)
    M, N, K_MAX = 1024, 1024, 1024
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32

    a = torch.randn((M, K_MAX), device="cuda", dtype=torch.float16)
    b = torch.randn((K_MAX, N), device="cuda", dtype=torch.float16)
    k_per_tile = build_ragged_workload(M, N, K_MAX, BLOCK_M, BLOCK_N, skew=0.75)

    total_tiles = k_per_tile.numel()
    cv = (k_per_tile.float().std() / k_per_tile.float().mean()).item()
    print(f"Workload: {total_tiles} tiles, K per tile in {{{K_MAX//8}, {K_MAX}}}, "
          f"coefficient of variation = {cv:.2f}")

    # Correctness.
    c_static = matmul_static_ragged(a, b, k_per_tile, BLOCK_M, BLOCK_N, BLOCK_K)
    c_dyn = matmul_dynamic_ragged(a, b, k_per_tile, BLOCK_M, BLOCK_N, BLOCK_K)
    c_ref = reference_ragged(a, b, k_per_tile, BLOCK_M, BLOCK_N)
    err_static = (c_static.float() - c_ref.float()).abs().max().item()
    err_dyn = (c_dyn.float() - c_ref.float()).abs().max().item()
    print(f"err static={err_static:.2e}  err dynamic={err_dyn:.2e}")

    ms_static = triton.testing.do_bench(
        lambda: matmul_static_ragged(a, b, k_per_tile, BLOCK_M, BLOCK_N, BLOCK_K))
    ms_dyn = triton.testing.do_bench(
        lambda: matmul_dynamic_ragged(a, b, k_per_tile, BLOCK_M, BLOCK_N, BLOCK_K))
    print(f"static persistent : {ms_static*1000:.2f} us")
    print(f"dynamic persistent: {ms_dyn*1000:.2f} us  ({ms_static/ms_dyn:.2f}x vs static)")

    print("\nExpected: dynamic 1.2-1.5x faster on this skewed workload. The gap")
    print("widens as CV(tile_cost) grows. For uniform tile cost (CV ~= 0), the")
    print("static schedule wins because it avoids the atomic. There is a crossover.")
    print("\nWhy this matters: variable-seqlen decode in vLLM has CV ~= 0.5-1.0;")
    print("MoE grouped GEMM with skewed expert loads is even higher. Both ship dynamic.")


if __name__ == "__main__":
    main()
