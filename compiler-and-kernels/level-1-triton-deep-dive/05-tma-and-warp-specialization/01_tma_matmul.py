"""
01_tma_matmul.py — Tiled GEMM with TMA descriptors, NO warp specialization.

This is the baseline for sub-module 05. It is the same shape of kernel as 04
ended on: tiled matmul, `tl.make_tensor_descriptor` for A/B/C, autotuned across
BLOCK_M/N/K. The only thing missing — and the thing 02 adds — is the
producer/consumer split via `warp_specialize=True`.

Run me first. Record the TFLOPS in notes.md. Then run 02 and compare.

Hardware notes:
- On H100/H200 expect ~600 TFLOPS fp16 at M=N=K=4096 (~61% of 989 peak).
- On T4 expect ~30-40 TFLOPS at the same shape (T4's fp16 peak is ~65).
- On B200 expect ~1200 TFLOPS (Blackwell tensor cores roughly 2x H100).

References:
- Tutorial: https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
- TMA descriptors: https://triton-lang.org/main/python-api/generated/triton.language.make_tensor_descriptor.html
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Hardware probe — print one line so the learner knows what they're on.
# ---------------------------------------------------------------------------

def describe_device():
    if not torch.cuda.is_available():
        return "no CUDA device — this script will not run"
    props = torch.cuda.get_device_properties(0)
    cc = props.major * 10 + props.minor
    name = props.name
    if cc >= 100:
        story = "Blackwell — TMA + tcgen05 + TMEM. Warp spec will help a lot in 02."
    elif cc == 90:
        story = "Hopper — TMA + wgmma. Warp spec will help (~1.3-1.5x) in 02."
    elif cc == 89:
        story = "Ada — no TMA hardware; descriptor lowers to cp.async. Warp spec ~flat in 02."
    elif cc == 80 or cc == 86:
        story = "Ampere — no TMA; cp.async only. Warp spec will not help in 02. That is expected."
    elif cc == 75:
        story = "Turing (T4) — no async copy hardware; falls back to ld.global. Warp spec no-ops."
    else:
        story = f"compute capability {cc} — older than Triton actively targets; YMMV."
    return f"{name} (cc {cc}) — {story}"


# ---------------------------------------------------------------------------
# Autotune configs — the same shape we used in 04. Note: NO warp_specialize
# and NO num_consumer_groups here. That is the whole point of this baseline.
# ---------------------------------------------------------------------------

def get_configs():
    configs = []
    for bm in [64, 128, 256]:
        for bn in [64, 128, 256]:
            for bk in [32, 64]:
                for nw in [4, 8]:
                    for ns in [2, 3, 4]:
                        # Skip configs likely to spill or be illegal.
                        if bm * bn * 2 > 128 * 1024:  # rough SMEM budget
                            continue
                        configs.append(
                            triton.Config(
                                {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk},
                                num_warps=nw,
                                num_stages=ns,
                            )
                        )
    return configs


@triton.autotune(configs=get_configs(), key=["M", "N", "K"])
@triton.jit
def matmul_kernel_tma(
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

    # On Hopper+, tl.make_tensor_descriptor lowers to TMA. On older arch it
    # lowers to cp.async or plain ld.global. Same source, three lowerings.
    a_desc = tl.make_tensor_descriptor(
        a_ptr, shape=[M, K], strides=[stride_am, stride_ak],
        block_shape=[BLOCK_M, BLOCK_K],
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr, shape=[K, N], strides=[stride_bk, stride_bn],
        block_shape=[BLOCK_K, BLOCK_N],
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr, shape=[M, N], strides=[stride_cm, stride_cn],
        block_shape=[BLOCK_M, BLOCK_N],
    )

    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    # NO warp_specialize here. That is what 02 adds.
    for k in tl.range(0, K, BLOCK_K, num_stages=3):
        a = a_desc.load([off_m, k])
        b = b_desc.load([k, off_n])
        acc = tl.dot(a, b, acc)

    c_desc.store([off_m, off_n], acc.to(tl.float16))


def matmul_tma(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, f"shape mismatch: {a.shape} @ {b.shape}"
    assert a.dtype == b.dtype == torch.float16, "this kernel is fp16-only for clarity"
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]), triton.cdiv(N, META["BLOCK_N"]))
    matmul_kernel_tma[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c


# ---------------------------------------------------------------------------
# Correctness + bench.
# ---------------------------------------------------------------------------

def main():
    print(f"device: {describe_device()}")
    if not torch.cuda.is_available():
        return

    torch.manual_seed(0)
    M = N = K = 4096
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)

    # Correctness — always before timing.
    c_triton = matmul_tma(a, b)
    c_torch = torch.matmul(a, b)
    # Loose tolerance because we accumulate fp32 and torch's reference may differ
    # in reduction order at this size.
    torch.testing.assert_close(c_triton, c_torch, atol=1.0, rtol=1e-2)
    print("correctness: ok")

    # Bench.
    ms_triton = triton.testing.do_bench(lambda: matmul_tma(a, b), warmup=25, rep=100)
    ms_torch = triton.testing.do_bench(lambda: torch.matmul(a, b), warmup=25, rep=100)

    flops = 2 * M * N * K
    tflops_triton = flops / (ms_triton * 1e-3) / 1e12
    tflops_torch = flops / (ms_torch * 1e-3) / 1e12

    print(f"shape: M=N=K={M}, dtype=fp16")
    print(f"triton (TMA, no warp spec): {ms_triton:7.3f} ms  ->  {tflops_triton:7.1f} TFLOPS")
    print(f"torch.matmul (cuBLAS):      {ms_torch:7.3f} ms  ->  {tflops_torch:7.1f} TFLOPS")
    print(f"ratio: {tflops_triton / tflops_torch * 100:.1f}% of cuBLAS")
    print()
    print("Record this number in notes.md, then run 02_warp_specialized_matmul.py.")


if __name__ == "__main__":
    main()
