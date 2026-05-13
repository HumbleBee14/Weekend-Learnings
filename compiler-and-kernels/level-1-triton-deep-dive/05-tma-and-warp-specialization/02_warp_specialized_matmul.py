"""
02_warp_specialized_matmul.py — same kernel as 01, plus `warp_specialize=True`.

The kernel source is byte-identical to 01 except for:

    for k in tl.range(0, K, BLOCK_K, warp_specialize=True, num_stages=4):

and the autotune configs grow two new knobs:

    num_consumer_groups  — 0 (none) or 2 (FA3 ping-pong)
    num_buffers_warp_spec — ring buffer depth

That's the entire diff. Read it slowly. Run me and compare TFLOPS against 01.

Expected behavior by hardware:
- Hopper / Blackwell: 20-50% speedup over 01. Real producer-consumer pipeline.
- Ada / Ampere / Turing: roughly flat (within noise). The compiler still emits
  a non-specialized lowering; the `warp_specialize=True` is a hint that gets
  ignored when the target arch can't benefit. Correctness is preserved.

If your speedup is 0% on Ampere, you have not done anything wrong. That is
the expected, instructive result. Warp specialization is a Hopper+ feature.

References:
- Tawa paper: https://arxiv.org/abs/2510.14719
- PyTorch design blog: https://pytorch.org/blog/warp-specialization-in-triton-design-and-roadmap/
- Triton PR #6288: https://github.com/triton-lang/triton/pull/6288
"""

import inspect
import torch
import triton
import triton.language as tl


def describe_device():
    if not torch.cuda.is_available():
        return "no CUDA device"
    props = torch.cuda.get_device_properties(0)
    cc = props.major * 10 + props.minor
    name = props.name
    will_help = cc >= 90
    note = "warp spec WILL help" if will_help else "warp spec will be ignored on this arch"
    return f"{name} (cc {cc}) — {note}"


# ---------------------------------------------------------------------------
# Detect whether this Triton version accepts num_consumer_groups /
# num_buffers_warp_spec on triton.Config. Older Triton (<3.4) doesn't, so we
# fall back to a config list without those keys.
# ---------------------------------------------------------------------------

def _config_supports_warp_spec_kwargs():
    sig = inspect.signature(triton.Config.__init__)
    return "num_consumer_groups" in sig.parameters


_HAS_WS_KWARGS = _config_supports_warp_spec_kwargs()


def get_configs():
    configs = []
    for bm in [128, 256]:
        for bn in [128, 256]:
            for bk in [64]:
                for nw in [4, 8]:
                    for ns in [3, 4]:
                        if bm * bn * 2 > 128 * 1024:
                            continue
                        base = {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk}
                        if _HAS_WS_KWARGS:
                            # FA3-style ping-pong: 2 consumer groups, deeper ring.
                            for ncg in [0, 2]:
                                for nbws in [2, 3, 4]:
                                    configs.append(
                                        triton.Config(
                                            base,
                                            num_warps=nw,
                                            num_stages=ns,
                                            num_consumer_groups=ncg,
                                            num_buffers_warp_spec=nbws,
                                        )
                                    )
                        else:
                            configs.append(triton.Config(base, num_warps=nw, num_stages=ns))
    return configs


# We pass warp_specialize as a constexpr to make graceful degradation explicit:
# on older Triton we'd just set it to False.
@triton.autotune(configs=get_configs(), key=["M", "N", "K"])
@triton.jit
def matmul_kernel_ws(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

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
    # THE ONE LINE THAT MATTERS.
    # When WARP_SPECIALIZE=True and arch is Hopper+, the compiler partitions
    # this loop body across producer warps (issuing TMA loads) and consumer
    # warps (running tl.dot). On older arch it silently lowers to the same
    # code as 01_tma_matmul.py.
    for k in tl.range(0, K, BLOCK_K, warp_specialize=WARP_SPECIALIZE, num_stages=4):
        a = a_desc.load([off_m, k])
        b = b_desc.load([k, off_n])
        acc = tl.dot(a, b, acc)

    c_desc.store([off_m, off_n], acc.to(tl.float16))


def matmul_ws(a: torch.Tensor, b: torch.Tensor, warp_specialize: bool = True) -> torch.Tensor:
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    assert a.dtype == b.dtype == torch.float16
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]), triton.cdiv(N, META["BLOCK_N"]))
    matmul_kernel_ws[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        WARP_SPECIALIZE=warp_specialize,
    )
    return c


def main():
    print(f"device: {describe_device()}")
    print(f"triton supports num_consumer_groups/num_buffers_warp_spec on Config: {_HAS_WS_KWARGS}")
    if not torch.cuda.is_available():
        return

    torch.manual_seed(0)
    M = N = K = 4096
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)

    # Correctness: run both warp_specialize on and off, verify against torch.
    c_ws = matmul_ws(a, b, warp_specialize=True)
    c_no = matmul_ws(a, b, warp_specialize=False)
    c_torch = torch.matmul(a, b)
    torch.testing.assert_close(c_ws, c_torch, atol=1.0, rtol=1e-2)
    torch.testing.assert_close(c_no, c_torch, atol=1.0, rtol=1e-2)
    print("correctness: ok (both warp_specialize=True and False)")

    # Bench all three.
    ms_ws = triton.testing.do_bench(lambda: matmul_ws(a, b, warp_specialize=True), warmup=25, rep=100)
    ms_no = triton.testing.do_bench(lambda: matmul_ws(a, b, warp_specialize=False), warmup=25, rep=100)
    ms_torch = triton.testing.do_bench(lambda: torch.matmul(a, b), warmup=25, rep=100)

    flops = 2 * M * N * K
    def tflops(ms):
        return flops / (ms * 1e-3) / 1e12

    print(f"shape: M=N=K={M}, dtype=fp16")
    print(f"triton, warp_specialize=False (== 01):  {ms_no:7.3f} ms  ->  {tflops(ms_no):7.1f} TFLOPS")
    print(f"triton, warp_specialize=True:           {ms_ws:7.3f} ms  ->  {tflops(ms_ws):7.1f} TFLOPS")
    print(f"torch.matmul (cuBLAS):                  {ms_torch:7.3f} ms  ->  {tflops(ms_torch):7.1f} TFLOPS")
    speedup = ms_no / ms_ws
    print(f"warp_specialize speedup over baseline:  {speedup:.2f}x")
    print(f"warp_specialize % of cuBLAS:            {tflops(ms_ws) / tflops(ms_torch) * 100:.1f}%")
    print()
    if speedup < 1.1:
        print("Speedup < 1.1x. If you're on H100/B200 this is unexpected — check that")
        print("autotune found a config with num_consumer_groups > 0. If you're on")
        print("Ampere/Turing/Ada this is expected and the point of the script is")
        print("understanding WHY: no TMA hardware -> no async pipeline to hide.")
    else:
        print(f"You hid {(1 - 1/speedup) * 100:.0f}% of HBM-latency cost behind tensor-core compute.")
        print("That is the producer/consumer pipeline doing its job.")


if __name__ == "__main__":
    main()
