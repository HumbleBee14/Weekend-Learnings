"""
03_warp_specialized_attention.py — minimal FA2-style forward attention with
warp specialization on the inner K loop.

This is forward-only, causal=False, no dropout, no varlen, head_dim in {64, 128}.
It is intentionally simple — about 150 lines — so the warp-spec pattern is
visible without 800 lines of paged-KV and varlen indexing on top.

The shape: for each (batch, head) and each query tile of BLOCK_M rows, we
stream over key/value tiles of BLOCK_N rows, accumulating an online softmax
(see 02-first-triton-kernel/CONCEPTS.md for the online softmax derivation).

The warp-spec point: the inner loop over key/value tiles is exactly the same
shape as the K-loop in matmul — long, async-load-followed-by-MMA — so the
producer/consumer split applies cleanly. Producer warps issue TMA loads of
the K and V tiles; consumer warps run the two MMAs (QK and PV) plus the
softmax math. With num_consumer_groups=2 (FA3 ping-pong), one consumer
group runs the softmax while the other runs the next MMA.

Benchmark vs torch.nn.functional.scaled_dot_product_attention, which
dispatches to FA3 on Hopper / FA2 elsewhere. You will not beat FA3 with 150
lines of Triton — FA3 is hand-tuned CuTe and uses tricks (overlapped
softmax-with-GEMM, FP8 paths, hand-laid-out epilogues) that Triton doesn't
yet expose. The lesson is HOW CLOSE you get. ~80-90% of FA3 is the
production-relevant bar that vLLM and SGLang shoot for.

References:
- FA3 blog: https://tridao.me/blog/2024/flash3/
- Triton tutorial: python/tutorials/06-fused-attention.py in triton-lang/triton
- Anatomy of a Triton Attention Kernel: https://arxiv.org/abs/2511.11581
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


def describe_device():
    if not torch.cuda.is_available():
        return "no CUDA device"
    props = torch.cuda.get_device_properties(0)
    cc = props.major * 10 + props.minor
    name = props.name
    if cc >= 90:
        note = "warp spec WILL help; SDPA dispatches to FA3 on Hopper+"
    else:
        note = "warp spec will not help; SDPA dispatches to FA2 / xformers / math"
    return f"{name} (cc {cc}) — {note}"


# Configs: keep the list small so the script doesn't autotune for an hour the
# first time. The right configs for attention are different from matmul —
# BLOCK_M typically smaller, BLOCK_N tunable per head_dim.
def _attn_configs():
    cfgs = []
    for bm in [64, 128]:
        for bn in [32, 64, 128]:
            for nw in [4, 8]:
                for ns in [2, 3, 4]:
                    cfgs.append(triton.Config({"BLOCK_M": bm, "BLOCK_N": bn},
                                              num_warps=nw, num_stages=ns))
    return cfgs


@triton.autotune(configs=_attn_configs(), key=["N_CTX", "HEAD_DIM"])
@triton.jit
def attn_fwd_kernel(
    Q, K, V, Out,
    sm_scale,
    stride_qz, stride_qh, stride_qm, stride_qd,
    stride_kz, stride_kh, stride_kn, stride_kd,
    stride_vz, stride_vh, stride_vn, stride_vd,
    stride_oz, stride_oh, stride_om, stride_od,
    Z, H, N_CTX,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
):
    pid_m = tl.program_id(0)       # query-tile index
    pid_bh = tl.program_id(1)      # flat (batch, head) index
    pid_b = pid_bh // H
    pid_h = pid_bh % H

    q_off = pid_b * stride_qz + pid_h * stride_qh
    k_off = pid_b * stride_kz + pid_h * stride_kh
    v_off = pid_b * stride_vz + pid_h * stride_vh
    o_off = pid_b * stride_oz + pid_h * stride_oh

    # Q, K, V tile pointers. We use the descriptor API for K and V because
    # those are what the inner loop streams; the warp-spec pipeline targets
    # those loads specifically. Q is loaded once and held in registers.
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)

    q_ptrs = Q + q_off + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)

    k_desc = tl.make_tensor_descriptor(
        K + k_off, shape=[N_CTX, HEAD_DIM], strides=[stride_kn, stride_kd],
        block_shape=[BLOCK_N, HEAD_DIM],
    )
    v_desc = tl.make_tensor_descriptor(
        V + v_off, shape=[N_CTX, HEAD_DIM], strides=[stride_vn, stride_vd],
        block_shape=[BLOCK_N, HEAD_DIM],
    )

    # Online softmax accumulators.
    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # The inner loop is the warp-specialized one. On Hopper, the producer
    # warps issue the K and V TMA loads; the consumer warps run the QK MMA,
    # the softmax math, and the PV MMA. With num_consumer_groups=2 the two
    # consumer groups ping-pong: one does softmax while the other does MMA.
    for kv_start in tl.range(0, N_CTX, BLOCK_N,
                             warp_specialize=WARP_SPECIALIZE,
                             num_stages=3):
        k = k_desc.load([kv_start, 0])
        v = v_desc.load([kv_start, 0])

        # QK^T : (BLOCK_M, HEAD_DIM) @ (HEAD_DIM, BLOCK_N) — Triton accepts
        # the transpose via the second operand layout.
        qk = tl.dot(q, tl.trans(k))
        qk = qk * sm_scale

        # Online softmax update: rescale running stats if the new row max is
        # bigger than the previous one. (See 02-first-triton-kernel/CONCEPTS.md
        # for the derivation.)
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    # Finalize.
    acc = acc / l_i[:, None]
    o_ptrs = Out + o_off + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < N_CTX)


def attention_fwd(q, k, v, warp_specialize: bool = True):
    """Forward-only attention. Shapes: (B, H, N, D), fp16, D in {64, 128}."""
    B, H, N, D = q.shape
    assert q.shape == k.shape == v.shape
    assert D in (64, 128), "this minimal kernel supports head_dim in {64, 128}"
    out = torch.empty_like(q)
    sm_scale = 1.0 / (D ** 0.5)

    grid = lambda META: (triton.cdiv(N, META["BLOCK_M"]), B * H)
    attn_fwd_kernel[grid](
        q, k, v, out,
        sm_scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        B, H, N,
        HEAD_DIM=D,
        WARP_SPECIALIZE=warp_specialize,
    )
    return out


def main():
    print(f"device: {describe_device()}")
    if not torch.cuda.is_available():
        return

    torch.manual_seed(0)
    B, H, N, D = 2, 16, 4096, 64
    dtype = torch.float16

    q = torch.randn((B, H, N, D), device="cuda", dtype=dtype) * 0.1
    k = torch.randn((B, H, N, D), device="cuda", dtype=dtype) * 0.1
    v = torch.randn((B, H, N, D), device="cuda", dtype=dtype) * 0.1

    # Correctness — vs SDPA (the production reference).
    out_triton = attention_fwd(q, k, v, warp_specialize=True)
    out_ref = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    # Loose tolerance — the reductions are in different orders.
    torch.testing.assert_close(out_triton, out_ref, atol=2e-2, rtol=1e-2)
    print("correctness: ok")

    # Bench.
    ms_ws = triton.testing.do_bench(lambda: attention_fwd(q, k, v, warp_specialize=True),
                                    warmup=25, rep=100)
    ms_no = triton.testing.do_bench(lambda: attention_fwd(q, k, v, warp_specialize=False),
                                    warmup=25, rep=100)
    ms_sdpa = triton.testing.do_bench(
        lambda: F.scaled_dot_product_attention(q, k, v, is_causal=False),
        warmup=25, rep=100,
    )

    # FLOPs for attention forward: 4 * B * H * N^2 * D (QK + softmax-scaled + PV).
    flops = 4 * B * H * N * N * D
    def tflops(ms): return flops / (ms * 1e-3) / 1e12

    print(f"shape: B={B} H={H} N={N} D={D}, dtype=fp16")
    print(f"triton, warp_specialize=False:  {ms_no:7.3f} ms  ->  {tflops(ms_no):7.1f} TFLOPS")
    print(f"triton, warp_specialize=True:   {ms_ws:7.3f} ms  ->  {tflops(ms_ws):7.1f} TFLOPS")
    print(f"torch SDPA (FA3 on Hopper):     {ms_sdpa:7.3f} ms  ->  {tflops(ms_sdpa):7.1f} TFLOPS")
    print(f"warp_specialize speedup:        {ms_no / ms_ws:.2f}x")
    print(f"yours vs SDPA:                  {tflops(ms_ws) / tflops(ms_sdpa) * 100:.1f}%")
    print()
    print("On H100, expect ~80-90% of SDPA in ~150 lines of Triton. FA3 is")
    print("hand-tuned CuTe with overlapped softmax-MMA; you won't beat it,")
    print("but you should be within striking distance. That ratio — what you")
    print("get for ~150 LoC vs ~70k LoC — is why Triton matters.")


if __name__ == "__main__":
    main()
