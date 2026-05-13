"""FA2 with warp specialization on the inner KV loop. SM90+ for real speedup.

Take the sub-module 03 kernel, flip warp_specialize=True. On T4/Ampere the flag
is silently ignored or falls back; on H100 you get producer/consumer scheduling.

Run on H100:
    python fa3_warp_specialized_triton.py
"""
from __future__ import annotations

import math
import sys
import os

import torch
import triton
import triton.language as tl

# Reuse the strides-and-launch wrapper from sub-module 03.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "03-fa2-tiling"))


@triton.jit
def _fa_ws_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr,
    sm_scale,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    B, H, N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_b = pid_bh // H
    pid_h = pid_bh % H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)

    q_base = Q_ptr + pid_b * stride_qb + pid_h * stride_qh
    k_base = K_ptr + pid_b * stride_kb + pid_h * stride_kh
    v_base = V_ptr + pid_b * stride_vb + pid_h * stride_vh
    o_base = O_ptr + pid_b * stride_ob + pid_h * stride_oh

    q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    m_mask = offs_m < N
    q = tl.load(q_ptrs, mask=m_mask[:, None], other=0.0)

    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    O_i = tl.zeros([BLOCK_M, D], tl.float32)

    # The single line that changes from FA2: warp_specialize=True on the inner loop.
    # On SM90+ the compiler partitions producer (loads) and consumer (compute)
    # across warp groups. num_stages controls the SMEM double-buffer depth.
    for start_n in tl.range(0, N, BLOCK_N, warp_specialize=True, num_stages=3):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n < N
        k_ptrs = k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        v_ptrs = v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        k = tl.load(k_ptrs, mask=n_mask[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0)

        s = tl.dot(q, tl.trans(k)) * sm_scale
        s = tl.where(n_mask[None, :], s, float("-inf"))
        if CAUSAL:
            causal_mask = offs_m[:, None] >= offs_n[None, :]
            s = tl.where(causal_mask, s, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        p = tl.exp(s - m_new[:, None])
        p = tl.where(s == float("-inf"), 0.0, p)
        rescale = tl.exp(m_i - m_new)
        rescale = tl.where(m_i == float("-inf"), 0.0, rescale)

        l_i = l_i * rescale + tl.sum(p, axis=1)
        O_i = O_i * rescale[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    safe_l = tl.where(l_i == 0.0, 1.0, l_i)
    O_out = O_i / safe_l[:, None]

    o_ptrs = o_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(o_ptrs, O_out.to(O_ptr.dtype.element_ty), mask=m_mask[:, None])
    l_ptrs = L_ptr + pid_bh * N + offs_m
    tl.store(l_ptrs, m_i + tl.log(safe_l), mask=m_mask)


def fa3_warp_specialized_forward(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    causal: bool = False, block_m: int = 128, block_n: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, H, N, D = q.shape
    o = torch.empty_like(q)
    L = torch.empty((B, H, N), device=q.device, dtype=torch.float32)
    sm_scale = 1.0 / math.sqrt(D)
    grid = (triton.cdiv(N, block_m), B * H)
    _fa_ws_fwd_kernel[grid](
        q, k, v, o, L, sm_scale,
        *q.stride(), *k.stride(), *v.stride(), *o.stride(),
        B, H, N,
        BLOCK_M=block_m, BLOCK_N=block_n, D=D, CAUSAL=causal,
        num_warps=8, num_stages=3,
    )
    return o, L


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required.")
    cap = torch.cuda.get_device_capability()
    print(f"GPU compute capability: {cap}. Warp specialization needs SM90+ for real speedup.")

    torch.manual_seed(0)
    B, H, N, D = 2, 8, 4096, 64
    q = torch.randn(B, H, N, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, H, N, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, H, N, D, device="cuda", dtype=torch.bfloat16)

    o_ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    o_ws, _ = fa3_warp_specialized_forward(q, k, v, causal=True)
    err = (o_ref - o_ws).abs().max().item()
    print(f"max abs err vs SDPA: {err:.2e}")

    ms = triton.testing.do_bench(lambda: fa3_warp_specialized_forward(q, k, v, causal=True), warmup=25, rep=100)
    flops = 4 * B * H * N * N * D * 0.5  # ~half due to causal
    print(f"warp-specialized FA: {ms:.3f} ms   {flops/ms/1e9:.1f} TFLOPs/s")


if __name__ == "__main__":
    main()
