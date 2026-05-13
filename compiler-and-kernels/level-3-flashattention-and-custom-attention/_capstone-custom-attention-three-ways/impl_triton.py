"""Hand-Triton FA2-style forward with sink + sliding-window + ALiBi + causal.

Forward only. Reuse the sub-module 03 kernel shape; add ALiBi inside the loop
and the sink+window+causal mask.

Run:
    python impl_triton.py
"""
from __future__ import annotations

import math
import sys
import os

import torch
import triton
import triton.language as tl


@triton.jit
def _capstone_kernel(
    Q, K, V, O,
    slopes_ptr,
    sm_scale,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    B, H, N,
    WINDOW: tl.constexpr,
    SINKS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_b = pid_bh // H
    pid_h = pid_bh % H
    slope = tl.load(slopes_ptr + pid_h)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)

    q_base = Q + pid_b * stride_qb + pid_h * stride_qh
    k_base = K + pid_b * stride_kb + pid_h * stride_kh
    v_base = V + pid_b * stride_vb + pid_h * stride_vh
    o_base = O + pid_b * stride_ob + pid_h * stride_oh

    q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    m_mask_q = offs_m < N
    q = tl.load(q_ptrs, mask=m_mask_q[:, None], other=0.0)

    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    O_i = tl.zeros([BLOCK_M, D], tl.float32)

    # Only iterate KV blocks that can be relevant.
    # Sinks live in [0, SINKS). Sliding window for row q covers [q-WINDOW, q].
    # The kernel-side decision: just iterate full range and rely on the mask.
    # A more sophisticated impl would do two phases: sink block, then window blocks.
    for start_n in range(0, N, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n < N
        k_ptrs = k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        v_ptrs = v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        k = tl.load(k_ptrs, mask=n_mask[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0)

        # QK^T * scale + ALiBi bias.
        s = tl.dot(q, tl.trans(k)) * sm_scale
        dist = offs_m[:, None].to(tl.float32) - offs_n[None, :].to(tl.float32)
        s = s - slope * tl.abs(dist)

        # Mask: causal AND (window OR sink).
        q2 = offs_m[:, None]
        k2 = offs_n[None, :]
        causal = q2 >= k2
        in_window = (q2 - k2) <= WINDOW
        is_sink = k2 < SINKS
        mask = n_mask[None, :] & causal & (in_window | is_sink)
        s = tl.where(mask, s, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        p = tl.exp(s - m_new[:, None])
        p = tl.where(s == float("-inf"), 0.0, p)
        rescale = tl.exp(m_i - m_new)
        rescale = tl.where(m_i == float("-inf"), 0.0, rescale)

        l_i = l_i * rescale + tl.sum(p, axis=1)
        O_i = O_i * rescale[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    safe_l = tl.where(l_i == 0.0, 1.0, l_i)
    out = O_i / safe_l[:, None]
    o_ptrs = o_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(o_ptrs, out.to(O.dtype.element_ty), mask=m_mask_q[:, None])


def capstone_attention_triton(q, k, v, window, sinks, slopes):
    B, H, N, D = q.shape
    o = torch.empty_like(q)
    sm_scale = 1.0 / math.sqrt(D)
    block_m, block_n = 64, 64
    grid = (triton.cdiv(N, block_m), B * H)
    _capstone_kernel[grid](
        q, k, v, o, slopes, sm_scale,
        *q.stride(), *k.stride(), *v.stride(), *o.stride(),
        B, H, N,
        WINDOW=window, SINKS=sinks,
        BLOCK_M=block_m, BLOCK_N=block_n, D=D,
        num_warps=4, num_stages=2,
    )
    return o


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required.")
    from reference import reference_attention, alibi_slopes

    torch.manual_seed(0)
    import numpy as np
    B, H, N, D = 1, 4, 256, 32
    W, S = 64, 2
    slopes_np = alibi_slopes(H)
    q_np = np.random.randn(B, H, N, D).astype(np.float32)
    k_np = np.random.randn(B, H, N, D).astype(np.float32)
    v_np = np.random.randn(B, H, N, D).astype(np.float32)
    o_ref = reference_attention(q_np.astype(np.float64), k_np.astype(np.float64),
                                v_np.astype(np.float64), window=W, sinks=S, slopes=slopes_np)

    q = torch.from_numpy(q_np).cuda()
    k = torch.from_numpy(k_np).cuda()
    v = torch.from_numpy(v_np).cuda()
    slopes = torch.from_numpy(slopes_np.astype(np.float32)).cuda()

    o_tri = capstone_attention_triton(q, k, v, window=W, sinks=S, slopes=slopes)
    err = float((torch.from_numpy(o_ref).cuda().float() - o_tri.float()).abs().max())
    print(f"max abs err vs reference (fp32): {err:.2e}")
    assert err < 1e-3, "hand-Triton capstone disagrees with reference; debug the mask"


if __name__ == "__main__":
    main()
