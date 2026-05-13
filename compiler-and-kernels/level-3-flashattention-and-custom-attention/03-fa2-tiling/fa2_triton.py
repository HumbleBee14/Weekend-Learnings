"""FA2 forward in Triton. One program per Q block.

Targets Triton 3.7+. Runs on T4 (no FA2 hardware support, but the kernel works);
on A100/H100 you get reasonable numbers. To hit FA2 parity you need warp
specialization (sub-module 04).

Run on a GPU:
    python fa2_triton.py
"""
from __future__ import annotations

import math

try:
    import torch
    import triton
    import triton.language as tl
except ImportError as e:
    raise SystemExit(f"This sub-module needs torch + triton. Got: {e}")


@triton.jit
def _fa2_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    L_ptr,  # log-sum-exp per row, for the backward pass
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

    # Load Q tile (BLOCK_M, D). Mask out-of-range rows.
    q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    m_mask = offs_m < N
    q = tl.load(q_ptrs, mask=m_mask[:, None], other=0.0)

    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    O_i = tl.zeros([BLOCK_M, D], tl.float32)

    # Inner loop over KV tiles.
    for start_n in range(0, N, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n < N
        k_ptrs = k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        v_ptrs = v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        k = tl.load(k_ptrs, mask=n_mask[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0)

        # S = Q @ K^T * scale
        s = tl.dot(q, tl.trans(k)) * sm_scale  # (BLOCK_M, BLOCK_N)
        # Mask invalid KV positions to -inf.
        s = tl.where(n_mask[None, :], s, float("-inf"))
        if CAUSAL:
            causal_mask = offs_m[:, None] >= offs_n[None, :]
            s = tl.where(causal_mask, s, float("-inf"))

        # Online softmax update.
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        # exp(-inf - -inf) is nan; tl.where below handles fully-masked rows.
        p = tl.exp(s - m_new[:, None])
        p = tl.where(s == float("-inf"), 0.0, p)
        rescale = tl.exp(m_i - m_new)
        rescale = tl.where(m_i == float("-inf"), 0.0, rescale)

        # ORDER: rescale O and l, then accumulate.
        l_i = l_i * rescale + tl.sum(p, axis=1)
        O_i = O_i * rescale[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    # Normalize.
    safe_l = tl.where(l_i == 0.0, 1.0, l_i)
    O_out = O_i / safe_l[:, None]

    o_ptrs = o_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(o_ptrs, O_out.to(O_ptr.dtype.element_ty), mask=m_mask[:, None])

    l_ptrs = L_ptr + pid_bh * N + offs_m
    tl.store(l_ptrs, m_i + tl.log(safe_l), mask=m_mask)


def fa2_forward_triton(
    q: torch.Tensor,  # (B, H, N, D)
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    block_m: int = 64,
    block_n: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert q.shape == k.shape == v.shape
    B, H, N, D = q.shape
    assert D in (16, 32, 64, 128), "head dim must be a small power of two"

    o = torch.empty_like(q)
    L = torch.empty((B, H, N), device=q.device, dtype=torch.float32)
    sm_scale = 1.0 / math.sqrt(D)

    grid = (triton.cdiv(N, block_m), B * H)
    _fa2_fwd_kernel[grid](
        q, k, v, o, L,
        sm_scale,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *o.stride(),
        B, H, N,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        D=D,
        CAUSAL=causal,
        num_warps=4,
        num_stages=2,
    )
    return o, L


def _smoke_test() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required.")
    torch.manual_seed(0)
    B, H, N, D = 2, 4, 256, 64
    dtype = torch.float32  # fp32 for tight tolerance vs reference
    q = torch.randn(B, H, N, D, device="cuda", dtype=dtype)
    k = torch.randn(B, H, N, D, device="cuda", dtype=dtype)
    v = torch.randn(B, H, N, D, device="cuda", dtype=dtype)

    o_ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
    o_tri, _ = fa2_forward_triton(q, k, v, causal=False)
    err = (o_ref - o_tri).abs().max().item()
    print(f"non-causal: max abs err vs SDPA = {err:.2e}")
    assert err < 1e-3, "fp32 attention should agree to ~1e-4; check rescale order"

    o_ref_c = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    o_tri_c, _ = fa2_forward_triton(q, k, v, causal=True)
    err_c = (o_ref_c - o_tri_c).abs().max().item()
    print(f"causal: max abs err vs SDPA = {err_c:.2e}")
    assert err_c < 1e-3

    print("fa2_triton matches SDPA. ship it.")


if __name__ == "__main__":
    _smoke_test()
