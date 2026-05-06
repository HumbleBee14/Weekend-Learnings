"""
A minimal FlashAttention-2 in Triton. ~80 lines.

Same algorithm as Tri Dao's FA2 paper, simplified for clarity. Computes:
    out = softmax(Q @ K^T / sqrt(d)) @ V

without ever materializing the (N, N) attention matrix. The inner loop maintains
running (m, l) state and rescales the output as new K, V tiles arrive — exactly
the online softmax recursion from `online_softmax_numpy.py`.

This is the kernel you read every line of. Don't move on until each line makes sense.

Run:
    pip install triton torch
    python flash_attention_minimal.py
"""

import math
import time

import torch
import triton
import triton.language as tl


@triton.jit
def flash_attention_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    B, H, N, D: tl.constexpr,
    sm_scale,                        # 1/sqrt(D)
    BLOCK_M: tl.constexpr,           # rows of Q per program
    BLOCK_N: tl.constexpr,           # cols of K, V per inner loop tile
):
    """
    Each program handles BLOCK_M rows of one (batch, head). Iterates over K, V tiles.

    Maintains running (m, l) state and the output accumulator in registers.
    """
    # 3D launch grid: (m_tile, head, batch)
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)

    # Compute base pointers for this (batch, head)
    q_base = Q_ptr + pid_b * stride_qb + pid_h * stride_qh
    k_base = K_ptr + pid_b * stride_kb + pid_h * stride_kh
    v_base = V_ptr + pid_b * stride_vb + pid_h * stride_vh
    o_base = O_ptr + pid_b * stride_ob + pid_h * stride_oh

    # Q tile: rows [pid_m*BLOCK_M, (pid_m+1)*BLOCK_M)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # row indices
    offs_d = tl.arange(0, D)                           # head_dim indices

    # Load Q tile once — stays in registers throughout the inner loop
    q_ptrs = q_base + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N, other=0.0)

    # Initialize running state
    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)   # running max per row
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)                 # running sum per row
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)              # running output per row

    # Inner loop: stream K, V tiles
    for start_n in range(0, N, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)

        # Load K tile (note: K is stored as (..., N, D); we want (D, N) for QK^T,
        # so we just transpose access pattern by swapping the strides used)
        k_ptrs = k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=offs_n[:, None] < N, other=0.0)

        v_ptrs = v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=offs_n[:, None] < N, other=0.0)

        # ---- Compute attention scores for this tile ----
        # s = q @ k.T  →  shape (BLOCK_M, BLOCK_N)
        s = tl.dot(q, tl.trans(k)) * sm_scale

        # Mask out positions past sequence end (for the last K tile)
        s = tl.where(offs_n[None, :] < N, s, -float("inf"))

        # ---- Online softmax update ----
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        scale = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])              # local softmax numerator
        l_new = l_i * scale + tl.sum(p, axis=1)

        # ---- Rescale running output, then add new contribution ----
        acc = acc * scale[:, None] + tl.dot(p.to(v.dtype), v).to(tl.float32)

        m_i = m_new
        l_i = l_new

    # Final normalization
    acc = acc / l_i[:, None]

    # Write output tile
    o_ptrs = o_base + offs_m[:, None] * stride_on + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc.to(O_ptr.dtype.element_ty), mask=offs_m[:, None] < N)


def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Driver. Inputs (B, H, N, D). Returns same shape."""
    B, H, N, D = q.shape
    assert k.shape == (B, H, N, D) and v.shape == (B, H, N, D)
    assert D in {32, 64, 128, 256}, "D must be a tile-friendly power of 2"
    sm_scale = 1.0 / math.sqrt(D)

    BLOCK_M = 64
    BLOCK_N = 64

    out = torch.empty_like(q)
    grid = (triton.cdiv(N, BLOCK_M), H, B)

    flash_attention_kernel[grid](
        q, k, v, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        B, H, N, D=D,
        sm_scale=sm_scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )
    return out


def reference_attention(q, k, v):
    """Standard PyTorch attention for correctness check."""
    sm_scale = 1.0 / math.sqrt(q.shape[-1])
    s = torch.matmul(q, k.transpose(-2, -1)) * sm_scale
    p = torch.softmax(s, dim=-1)
    return torch.matmul(p, v)


def main():
    if not torch.cuda.is_available():
        raise SystemExit("Needs CUDA.")

    B, H, N, D = 1, 8, 1024, 64
    torch.manual_seed(0)
    q = torch.randn((B, H, N, D), device="cuda", dtype=torch.float16)
    k = torch.randn((B, H, N, D), device="cuda", dtype=torch.float16)
    v = torch.randn((B, H, N, D), device="cuda", dtype=torch.float16)

    # Warmup
    out_fa = flash_attention(q, k, v)
    out_ref = reference_attention(q, k, v)

    err = (out_fa - out_ref).abs().max().item()
    print(f"max abs error vs reference: {err:.4f}  (small fp16 noise expected)")

    # Benchmark
    def bench(fn, name):
        for _ in range(3): fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(50): fn()
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000 / 50
        # FLOPs: 2 matmuls of size (N,D)·(D,N) and (N,N)·(N,D) = 4*B*H*N²*D
        flops = 4 * B * H * N * N * D
        print(f"  {name:<24} {ms:.3f} ms  ({flops / ms / 1e9:.0f} GFLOPS)")

    print(f"\nB={B}, H={H}, N={N}, D={D}, fp16:")
    bench(lambda: reference_attention(q, k, v), "reference (torch)")
    bench(lambda: flash_attention(q, k, v),     "minimal flash attn")
    bench(lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v),
          "torch SDPA (FA2 internal)")


if __name__ == "__main__":
    main()
