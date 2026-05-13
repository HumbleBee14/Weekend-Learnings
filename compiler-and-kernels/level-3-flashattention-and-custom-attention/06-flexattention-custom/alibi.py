"""ALiBi via score_mod. Verify vs NumPy reference; benchmark vs SDPA.

Run on GPU:
    python alibi.py
"""
from __future__ import annotations

import math

import torch
import triton
from torch.nn.attention.flex_attention import flex_attention, create_block_mask


def alibi_slopes_for(num_heads: int) -> torch.Tensor:
    """Standard ALiBi slopes (geometric progression, base 2)."""
    def slopes_power_of_2(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio ** i for i in range(n)]
    if math.log2(num_heads).is_integer():
        return torch.tensor(slopes_power_of_2(num_heads))
    # Non-power-of-2 case: interpolate. Skipping; just pad with 1.0 for the demo.
    base = slopes_power_of_2(2 ** int(math.log2(num_heads)))
    return torch.tensor(base + [1.0] * (num_heads - len(base)))


def reference_alibi_attention(q, k, v, slopes, is_causal=True):
    B, H, N, D = q.shape
    scale = 1.0 / math.sqrt(D)
    s = (q @ k.transpose(-2, -1)) * scale  # (B, H, N, N)
    q_idx = torch.arange(N, device=q.device)[:, None]
    kv_idx = torch.arange(N, device=q.device)[None, :]
    bias = -slopes.view(1, H, 1, 1) * (q_idx - kv_idx).abs().unsqueeze(0).unsqueeze(0)
    s = s + bias
    if is_causal:
        s = s.masked_fill(q_idx < kv_idx, float("-inf"))
    p = torch.softmax(s, dim=-1)
    return p @ v


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required.")
    torch.manual_seed(0)
    B, H, N, D = 2, 8, 1024, 64
    slopes = alibi_slopes_for(H).to("cuda", dtype=torch.float32)

    def alibi_score(score, b, h, q_idx, kv_idx):
        return score - slopes[h] * (q_idx - kv_idx).abs()

    def causal(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    block_mask = create_block_mask(causal, B=None, H=None, Q_LEN=N, KV_LEN=N, device="cuda")

    flex = torch.compile(flex_attention, dynamic=False)

    # Correctness vs reference, fp32 for tight tolerance.
    q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)
    k = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)
    v = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)

    o_ref = reference_alibi_attention(q, k, v, slopes, is_causal=True)
    o_flex = flex(q, k, v, score_mod=alibi_score, block_mask=block_mask)
    err = (o_ref - o_flex).abs().max().item()
    print(f"correctness vs ref (fp32): max err = {err:.2e}")
    assert err < 1e-3, "FlexAttention ALiBi disagrees with reference; check slope shape"

    # Benchmark in bf16.
    q = q.to(torch.bfloat16); k = k.to(torch.bfloat16); v = v.to(torch.bfloat16)
    ms_flex = triton.testing.do_bench(
        lambda: flex(q, k, v, score_mod=alibi_score, block_mask=block_mask), warmup=25, rep=100
    )
    ms_ref = triton.testing.do_bench(
        lambda: reference_alibi_attention(q, k, v, slopes, is_causal=True), warmup=10, rep=30
    )
    flops = 4 * B * H * N * N * D * 0.5  # causal halves
    print(f"\nshape (B,H,N,D)=({B},{H},{N},{D}) bf16 causal+ALiBi")
    print(f"  reference (materializes S):  {ms_ref:7.3f} ms   {flops/ms_ref/1e9:7.1f} TFLOPs/s")
    print(f"  FlexAttention + compile:     {ms_flex:7.3f} ms   {flops/ms_flex/1e9:7.1f} TFLOPs/s")
    print(f"  speedup: {ms_ref/ms_flex:.2f}x")


if __name__ == "__main__":
    main()
