"""FlexAttention implementation of sink + sliding-window + ALiBi + causal.

10 lines of attention logic; everything else is plumbing.

Run:
    python impl_flex.py
"""
from __future__ import annotations

import math

import torch
from torch.nn.attention.flex_attention import flex_attention, create_block_mask


def make_capstone_flex(N: int, num_heads: int, window: int, sinks: int):
    slopes = torch.tensor(_alibi_slopes(num_heads), device="cuda", dtype=torch.float32)

    def score_mod(score, b, h, q_idx, kv_idx):
        return score - slopes[h] * (q_idx - kv_idx).abs()

    def mask_mod(b, h, q_idx, kv_idx):
        causal = q_idx >= kv_idx
        in_window = (q_idx - kv_idx) <= window
        is_sink = kv_idx < sinks
        return causal & (in_window | is_sink)

    block_mask = create_block_mask(mask_mod, B=None, H=None, Q_LEN=N, KV_LEN=N, device="cuda")
    flex = torch.compile(flex_attention, dynamic=False)
    return flex, score_mod, block_mask


def _alibi_slopes(n):
    def power_of_2(m):
        start = 2 ** (-(2 ** -(math.log2(m) - 3)))
        r = start
        return [start * r ** i for i in range(m)]
    if math.log2(n).is_integer():
        return power_of_2(n)
    base = power_of_2(2 ** int(math.log2(n)))
    return base + [1.0] * (n - len(base))


def capstone_attention_flex(q, k, v, window, sinks):
    B, H, N, D = q.shape
    flex, score_mod, block_mask = make_capstone_flex(N, H, window, sinks)
    return flex(q, k, v, score_mod=score_mod, block_mask=block_mask)


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required.")
    from reference import reference_attention, alibi_slopes

    import numpy as np
    np.random.seed(0)
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
    o_flex = capstone_attention_flex(q, k, v, window=W, sinks=S)
    err = float((torch.from_numpy(o_ref).cuda().float() - o_flex.float()).abs().max())
    print(f"max abs err vs reference (fp32): {err:.2e}")
    assert err < 1e-3, "FlexAttention capstone disagrees with reference; debug the mask/score"


if __name__ == "__main__":
    main()
