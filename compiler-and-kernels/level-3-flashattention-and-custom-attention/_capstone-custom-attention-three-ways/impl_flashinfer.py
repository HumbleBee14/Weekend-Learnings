"""FlashInfer implementation of sink + sliding-window + ALiBi causal.

Uses BatchPrefillWithRaggedKVCacheWrapper with a custom mask + ALiBi via
the wrapper's positional-bias hook (or a JIT variant if needed).

Run:
    python impl_flashinfer.py
"""
from __future__ import annotations

import math

import torch

try:
    import flashinfer
except ImportError:
    raise SystemExit("pip install flashinfer")


def _alibi_slopes(n):
    def power_of_2(m):
        start = 2 ** (-(2 ** -(math.log2(m) - 3)))
        r = start
        return [start * r ** i for i in range(m)]
    if math.log2(n).is_integer():
        return power_of_2(n)
    base = power_of_2(2 ** int(math.log2(n)))
    return base + [1.0] * (n - len(base))


def build_sink_window_mask(n: int, window: int, sinks: int) -> torch.Tensor:
    """Dense (N, N) bool mask. FlashInfer accepts it via custom_mask.

    For very large N you'd build a BSR sparse representation instead; this is
    the simplest correct version for the capstone benchmark."""
    q = torch.arange(n)[:, None]
    kv = torch.arange(n)[None, :]
    causal = q >= kv
    in_window = (q - kv) <= window
    is_sink = kv < sinks
    return causal & (in_window | is_sink)


def capstone_attention_flashinfer(q, k, v, window, sinks):
    """q, k, v: (B, H, N, D). Single-batch path here for simplicity; the real
    win of FlashInfer is on ragged batches — covered in sub-module 07."""
    B, H, N, D = q.shape
    assert B == 1, "this demo runs single-batch; ragged-batch is sub-module 07"

    # Flatten to (N, H, D) for FlashInfer's expected layout.
    q_flat = q[0].transpose(0, 1).contiguous()  # (N, H, D)
    k_flat = k[0].transpose(0, 1).contiguous()
    v_flat = v[0].transpose(0, 1).contiguous()

    qo_indptr = torch.tensor([0, N], dtype=torch.int32, device="cuda")
    kv_indptr = torch.tensor([0, N], dtype=torch.int32, device="cuda")

    mask = build_sink_window_mask(N, window, sinks).to("cuda")
    # FlashInfer custom_mask convention: dense bool per (q, kv); the wrapper
    # converts to its BSR internal layout via plan().
    mask_flat = mask.reshape(-1)  # (N*N,) bool

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    wrap = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(workspace, kv_layout="NHD")

    slopes = torch.tensor(_alibi_slopes(H), device="cuda", dtype=torch.float32)

    wrap.plan(
        qo_indptr=qo_indptr, kv_indptr=kv_indptr,
        num_qo_heads=H, num_kv_heads=H,
        head_dim_qk=D, head_dim_vo=D,
        custom_mask=mask_flat,
        # FlashInfer supports ALiBi slopes natively for many wrappers.
        pos_encoding_mode="ALIBI",
        # The slopes tensor is consumed via the planning-side metadata in
        # FlashInfer 0.2+. Earlier versions take it on run(). Adjust to your version.
    )
    out_flat = wrap.run(q_flat, k_flat, v_flat)  # (N, H, D)
    return out_flat.transpose(0, 1).unsqueeze(0)  # back to (B=1, H, N, D)


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

    q = torch.from_numpy(q_np).cuda().to(torch.bfloat16)
    k = torch.from_numpy(k_np).cuda().to(torch.bfloat16)
    v = torch.from_numpy(v_np).cuda().to(torch.bfloat16)

    try:
        o_fi = capstone_attention_flashinfer(q, k, v, window=W, sinks=S)
    except Exception as e:
        print(f"FlashInfer call failed (likely API version drift): {e}")
        print("Adjust pos_encoding_mode / slopes plumbing to your FlashInfer version.")
        return

    err = float((torch.from_numpy(o_ref).cuda().to(torch.bfloat16) - o_fi).float().abs().max())
    print(f"max abs err vs reference (bf16 tol): {err:.2e}")
    # bf16 has ~1e-2 floor; allow that.
    assert err < 5e-2, "FlashInfer capstone disagrees with reference; debug the mask/slopes"


if __name__ == "__main__":
    main()
