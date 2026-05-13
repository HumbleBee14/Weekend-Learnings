"""Ground-truth NumPy implementation of sink + sliding-window + ALiBi causal attention.

Every other impl must match this to bf16 tolerance.

Run:
    python reference.py
"""
from __future__ import annotations

import math

import numpy as np


def alibi_slopes(num_heads: int) -> np.ndarray:
    def slopes_pow2(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio ** i for i in range(n)]
    if math.log2(num_heads).is_integer():
        return np.array(slopes_pow2(num_heads), dtype=np.float64)
    base = slopes_pow2(2 ** int(math.log2(num_heads)))
    return np.array(base + [1.0] * (num_heads - len(base)), dtype=np.float64)


def sink_window_causal_mask(n: int, window: int, sinks: int) -> np.ndarray:
    q = np.arange(n)[:, None]
    kv = np.arange(n)[None, :]
    causal = q >= kv
    in_window = (q - kv) <= window
    is_sink = kv < sinks
    return causal & (in_window | is_sink)


def reference_attention(
    q: np.ndarray,  # (B, H, N, D)
    k: np.ndarray,
    v: np.ndarray,
    window: int,
    sinks: int,
    slopes: np.ndarray | None = None,
) -> np.ndarray:
    B, H, N, D = q.shape
    scale = 1.0 / np.sqrt(D)
    if slopes is None:
        slopes = alibi_slopes(H)

    s = np.einsum("bhnd,bhmd->bhnm", q, k) * scale  # (B, H, N, N)

    # ALiBi bias.
    q_idx = np.arange(N)[None, None, :, None]
    kv_idx = np.arange(N)[None, None, None, :]
    bias = -slopes.reshape(1, H, 1, 1) * np.abs(q_idx - kv_idx)
    s = s + bias

    # Mask.
    mask = sink_window_causal_mask(N, window, sinks)
    s = np.where(mask[None, None, :, :], s, -np.inf)

    # Stable softmax.
    m = np.max(s, axis=-1, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    e = np.exp(s - m)
    e = np.where(np.isfinite(e), e, 0.0)
    p = e / (np.sum(e, axis=-1, keepdims=True) + 1e-30)
    return np.einsum("bhnm,bhmd->bhnd", p, v)


def _smoke_test() -> None:
    rng = np.random.default_rng(0)
    B, H, N, D = 1, 4, 128, 32
    q = rng.standard_normal((B, H, N, D)).astype(np.float64)
    k = rng.standard_normal((B, H, N, D)).astype(np.float64)
    v = rng.standard_normal((B, H, N, D)).astype(np.float64)
    o = reference_attention(q, k, v, window=32, sinks=2)
    assert o.shape == (B, H, N, D)
    assert np.all(np.isfinite(o))
    # First token attends only to itself (it's a sink; window distance 0).
    np.testing.assert_allclose(o[0, 0, 0], v[0, 0, 0], atol=1e-10)
    print("reference attention smoke test passed")


if __name__ == "__main__":
    _smoke_test()
