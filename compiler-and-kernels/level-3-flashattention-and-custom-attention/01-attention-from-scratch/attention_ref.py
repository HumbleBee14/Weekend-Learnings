"""Reference NumPy attention. The ground truth for every kernel in this level.

Run: python attention_ref.py
"""
from __future__ import annotations

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=axis, keepdims=True)


def attention_ref(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    mask: np.ndarray | None = None,
    scale: float | None = None,
) -> np.ndarray:
    """Naive scaled dot-product attention. (N, d) inputs, (N, d) output.

    Materializes the full (N, N) score matrix S. Use this only as a reference
    for correctness — at N >= ~4096 you will feel the memory wall.
    """
    n, d = q.shape
    if scale is None:
        scale = 1.0 / np.sqrt(d)
    s = (q @ k.T) * scale  # (N, N) — the tensor FlashAttention refuses to materialize
    if mask is not None:
        # mask is True where attention is allowed.
        s = np.where(mask, s, -np.inf)
    p = softmax(s, axis=-1)
    o = p @ v
    return o


def attention_ref_mh(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Multi-head variant. Inputs are (B, H, N, d). Mask broadcasts over (B, H)."""
    *_, d = q.shape
    scale = 1.0 / np.sqrt(d)
    s = np.einsum("bhnd,bhmd->bhnm", q, k) * scale
    if mask is not None:
        s = np.where(mask, s, -np.inf)
    p = softmax(s, axis=-1)
    o = np.einsum("bhnm,bhmd->bhnd", p, v)
    return o


def _smoke_test() -> None:
    rng = np.random.default_rng(0)
    n, d = 64, 16
    q = rng.standard_normal((n, d)).astype(np.float32)
    k = rng.standard_normal((n, d)).astype(np.float32)
    v = rng.standard_normal((n, d)).astype(np.float32)
    o = attention_ref(q, k, v)
    # rows of P sum to 1, so |O|_row <= max(|V|_row)
    assert o.shape == (n, d)
    assert np.all(np.isfinite(o))

    # Causal mask check.
    mask = np.tril(np.ones((n, n), dtype=bool))
    o_causal = attention_ref(q, k, v, mask=mask)
    # First token only attends to itself => O[0] == V[0]
    np.testing.assert_allclose(o_causal[0], v[0], atol=1e-5)
    print("attention_ref smoke test passed")


if __name__ == "__main__":
    _smoke_test()
