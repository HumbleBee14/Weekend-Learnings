"""One-row attention via the online softmax recursion on (m, ell, O).

This is the algorithmic core of FlashAttention's forward pass, in 40 lines of NumPy.
The Triton kernel in sub-module 03 is the same code, tiled across rows.

Run: python online_attention_np.py
"""
from __future__ import annotations

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "01-attention-from-scratch"))
from attention_ref import attention_ref  # noqa: E402


def online_attention_row(
    q: np.ndarray,  # (d,)
    K: np.ndarray,  # (N, d)
    V: np.ndarray,  # (N, d)
    tile: int = 64,
) -> np.ndarray:
    """Compute one row of softmax(QK^T / sqrt(d)) @ V via the online recursion.

    State: (m, ell, O).
        m: running max of the score row, scalar.
        ell: running denominator, scalar.
        O: running output accumulator, (d,).
    """
    n, d = K.shape
    scale = 1.0 / np.sqrt(d)
    m = -np.inf
    ell = 0.0
    O = np.zeros(d, dtype=np.float64)

    for t_start in range(0, n, tile):
        t_end = min(t_start + tile, n)
        K_tile = K[t_start:t_end]  # (tile, d)
        V_tile = V[t_start:t_end]  # (tile, d)

        # 1. New scores for this tile.
        s_tile = (q @ K_tile.T) * scale  # (tile,)

        # 2. Update running max.
        tile_max = float(np.max(s_tile))
        m_new = max(m, tile_max)

        # 3. Rescale factor for old (m, ell, O).
        rescale = float(np.exp(m - m_new)) if m != -np.inf else 0.0

        # 4. Tile contributions (with the new max).
        p_tile = np.exp(s_tile - m_new)  # (tile,)
        tile_sum = float(np.sum(p_tile))

        # 5. Update ell and O.
        # ORDER MATTERS: rescale O *before* adding the new tile's PV.
        # Get this backwards and the output drifts every time m changes.
        ell = ell * rescale + tile_sum
        O = O * rescale + p_tile @ V_tile

        # 6. Commit new max.
        m = m_new

    return O / ell


def online_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, tile: int = 64) -> np.ndarray:
    """Row-by-row online attention. Trivially parallelizable across rows."""
    n = Q.shape[0]
    out = np.zeros_like(Q)
    for i in range(n):
        out[i] = online_attention_row(Q[i], K, V, tile=tile)
    return out


def main() -> None:
    rng = np.random.default_rng(0)
    N, d = 128, 32
    Q = rng.standard_normal((N, d)).astype(np.float64)
    K = rng.standard_normal((N, d)).astype(np.float64)
    V = rng.standard_normal((N, d)).astype(np.float64)

    O_ref = attention_ref(Q, K, V)
    for tile in [1, 8, 16, 64, 128]:
        O_on = online_attention(Q, K, V, tile=tile)
        max_err = float(np.max(np.abs(O_on - O_ref)))
        print(f"tile={tile:>3}  max_err vs ref = {max_err:.2e}")
        assert max_err < 1e-10, "online attention drifted; check the rescale order"
    print("\nonline_attention matches attention_ref for every tile size. ship it.")


if __name__ == "__main__":
    main()
