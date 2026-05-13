"""FA2 forward in NumPy. The algorithm in 60 lines. No GPU.

Run: python fa2_numpy.py
"""
from __future__ import annotations

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "01-attention-from-scratch"))
from attention_ref import attention_ref  # noqa: E402


def fa2_forward_numpy(
    Q: np.ndarray,  # (N, d)
    K: np.ndarray,  # (N, d)
    V: np.ndarray,  # (N, d)
    block_m: int = 64,
    block_n: int = 64,
    causal: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (O, L) where L = m + log(ell) is the log-sum-exp per query row,
    matching what FA2 saves for the backward pass."""
    N, d = Q.shape
    scale = 1.0 / np.sqrt(d)

    O = np.zeros_like(Q)
    L = np.zeros(N, dtype=Q.dtype)

    for i in range(0, N, block_m):  # outer: parallel across these in the kernel
        i_end = min(i + block_m, N)
        Q_i = Q[i:i_end]  # (bm, d)
        bm = Q_i.shape[0]

        m_i = np.full(bm, -np.inf, dtype=np.float64)
        l_i = np.zeros(bm, dtype=np.float64)
        O_i = np.zeros((bm, d), dtype=np.float64)

        for j in range(0, N, block_n):
            j_end = min(j + block_n, N)
            K_j = K[j:j_end]
            V_j = V[j:j_end]

            S = (Q_i @ K_j.T) * scale  # (bm, bn)

            if causal:
                q_idx = np.arange(i, i_end)[:, None]
                kv_idx = np.arange(j, j_end)[None, :]
                S = np.where(q_idx >= kv_idx, S, -np.inf)

            tile_max = np.max(S, axis=-1)  # (bm,)
            m_new = np.maximum(m_i, tile_max)

            # If a whole row of the tile is -inf (fully masked), m_new stays at m_i (or -inf).
            # exp(-inf - -inf) is nan; clamp that explicitly.
            rescale = np.where(m_i == -np.inf, 0.0, np.exp(m_i - m_new))
            rescale = np.where(m_new == -np.inf, 0.0, rescale)

            # Tile contributions (with new max).
            P = np.exp(S - m_new[:, None])  # rows that are fully masked become 0 here
            P = np.where(np.isfinite(P), P, 0.0)
            tile_sum = np.sum(P, axis=-1)

            # ORDER: rescale O and l first, then add new tile.
            l_i = l_i * rescale + tile_sum
            O_i = O_i * rescale[:, None] + P @ V_j
            m_i = m_new

        # Normalize.
        safe_l = np.where(l_i == 0, 1.0, l_i)
        O[i:i_end] = (O_i / safe_l[:, None]).astype(Q.dtype)
        L[i:i_end] = (m_i + np.log(safe_l)).astype(Q.dtype)

    return O, L


def main() -> None:
    rng = np.random.default_rng(42)
    N, d = 256, 64
    Q = rng.standard_normal((N, d)).astype(np.float64)
    K = rng.standard_normal((N, d)).astype(np.float64)
    V = rng.standard_normal((N, d)).astype(np.float64)

    O_ref = attention_ref(Q, K, V)

    for bm, bn in [(32, 32), (64, 64), (128, 128), (17, 23)]:  # try non-divisible
        O_fa2, L_fa2 = fa2_forward_numpy(Q, K, V, block_m=bm, block_n=bn)
        err = float(np.max(np.abs(O_fa2 - O_ref)))
        print(f"block_m={bm:>4} block_n={bn:>4}  max_err = {err:.2e}")
        assert err < 1e-10, "FA2 numpy drifted; check rescale order or boundary handling"

    # Causal sanity: first query attends only to itself.
    O_causal, _ = fa2_forward_numpy(Q, K, V, block_m=64, block_n=64, causal=True)
    np.testing.assert_allclose(O_causal[0], V[0], atol=1e-10)
    print("\nfa2_numpy matches attention_ref for every block size, including causal. ship it.")


if __name__ == "__main__":
    main()
