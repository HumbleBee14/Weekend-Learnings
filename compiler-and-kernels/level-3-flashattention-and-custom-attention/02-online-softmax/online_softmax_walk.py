"""Step through online softmax tile-by-tile. Print the running state. No GPU.

Run: python online_softmax_walk.py
"""
from __future__ import annotations

import numpy as np


def three_pass_softmax(x: np.ndarray) -> tuple[float, float, np.ndarray]:
    m = float(np.max(x))
    e = np.exp(x - m)
    ell = float(np.sum(e))
    return m, ell, e / ell


def online_softmax(x: np.ndarray, tile: int = 4, verbose: bool = True) -> tuple[float, float]:
    """Process x in tiles. Return final (m, ell). Print state per tile."""
    m = -np.inf
    ell = 0.0
    n = x.shape[0]
    for t_start in range(0, n, tile):
        t_end = min(t_start + tile, n)
        x_tile = x[t_start:t_end]
        tile_max = float(np.max(x_tile))
        m_new = max(m, tile_max)
        # Rescale factor for the previously-accumulated denominator.
        rescale = float(np.exp(m - m_new)) if m != -np.inf else 0.0
        tile_sum = float(np.sum(np.exp(x_tile - m_new)))
        ell_new = ell * rescale + tile_sum
        if verbose:
            print(
                f"tile=[{t_start}:{t_end}] x_tile={x_tile.tolist()}  "
                f"m: {m:.4f} -> {m_new:.4f}  "
                f"rescale=exp({m:.4f}-{m_new:.4f})={rescale:.4f}  "
                f"tile_sum={tile_sum:.4f}  ell: {ell:.4f} -> {ell_new:.4f}"
            )
        m, ell = m_new, ell_new
    return m, ell


def main() -> None:
    x = np.array([1.0, 2.0, 4.0, 3.5, 0.5, 5.0, 4.8, 2.2], dtype=np.float64)

    print("=== Three-pass reference ===")
    m_ref, ell_ref, p_ref = three_pass_softmax(x)
    print(f"m = {m_ref:.4f}, ell = {ell_ref:.4f}")
    print(f"softmax = {np.round(p_ref, 5).tolist()}")
    print()

    print("=== Online, tile=4 ===")
    m_on, ell_on = online_softmax(x, tile=4)
    print(f"final m = {m_on:.4f}, ell = {ell_on:.4f}")
    print()

    assert abs(m_on - m_ref) < 1e-10
    assert abs(ell_on - ell_ref) < 1e-10

    # Try tile sizes that don't divide N.
    print("=== Online, tile=3 (doesn't divide N=8) ===")
    m2, ell2 = online_softmax(x, tile=3)
    assert abs(m2 - m_ref) < 1e-10
    assert abs(ell2 - ell_ref) < 1e-10
    print("ok")
    print()

    print("=== Online, tile=1 (per-element streaming) ===")
    m3, ell3 = online_softmax(x, tile=1, verbose=False)
    assert abs(m3 - m_ref) < 1e-10
    assert abs(ell3 - ell_ref) < 1e-10
    print("ok")


if __name__ == "__main__":
    main()
