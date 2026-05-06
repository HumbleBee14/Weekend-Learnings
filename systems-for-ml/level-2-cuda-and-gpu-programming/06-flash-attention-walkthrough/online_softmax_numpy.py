"""
The online softmax recursion, in pure NumPy. This is the heart of FlashAttention.

Compute softmax in one pass over the data, maintaining (running_max, running_sum)
and rescaling the output whenever a new max appears.

This is the same recursion that lives inside the inner loop of FlashAttention.
Once you've written it on the CPU it stops being mysterious on the GPU.

Run:
    python online_softmax_numpy.py
"""

import numpy as np


def softmax_naive(x: np.ndarray) -> np.ndarray:
    """The standard, two-pass softmax: subtract max, exp, divide."""
    m = x.max()
    e = np.exp(x - m)
    return e / e.sum()


def softmax_online(x: np.ndarray, tile_size: int = 64) -> np.ndarray:
    """
    Compute softmax in one streaming pass over `x`, processing `tile_size` elements at a time.

    State:
      m_old : running max
      l_old : running sum (in the rescaled space — see below)
      acc   : running output, normalized to use m_old as its max

    On each new tile:
      1. Find the tile's local max.
      2. Compute m_new = max(m_old, tile_max).
      3. Rescale the running sum and output: scale = exp(m_old - m_new).
      4. Add the tile's contribution to running sum and output.
      5. Update (m_old, l_old) = (m_new, l_new).

    At the end, divide acc by l to get the final softmax.
    """
    m_old = -np.inf
    l_old = 0.0
    acc = np.zeros_like(x, dtype=np.float64)   # the "unnormalized output"

    for start in range(0, len(x), tile_size):
        tile = x[start:start + tile_size].astype(np.float64)
        tile_max = tile.max()

        m_new = max(m_old, tile_max)
        scale = np.exp(m_old - m_new)
        tile_exp = np.exp(tile - m_new)
        l_new = l_old * scale + tile_exp.sum()

        # Rescale the running output, then add this tile's contribution.
        # Note: in FlashAttention this is where "P @ V" replaces "tile_exp", but the
        # bookkeeping is identical.
        acc[:start] *= scale          # rescale earlier elements
        acc[start:start + tile_size] = tile_exp

        m_old = m_new
        l_old = l_new

    return acc / l_old


def main():
    # Test against scipy on tricky inputs
    np.random.seed(0)

    test_cases = {
        "simple": np.array([1, 2, 3, 4, 5], dtype=np.float32),
        "with negatives": np.array([-2, -1, 0, 1, 2], dtype=np.float32),
        "needs subtract-max": np.random.randn(1000).astype(np.float32) * 50,
        "long with one big spike": np.concatenate([
            np.random.randn(2000).astype(np.float32),
            np.array([100.0], dtype=np.float32),
            np.random.randn(2000).astype(np.float32),
        ]),
        "large random": np.random.randn(8192).astype(np.float32),
    }

    print(f"{'case':<28} {'tile_size':<10} {'max_err':<12}")
    print("-" * 50)
    for name, x in test_cases.items():
        ref = softmax_naive(x)
        for tile_size in [16, 64, 256]:
            online = softmax_online(x, tile_size=tile_size)
            err = np.max(np.abs(ref - online))
            print(f"{name:<28} {tile_size:<10} {err:.2e}")


if __name__ == "__main__":
    main()
