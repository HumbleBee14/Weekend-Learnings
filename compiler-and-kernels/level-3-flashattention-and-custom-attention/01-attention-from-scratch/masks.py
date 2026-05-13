"""Build the four masks we use throughout the level. NumPy only.

Run: python masks.py
"""
from __future__ import annotations

import numpy as np


def causal_mask(n: int) -> np.ndarray:
    """Lower-triangular: q can attend to kv when kv <= q."""
    q = np.arange(n)[:, None]
    kv = np.arange(n)[None, :]
    return q >= kv


def sliding_window_causal_mask(n: int, window: int) -> np.ndarray:
    """Causal band of width `window`."""
    q = np.arange(n)[:, None]
    kv = np.arange(n)[None, :]
    return (q >= kv) & (q - kv <= window)


def document_mask(doc_ids: np.ndarray) -> np.ndarray:
    """doc_ids: (N,) int array. Token attends only within its document."""
    return doc_ids[:, None] == doc_ids[None, :]


def sink_plus_window_mask(n: int, window: int, sinks: int) -> np.ndarray:
    """StreamingLLM-shaped: first `sinks` tokens are always visible,
    plus a causal sliding window of `window` tokens."""
    q = np.arange(n)[:, None]
    kv = np.arange(n)[None, :]
    causal = q >= kv
    window_ok = (q - kv) <= window
    sink_ok = kv < sinks
    return causal & (window_ok | sink_ok)


def sparsity(mask: np.ndarray) -> float:
    return float(np.mean(mask))


def main() -> None:
    n, window, sinks = 8192, 512, 4
    doc_ids = np.concatenate([np.full(2048, i) for i in range(4)])  # 4 docs

    masks = {
        "causal": causal_mask(n),
        "sliding_causal(W=512)": sliding_window_causal_mask(n, window),
        "document(4 docs of 2048)": document_mask(doc_ids),
        "sink+window(W=512, S=4)": sink_plus_window_mask(n, window, sinks),
    }
    print(f"N={n}")
    print(f"{'mask':>30} {'sparsity (frac True)':>22}")
    for name, m in masks.items():
        print(f"{name:>30} {sparsity(m):>22.4f}")


if __name__ == "__main__":
    main()
