"""Break 05 — .tolist() / numpy round-trip.

A common pattern: extract tensor values to Python lists, do some math in
numpy, push back into a tensor. Every conversion is a graph break.

Run as-is, then apply the FIX HERE and re-run with FULLGRAPH=1.
"""

from __future__ import annotations

import os

import numpy as np
import torch


def model_forward(x: torch.Tensor) -> torch.Tensor:
    y = x * 2
    # ===== FIX HERE =====
    # The whole point of this code was 'square then add'. Rewrite in torch:
    #     y = y * y + 1.0
    # Original — every line breaks the graph:
    vals = y.tolist()
    sq = np.array(vals) ** 2
    y = torch.from_numpy(sq).to(x.device).float() + 1.0
    # ====================
    return y.relu()


def main() -> None:
    fullgraph = os.environ.get("FULLGRAPH") == "1"
    compiled = torch.compile(model_forward, fullgraph=fullgraph)

    x = torch.randn(64)

    if not fullgraph:
        explanation = torch._dynamo.explain(model_forward)(x)
        print(explanation)
        print("---")
    out = compiled(x)
    print(f"ok, out shape: {tuple(out.shape)}")


if __name__ == "__main__":
    main()
