"""Break 01 — .item() in a conditional.

Pattern: Python branches on a value extracted from a tensor. Dynamo cannot
trace the if because the predicate is a Python bool that depends on tensor
data; it cuts the graph.

Run as-is to see the break. Apply the FIX HERE and re-run with FULLGRAPH=1.
"""

from __future__ import annotations

import os

import torch


def model_forward(x: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
    y = x * 2

    # ===== FIX HERE =====
    # Option A: move the .item() to outside the compiled function and pass the
    #           Python bool in. The compiled function gets a constant.
    # Option B: keep math in tensor land:
    #               y = torch.where(y.sum() > threshold, y + 1.0, y - 1.0)
    # Option C: torch._dynamo.config.capture_scalar_outputs = True
    #           lets Dynamo carry the scalar symbolically. Cheapest, has caveats.
    if y.sum().item() > threshold.item():
        y = y + 1.0
    else:
        y = y - 1.0
    # ====================

    return y.relu()


def main() -> None:
    fullgraph = os.environ.get("FULLGRAPH") == "1"
    compiled = torch.compile(model_forward, fullgraph=fullgraph)

    x = torch.randn(64)
    threshold = torch.tensor(0.0)

    if not fullgraph:
        # Explain mode: prints why Dynamo cut the graph.
        explanation = torch._dynamo.explain(model_forward)(x, threshold)
        print(explanation)
        print("---")
    out = compiled(x, threshold)
    print(f"ok, out shape: {tuple(out.shape)}, mean: {out.mean().item():.4f}")


if __name__ == "__main__":
    main()
