"""Break 02 — print() inside forward.

print() has Python side effects. Dynamo treats it as opaque and cuts the graph.

Run as-is to see the break. Apply the FIX HERE and re-run with FULLGRAPH=1.
"""

from __future__ import annotations

import os

import torch


def model_forward(x: torch.Tensor) -> torch.Tensor:
    y = x * 2
    # ===== FIX HERE =====
    # Option A: delete the print. Use logging that you reorder out.
    # Option B: torch._dynamo.config.reorderable_logging_functions.add(print)
    #           This moves the print to the end of the traced region rather
    #           than breaking. Works for stateless logging.
    # Option C: torch.compiler.disable() the helper that does the printing.
    print("debug: y mean is", y.mean().item())
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
