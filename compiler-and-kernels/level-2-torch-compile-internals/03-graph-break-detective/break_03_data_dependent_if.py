"""Break 03 — data-dependent if.

`if tensor.sum() > 0:` requires the *value* to decide which branch runs.
Dynamo cannot trace both branches simultaneously without help.

Run as-is to see the break. Apply the FIX HERE and re-run with FULLGRAPH=1.
"""

from __future__ import annotations

import os

import torch


def model_forward(x: torch.Tensor) -> torch.Tensor:
    # ===== FIX HERE =====
    # Option A: torch.where — keeps both branches in the graph, picks
    #           per-element. Use when both branches are cheap.
    #               y = torch.where(x.sum() > 0, x + 1.0, x - 1.0)
    # Option B: torch.cond — keeps both branches as sub-graphs, picks one.
    #           Has overhead; use only when branches are large.
    #               y = torch.cond(x.sum() > 0,
    #                              lambda x: x + 1.0,
    #                              lambda x: x - 1.0, (x,))
    # Option C: restructure so the predicate is shape-derivable (often the
    #           best answer — the branch was really a config choice, not a
    #           runtime data check).
    if x.sum() > 0:
        y = x + 1.0
    else:
        y = x - 1.0
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
