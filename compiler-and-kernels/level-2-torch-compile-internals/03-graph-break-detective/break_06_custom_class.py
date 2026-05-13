"""Break 06 — custom Python class flowing through forward.

A custom class holding tensors is opaque to Dynamo unless it is registered
as a pytree node. The fix is one line.

Run as-is, then apply the FIX HERE and re-run with FULLGRAPH=1.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
from torch.utils import _pytree as pytree


@dataclass
class HiddenState:
    h: torch.Tensor
    aux: torch.Tensor


# ===== FIX HERE =====
# Register HiddenState as a pytree node. Dynamo can now flatten/unflatten it.
#
# def _flatten(s: HiddenState):
#     return [s.h, s.aux], None
# def _unflatten(values, _):
#     return HiddenState(values[0], values[1])
# pytree.register_pytree_node(HiddenState, _flatten, _unflatten)
# ====================


def model_forward(s: HiddenState) -> torch.Tensor:
    return (s.h * 2.0 + s.aux).relu()


def main() -> None:
    fullgraph = os.environ.get("FULLGRAPH") == "1"
    compiled = torch.compile(model_forward, fullgraph=fullgraph)

    s = HiddenState(h=torch.randn(64), aux=torch.randn(64))

    if not fullgraph:
        try:
            explanation = torch._dynamo.explain(model_forward)(s)
            print(explanation)
            print("---")
        except Exception as e:
            print(f"explain itself failed: {e}\n---")
    out = compiled(s)
    print(f"ok, out shape: {tuple(out.shape)}")


if __name__ == "__main__":
    main()
