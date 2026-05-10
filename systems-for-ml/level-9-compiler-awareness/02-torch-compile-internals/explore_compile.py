"""Drive Dynamo + Inductor on a tiny transformer block and inspect each
intermediate IR.

Three deliberate experiments:

  exp1_clean_compile   — well-behaved forward; one graph, one kernel.
  exp2_graph_break     — inserts a print(); Dynamo splits into two graphs.
  exp3_data_dependent  — branches on a tensor value; Dynamo emits a guard +
                         partial graph, falls back to eager on the branch.

Run with:
  TORCH_LOGS="dynamo,graph_breaks,recompiles,output_code" python explore_compile.py
"""

import os
import torch
import torch.nn as nn

os.environ.setdefault(
    "TORCH_LOGS",
    "graph_breaks,recompiles,output_code",
)


class Block(nn.Module):
    def __init__(self, d: int = 128, h: int = 4):
        super().__init__()
        self.h = h
        self.dh = d // h
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.ln = nn.LayerNorm(d)
        self.up = nn.Linear(d, 4 * d, bias=False)
        self.down = nn.Linear(4 * d, d, bias=False)

    def attn(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        qkv = self.qkv(x).reshape(b, t, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(b, t, d)
        return self.proj(out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln(x))
        x = x + self.down(torch.nn.functional.silu(self.up(self.ln(x))))
        return x


def exp1_clean_compile(model: nn.Module, x: torch.Tensor) -> None:
    print("\n=== exp1: clean compile, one graph expected ===")
    f = torch.compile(model, fullgraph=True)
    f(x)
    f(x)


def exp2_graph_break(d: int) -> None:
    print("\n=== exp2: graph break from print() inside forward ===")

    class Bad(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(d, d)

        def forward(self, x):
            x = self.lin(x)
            print("hi from forward; this causes a graph break")
            return x.relu()

    f = torch.compile(Bad())  # no fullgraph; allow break
    f(torch.randn(2, d))


def exp3_data_dependent(d: int) -> None:
    print("\n=== exp3: data-dependent control flow ===")

    class Branchy(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(d, d)

        def forward(self, x):
            if x.sum() > 0:  # depends on tensor value -> graph break / fallback
                return self.lin(x).relu()
            return self.lin(x).neg()

    f = torch.compile(Branchy())
    f(torch.randn(2, d))
    f(torch.randn(2, d) - 100)  # force the other branch; expect recompile or fallback


def main() -> None:
    torch.manual_seed(0)
    d, h, t, b = 128, 4, 8, 2
    model = Block(d, h)
    x = torch.randn(b, t, d)

    exp1_clean_compile(model, x)
    exp2_graph_break(d)
    exp3_data_dependent(d)

    print("\nDone. Re-read the log: every region of generated code is one compile unit.")


if __name__ == "__main__":
    main()
