"""Inspect the lowering chain on a tiny model.

Run on CPU; everything below CPU codegen is logged but the kernels that
actually execute are C++/OpenMP. To see Triton output, run on a GPU box
and rerun with the same script — the env var path is identical.

Goal: see four IRs in one run.
  1. Module source        (Python)
  2. FX graph             (Dynamo output)
  3. Joint / post-grad    (AOTAutograd output)
  4. Generated kernel     (Inductor codegen — Triton or C++)
"""

import os
import torch
import torch.nn as nn

# Turn on every log we care about. These names are stable in PyTorch 2.5+.
os.environ.setdefault(
    "TORCH_LOGS",
    "graph,graph_code,aot_graphs,output_code,recompiles,graph_breaks",
)


class TinyBlock(nn.Module):
    """One linear -> SiLU -> linear. Smallest thing that fuses."""

    def __init__(self, dim: int = 64):
        super().__init__()
        self.up = nn.Linear(dim, dim * 2, bias=False)
        self.down = nn.Linear(dim * 2, dim, bias=False)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.act(self.up(x)))


def main() -> None:
    torch.manual_seed(0)
    model = TinyBlock()
    x = torch.randn(4, 64)

    # Eager — establish a numerical baseline.
    eager_out = model(x)

    # Dynamo + Inductor. mode="reduce-overhead" enables CUDA graphs on GPU
    # but is harmless on CPU; "max-autotune" would force kernel autotuning.
    compiled = torch.compile(model, backend="inductor", fullgraph=True)
    compiled_out = compiled(x)  # first call: trace + compile
    compiled_out = compiled(x)  # second call: hit the cache

    diff = (eager_out - compiled_out).abs().max().item()
    print(f"\nmax |eager - compiled| = {diff:.3e}")
    print("If this is ~0 (within fp tolerance), correctness is preserved.")
    print("Scroll up: you should see GRAPH, AOT_GRAPHS, and OUTPUT_CODE sections.")


if __name__ == "__main__":
    main()
