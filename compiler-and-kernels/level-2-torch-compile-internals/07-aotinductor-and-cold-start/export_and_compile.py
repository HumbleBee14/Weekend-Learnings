"""Export a small model with torch.export and package it via AOTInductor.

Produces ./packaged_model.pt2.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
from torch.export import Dim


class TinyTransformerBlock(nn.Module):
    def __init__(self, dim: int = 1024) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.o = nn.Linear(dim, dim, bias=False)
        self.norm2 = nn.LayerNorm(dim)
        self.w1 = nn.Linear(dim, 4 * dim, bias=False)
        self.w2 = nn.Linear(4 * dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        q, k, v = self.q(h), self.k(h), self.v(h)
        attn = torch.nn.functional.scaled_dot_product_attention(
            q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1)
        ).squeeze(1)
        x = x + self.o(attn)
        h = self.norm2(x)
        h = torch.nn.functional.silu(self.w1(h))
        return x + self.w2(h)


def main() -> None:
    assert torch.cuda.is_available(), "AOTInductor demo expects CUDA"
    device = "cuda"
    dtype = torch.bfloat16

    torch.manual_seed(0)
    model = TinyTransformerBlock(dim=1024).to(device=device, dtype=dtype).eval()
    example = torch.randn(8, 1024, device=device, dtype=dtype)

    # Declare batch as dynamic, dim is fixed.
    batch = Dim("batch", min=1, max=64)

    with torch.inference_mode():
        ep = torch.export.export(
            model, (example,),
            dynamic_shapes={"x": {0: batch}},
        )
        path = os.path.abspath("./packaged_model.pt2")
        torch._inductor.aoti_compile_and_package(
            ep,
            package_path=path,
        )

    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"wrote {path}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
