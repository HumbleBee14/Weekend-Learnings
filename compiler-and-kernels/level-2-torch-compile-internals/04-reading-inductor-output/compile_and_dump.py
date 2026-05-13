"""Compile a small block and dump Inductor output for reading.

Run:
    TORCH_COMPILE_DEBUG=1 python compile_and_dump.py

Then read /tmp/torchinductor_<user>/.../output_code.py.

If you want only the FX graph (no Inductor lowering), set
    TORCH_LOGS="aot_graphs"
If you want the recompilation log:
    TORCH_LOGS="recompiles"
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # mean of squares -> rsqrt -> scale
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


class TinyBlock(nn.Module):
    def __init__(self, dim: int = 1024) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, dim, bias=False)
        self.norm = RMSNorm(dim)
        self.w2 = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.w1(x)
        h = self.norm(h)
        h = F.gelu(h)
        return self.w2(h)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    torch.manual_seed(0)
    block = TinyBlock(dim=1024).to(device=device, dtype=dtype).eval()
    x = torch.randn(8, 1024, device=device, dtype=dtype)

    with torch.inference_mode():
        compiled = torch.compile(block, mode="default", fullgraph=True)
        # First call triggers compile + dump
        y = compiled(x)
        # Second call confirms cache hit
        y = compiled(x)

    print(f"output mean: {y.float().mean().item():.4f}")
    print()
    print("Look for output_code.py under /tmp/torchinductor_<user>/")
    print("  find /tmp/torchinductor_$USER -name output_code.py")


if __name__ == "__main__":
    main()
