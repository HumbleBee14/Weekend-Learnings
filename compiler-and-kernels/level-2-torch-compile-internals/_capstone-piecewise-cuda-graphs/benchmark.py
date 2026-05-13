"""Benchmark five variants of a LLaMA decoder block.

Variants:
  1. Eager
  2. torch.compile(mode="default") — may have graph breaks if attention is
     not wrapped; with install_custom_attn() it doesn't.
  3. torch.compile(mode="default", fullgraph=True) — clean compile, no graph
     break, but no CUDA graph capture
  4. torch.compile(mode="reduce-overhead") — full CUDA graph capture
  5. Piecewise wrapper — the capstone implementation

Fill in your numbers in notes.md.
"""

from __future__ import annotations

import time

import torch

from llama_block import make_block
from piecewise_wrapper import PiecewiseCUDAGraph, install_custom_attn


def bench(fn, *args, n_warm: int = 10, n: int = 50) -> float:
    for _ in range(n_warm):
        y = fn(*args)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        y = fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / n


@torch.inference_mode()
def main() -> None:
    assert torch.cuda.is_available()
    install_custom_attn()

    hidden = 2048  # fit T4
    block = make_block(hidden=hidden)

    cases = {
        "prefill (B=1, S=128)": torch.randn(1, 128, hidden, device="cuda", dtype=torch.bfloat16),
        "decode  (B=1, S=1)":   torch.randn(1, 1,   hidden, device="cuda", dtype=torch.bfloat16),
    }

    print(f"{'variant':40s}  " + "  ".join(f"{k:>22s}" for k in cases))
    print("-" * 110)

    # 1. Eager
    row = [f"{'eager':40s}"]
    for k, x in cases.items():
        row.append(f"{bench(block, x):22.3f}")
    print("  ".join(row))

    # 2. compile default
    compiled = torch.compile(block, mode="default")
    row = [f"{'compile(default)':40s}"]
    for k, x in cases.items():
        row.append(f"{bench(compiled, x):22.3f}")
    print("  ".join(row))

    # 3. compile fullgraph
    compiled_fg = torch.compile(block, mode="default", fullgraph=True)
    row = [f"{'compile(default, fullgraph=True)':40s}"]
    for k, x in cases.items():
        row.append(f"{bench(compiled_fg, x):22.3f}")
    print("  ".join(row))

    # 4. compile reduce-overhead (full CUDA graph)
    compiled_ro = torch.compile(block, mode="reduce-overhead", fullgraph=True)
    row = [f"{'compile(reduce-overhead)':40s}"]
    for k, x in cases.items():
        row.append(f"{bench(compiled_ro, x):22.3f}")
    print("  ".join(row))

    # 5. Piecewise
    piecewise = PiecewiseCUDAGraph(block)
    row = [f"{'piecewise (yours)':40s}"]
    for k, x in cases.items():
        row.append(f"{bench(piecewise, x):22.3f}")
    print("  ".join(row))


if __name__ == "__main__":
    main()
