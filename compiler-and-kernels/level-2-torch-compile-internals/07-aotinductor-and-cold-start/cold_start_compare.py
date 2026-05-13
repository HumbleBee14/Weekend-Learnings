"""Compare cold-start: eager, JIT torch.compile, AOTInductor.

Run after export_and_compile.py has produced packaged_model.pt2.
"""

from __future__ import annotations

import time

import torch

from export_and_compile import TinyTransformerBlock


def cold_eager() -> float:
    model = TinyTransformerBlock(dim=1024).to("cuda", dtype=torch.bfloat16).eval()
    x = torch.randn(8, 1024, device="cuda", dtype=torch.bfloat16)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        y = model(x)
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def cold_jit_compile() -> float:
    model = TinyTransformerBlock(dim=1024).to("cuda", dtype=torch.bfloat16).eval()
    compiled = torch.compile(model, fullgraph=True)
    x = torch.randn(8, 1024, device="cuda", dtype=torch.bfloat16)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        y = compiled(x)
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def cold_aotinductor() -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    loaded = torch._inductor.aoti_load_package("./packaged_model.pt2")
    x = torch.randn(8, 1024, device="cuda", dtype=torch.bfloat16)
    y = loaded(x)
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def main() -> None:
    assert torch.cuda.is_available()
    print(f"{'path':25s}  {'cold start (s)':>16s}")
    print("-" * 45)
    print(f"{'eager':25s}  {cold_eager():16.3f}")
    print(f"{'torch.compile (JIT)':25s}  {cold_jit_compile():16.3f}")
    print(f"{'AOTInductor':25s}  {cold_aotinductor():16.3f}")


if __name__ == "__main__":
    main()
