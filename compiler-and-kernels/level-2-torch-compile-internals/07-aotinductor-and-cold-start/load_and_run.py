"""Load the .pt2 archive and run inference. Measures load + first-call latency."""

from __future__ import annotations

import time

import torch


def main() -> None:
    device = "cuda"
    dtype = torch.bfloat16

    t0 = time.perf_counter()
    loaded = torch._inductor.aoti_load_package("./packaged_model.pt2")
    t_load = time.perf_counter() - t0

    x = torch.randn(8, 1024, device=device, dtype=dtype)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    y = loaded(x)
    torch.cuda.synchronize()
    t_first = time.perf_counter() - t0

    # Steady state
    for _ in range(5):
        y = loaded(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    n = 100
    for _ in range(n):
        y = loaded(x)
    torch.cuda.synchronize()
    t_steady = (time.perf_counter() - t0) / n

    print(f"load: {t_load*1000:.1f} ms")
    print(f"first inference: {t_first*1000:.1f} ms")
    print(f"steady state: {t_steady*1000:.3f} ms")
    print(f"output mean: {y.float().mean().item():.4f}")


if __name__ == "__main__":
    main()
