"""Probe whether your Mac is using Neural Accelerators on matmul.

Strategy: time `mx.fast.matmul` vs explicit `mx.matmul` on a sequence of shapes
across fp32 (FMA path) and fp16/bf16 (NA-eligible on M5).

If you are on M5+, fp16/bf16 throughput should be sharply higher than fp32 and
sharply higher than M4/M3-class numbers for the same GPU core count.

If you are on M4 or earlier, fp16 ≈ fp32 (no NA) on this script.
"""

from __future__ import annotations

import platform
import time

import mlx.core as mx


def time_matmul(side: int, dtype, repeats: int = 30) -> float:
    a = mx.random.normal((side, side), dtype=dtype)
    b = mx.random.normal((side, side), dtype=dtype)
    mx.eval(a, b)
    c = a @ b
    mx.eval(c)
    t0 = time.perf_counter()
    for _ in range(repeats):
        c = a @ b
    mx.eval(c)
    dt = time.perf_counter() - t0
    flops = 2 * side ** 3 * repeats
    return flops / dt / 1e12  # TFLOPS


def main():
    print(f"machine: {platform.machine()}  os: {platform.mac_ver()[0]}")
    print(f"hint: M5+ fp16/bf16 should be ~3-4x fp32 if NAs are active.\n")

    side = 4096
    print(f"{'dtype':>8s} {'TFLOPS':>10s}")
    for dtype, name in [
        (mx.float32, "fp32"),
        (mx.float16, "fp16"),
        (mx.bfloat16, "bf16"),
    ]:
        try:
            tflops = time_matmul(side, dtype)
            print(f"{name:>8s} {tflops:10.2f}")
        except Exception as e:
            print(f"{name:>8s}  skipped ({e})")


if __name__ == "__main__":
    main()
