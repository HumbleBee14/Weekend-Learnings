"""Measure effective unified memory bandwidth on Apple Silicon.

Two probes:
  1. CPU-only memcpy-style read (NumPy) — what the CPU sees alone.
  2. GPU-only large matmul on MLX — what the GPU sees alone.
  3. Both at once — the contention you actually pay in production.

Run on any M-series Mac. No extras beyond `pip install mlx numpy`.
"""

from __future__ import annotations

import time
import threading

import numpy as np

try:
    import mlx.core as mx
except ImportError as e:
    raise SystemExit("This script requires MLX: `pip install mlx`") from e


def cpu_bandwidth_gb_s(size_gb: float = 2.0, repeats: int = 5) -> float:
    """Sequential read bandwidth from a NumPy buffer."""
    n = int(size_gb * 1024**3 // 8)  # float64 elements
    a = np.ones(n, dtype=np.float64)
    # warm
    _ = a.sum()
    t0 = time.perf_counter()
    for _ in range(repeats):
        _ = a.sum()
    dt = time.perf_counter() - t0
    bytes_read = repeats * n * 8
    return bytes_read / dt / 1e9


def gpu_bandwidth_gb_s(side: int = 8192, repeats: int = 20) -> float:
    """Square matmul tok/s ceiling, expressed as memory bandwidth.

    A matmul of size (N x N) reads ~ 2 * N*N * 4 bytes (fp32) per call.
    """
    a = mx.random.normal((side, side), dtype=mx.float32)
    b = mx.random.normal((side, side), dtype=mx.float32)
    mx.eval(a, b)
    # warm
    c = a @ b
    mx.eval(c)
    t0 = time.perf_counter()
    for _ in range(repeats):
        c = a @ b
        mx.eval(c)
    dt = time.perf_counter() - t0
    bytes_read = repeats * 2 * side * side * 4
    return bytes_read / dt / 1e9


def contention_test(seconds: float = 3.0):
    """Run CPU sweep and GPU matmul concurrently, measure both."""
    cpu_done = {"bw": 0.0}
    gpu_done = {"bw": 0.0}

    def cpu_worker():
        n = int(1.0 * 1024**3 // 8)
        a = np.ones(n, dtype=np.float64)
        bytes_read = 0
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < seconds:
            _ = a.sum()
            bytes_read += n * 8
        cpu_done["bw"] = bytes_read / (time.perf_counter() - t0) / 1e9

    def gpu_worker():
        side = 8192
        a = mx.random.normal((side, side), dtype=mx.float32)
        b = mx.random.normal((side, side), dtype=mx.float32)
        mx.eval(a, b)
        bytes_read = 0
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < seconds:
            c = a @ b
            mx.eval(c)
            bytes_read += 2 * side * side * 4
        gpu_done["bw"] = bytes_read / (time.perf_counter() - t0) / 1e9

    tc = threading.Thread(target=cpu_worker)
    tg = threading.Thread(target=gpu_worker)
    tc.start()
    tg.start()
    tc.join()
    tg.join()
    return cpu_done["bw"], gpu_done["bw"]


def main():
    print("== UMA bandwidth probe ==\n")

    cpu_alone = cpu_bandwidth_gb_s()
    print(f"CPU alone:                {cpu_alone:7.1f} GB/s")

    gpu_alone = gpu_bandwidth_gb_s()
    print(f"GPU alone (matmul-implied): {gpu_alone:7.1f} GB/s")

    cpu_under_load, gpu_under_load = contention_test()
    print(f"CPU under contention:     {cpu_under_load:7.1f} GB/s "
          f"({cpu_under_load / cpu_alone:.0%} of alone)")
    print(f"GPU under contention:     {gpu_under_load:7.1f} GB/s "
          f"({gpu_under_load / gpu_alone:.0%} of alone)")
    print(f"Sum during contention:    {cpu_under_load + gpu_under_load:7.1f} GB/s")
    print("\nWhen this sum approaches 'GPU alone', the controller is saturated")
    print("and CPU work directly steals from GPU decode bandwidth.")


if __name__ == "__main__":
    main()
