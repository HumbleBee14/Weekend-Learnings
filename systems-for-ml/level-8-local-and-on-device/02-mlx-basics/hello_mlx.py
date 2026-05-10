"""MLX in five minutes.

Demonstrates:
  - lazy graph construction and explicit eval
  - basic ops, broadcasting, dtypes
  - autograd via mx.grad
  - a tiny "transformer-ish" matmul timing
"""

from __future__ import annotations

import time

import mlx.core as mx
import mlx.nn as nn


def lazy_eval_demo():
    print("-- lazy graph --")
    a = mx.array([1.0, 2.0, 3.0])
    b = mx.array([10.0, 20.0, 30.0])
    c = a * b + 1
    print(f"  c (before eval) is a graph node: {type(c).__name__}")
    mx.eval(c)
    print(f"  c (after eval): {c.tolist()}")


def autograd_demo():
    print("\n-- autograd --")

    def f(x):
        return (x ** 3 + 2 * x).sum()

    df = mx.grad(f)
    x = mx.array([1.0, 2.0, 3.0])
    g = df(x)
    mx.eval(g)
    # d/dx (x^3 + 2x) = 3x^2 + 2 -> [5, 14, 29]
    print(f"  grad of x^3 + 2x at [1,2,3] = {g.tolist()}")


def matmul_throughput():
    print("\n-- matmul throughput (proxy for decode bandwidth) --")
    side = 4096
    a = mx.random.normal((side, side), dtype=mx.float16)
    b = mx.random.normal((side, side), dtype=mx.float16)
    mx.eval(a, b)

    # warm
    c = a @ b
    mx.eval(c)

    repeats = 50
    t0 = time.perf_counter()
    for _ in range(repeats):
        c = a @ b
    mx.eval(c)
    dt = time.perf_counter() - t0

    flops = 2 * side ** 3 * repeats
    print(f"  {side}x{side} fp16 matmul: {flops / dt / 1e12:.2f} TFLOPS effective")


def tiny_mlp_demo():
    print("\n-- tiny MLP forward --")

    class MLP(nn.Module):
        def __init__(self, d_in: int, d_h: int, d_out: int):
            super().__init__()
            self.fc1 = nn.Linear(d_in, d_h)
            self.fc2 = nn.Linear(d_h, d_out)

        def __call__(self, x):
            return self.fc2(nn.relu(self.fc1(x)))

    model = MLP(128, 512, 10)
    x = mx.random.normal((4, 128))
    y = model(x)
    mx.eval(y)
    print(f"  output shape: {y.shape}")


def main():
    print("MLX:", mx.__version__ if hasattr(mx, "__version__") else "(version not exported)")
    lazy_eval_demo()
    autograd_demo()
    matmul_throughput()
    tiny_mlp_demo()


if __name__ == "__main__":
    main()
