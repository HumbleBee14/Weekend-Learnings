"""A tiny custom Metal kernel via MLX's mx.fast.metal_kernel.

Computes y = x * 2 + 1 elementwise, then compares to the MLX-built equivalent.

This is the lightest-touch path to writing real Metal: write MSL in a Python
string, hand it to MLX, get an op back. No Xcode, no Swift, no project file.
"""

from __future__ import annotations

import time

import mlx.core as mx


KERNEL_SRC = """
[[kernel]] void mul2_add1(
    device const float* in   [[buffer(0)]],
    device       float* out  [[buffer(1)]],
    uint                 tid [[thread_position_in_grid]])
{
    out[tid] = in[tid] * 2.0f + 1.0f;
}
"""


def build_kernel():
    return mx.fast.metal_kernel(
        name="mul2_add1",
        input_names=["in"],
        output_names=["out"],
        source=KERNEL_SRC,
    )


def main():
    kernel = build_kernel()
    n = 1 << 22  # 4M floats

    x = mx.arange(n, dtype=mx.float32)
    mx.eval(x)

    # warm
    (y_custom,) = kernel(
        inputs=[x],
        output_shapes=[(n,)],
        output_dtypes=[mx.float32],
        grid=(n, 1, 1),
        threadgroup=(256, 1, 1),
    )
    mx.eval(y_custom)

    repeats = 200
    t0 = time.perf_counter()
    for _ in range(repeats):
        (y_custom,) = kernel(
            inputs=[x],
            output_shapes=[(n,)],
            output_dtypes=[mx.float32],
            grid=(n, 1, 1),
            threadgroup=(256, 1, 1),
        )
    mx.eval(y_custom)
    dt_custom = time.perf_counter() - t0

    # equivalent in MLX ops (lazy, fused by the compiler)
    y_mlx = x * 2.0 + 1.0
    mx.eval(y_mlx)
    t0 = time.perf_counter()
    for _ in range(repeats):
        y_mlx = x * 2.0 + 1.0
    mx.eval(y_mlx)
    dt_mlx = time.perf_counter() - t0

    print(f"custom Metal kernel : {dt_custom*1000:7.2f} ms / {repeats} runs")
    print(f"mx ops (compiled)   : {dt_mlx*1000:7.2f} ms / {repeats} runs")
    print(f"correctness         : max diff = {float(mx.abs(y_custom - y_mlx).max()):.3e}")


if __name__ == "__main__":
    main()
