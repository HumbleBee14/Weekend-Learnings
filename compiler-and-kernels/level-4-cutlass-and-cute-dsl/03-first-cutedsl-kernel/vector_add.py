"""
vector_add.py — elementwise add in CuTe-DSL. Demonstrates:
  - DLPack PyTorch → cute.Tensor crossing (no copy)
  - 1D launch grid with bounds checking
  - JIT compile happens on first launch; subsequent launches reuse cache

Run:   python vector_add.py
Expect: "ok" if c == a + b.
"""

import torch
import cutlass
import cutlass.cute as cute


BLOCK = 128


@cute.kernel
def vector_add_kernel(
    a: cute.Tensor,
    b: cute.Tensor,
    c: cute.Tensor,
    n: cutlass.Int32,
):
    bid, _, _ = cute.arch.block_idx()
    tid, _, _ = cute.arch.thread_idx()
    i = bid * BLOCK + tid
    if i < n:
        c[i] = a[i] + b[i]


@cute.jit
def vector_add(
    a: cute.Tensor,
    b: cute.Tensor,
    c: cute.Tensor,
    n: cutlass.Int32,
):
    grid_x = (n + BLOCK - 1) // BLOCK
    vector_add_kernel(a, b, c, n).launch(grid=(grid_x, 1, 1), block=(BLOCK, 1, 1))


if __name__ == "__main__":
    N = 1 << 20
    a = torch.randn(N, device="cuda", dtype=torch.float32)
    b = torch.randn(N, device="cuda", dtype=torch.float32)
    c = torch.empty(N, device="cuda", dtype=torch.float32)

    vector_add(
        cute.make_tensor_from_torch(a),
        cute.make_tensor_from_torch(b),
        cute.make_tensor_from_torch(c),
        N,
    )

    expected = a + b
    max_diff = (c - expected).abs().max().item()
    assert max_diff < 1e-5, f"max diff {max_diff}"
    print("ok")
