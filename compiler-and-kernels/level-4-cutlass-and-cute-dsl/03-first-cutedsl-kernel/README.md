# 03 — Your first CuTe-DSL kernel

> Outer: [`../README.md`](../README.md) · Hardware: any CUDA GPU (T4 fine for this submodule).

The goal of this submodule is narrow: get the toolchain working and write three small kernels. You meet `@cute.kernel`, `@cute.jit`, the DLPack tensor crossing, and your first swizzled SMEM tile. Nothing here is fast; everything here teaches a vocabulary item.

## Install

```bash
pip install nvidia-cutlass-dsl
```

CUDA 12.4+ and Python 3.10+ required. The wheel bundles `nvcc`, `ptxas`, and the MLIR-based JIT — you don't need a separate CUTLASS clone for the DSL. (You'll want one anyway for reading examples — `git clone https://github.com/NVIDIA/cutlass`.)

Verify with `hello.py`:

```bash
python hello.py
```

Expected output: a "Hello from GPU" line. If you hit `ImportError`, the wheel didn't land — confirm CUDA visible with `nvidia-smi`, confirm Python version, retry the install. If you hit a JIT compile error, you're missing `ptxas` on the PATH; the wheel should have provided it.

## The three kernels

### hello.py — Hello from GPU

```python
import cutlass
import cutlass.cute as cute


@cute.kernel
def hello_kernel():
    tidx, _, _ = cute.arch.thread_idx()
    if tidx == 0:
        cute.printf("Hello from GPU\n")


@cute.jit
def hello():
    cutlass.cuda.initialize_cuda_context()
    hello_kernel().launch(grid=(1, 1, 1), block=(32, 1, 1))


if __name__ == "__main__":
    hello()
```

Two things to internalize from this file:

1. **`@cute.kernel` defines GPU code; `@cute.jit` defines host code that launches it.** You cannot call `hello_kernel()` directly from Python — you must go through `hello()`. The error message if you try is informative; trigger it once on purpose so you recognize it later.
2. **`.launch(grid=..., block=...)` is the launch syntax.** Same `(x, y, z)` convention as CUDA. `block=(32, 1, 1)` is one warp; you'd use 128 or 256 in real kernels.

### vector_add.py — Vector add with DLPack

```python
import torch
import cutlass
import cutlass.cute as cute


@cute.kernel
def vector_add_kernel(a: cute.Tensor, b: cute.Tensor, c: cute.Tensor, n: cutlass.Int32):
    bid, _, _ = cute.arch.block_idx()
    tid, _, _ = cute.arch.thread_idx()
    i = bid * 128 + tid          # block_size = 128
    if i < n:
        c[i] = a[i] + b[i]


@cute.jit
def vector_add(a: cute.Tensor, b: cute.Tensor, c: cute.Tensor, n: cutlass.Int32):
    grid_x = (n + 127) // 128
    vector_add_kernel(a, b, c, n).launch(grid=(grid_x, 1, 1), block=(128, 1, 1))


if __name__ == "__main__":
    N = 1 << 20
    a = torch.randn(N, device="cuda", dtype=torch.float32)
    b = torch.randn(N, device="cuda", dtype=torch.float32)
    c = torch.empty(N, device="cuda", dtype=torch.float32)

    # PyTorch tensors → cute.Tensor via DLPack (automatic).
    vector_add(
        cute.make_tensor_from_torch(a),
        cute.make_tensor_from_torch(b),
        cute.make_tensor_from_torch(c),
        N,
    )

    expected = a + b
    assert torch.allclose(c, expected), f"max diff {(c - expected).abs().max()}"
    print("ok")
```

Things to notice:

1. **PyTorch tensors become `cute.Tensor` via DLPack** with no copy. `make_tensor_from_torch` is the conversion. You write Python; you don't write CUDA memory management.
2. **The first launch JIT-compiles** (a few hundred ms). Subsequent launches with the same signature are cached. Measure with `torch.cuda.Event` and you'll see the difference.
3. **Bounds checking is your responsibility.** The `if i < n` guard prevents the last partial block from writing past the buffer.

### transpose_swizzled.py — 64×64 transpose with SMEM swizzle

Vector add doesn't use SMEM. Transpose does, and it's the smallest kernel where SMEM bank conflicts matter. We write two versions and measure the difference.

```python
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import Swizzle


TILE = 64


@cute.kernel
def transpose_kernel(
    src: cute.Tensor,
    dst: cute.Tensor,
    swizzled: cutlass.Constexpr[bool],
):
    bm, bn, _ = cute.arch.block_idx()
    tm, tn, _ = cute.arch.thread_idx()

    # Allocate SMEM tile. With swizzle: 128B XOR pattern; without: plain.
    if swizzled:
        sA = cute.make_smem_tensor(
            shape=(TILE, TILE),
            dtype=cutlass.Float32,
            swizzle=Swizzle(3, 4, 3),
        )
    else:
        sA = cute.make_smem_tensor(shape=(TILE, TILE), dtype=cutlass.Float32)

    # Coalesced GMEM load into SMEM.
    g_row = bm * TILE + tm
    g_col = bn * TILE + tn
    sA[tm, tn] = src[g_row, g_col]
    cute.arch.barrier()

    # Transposed store from SMEM.
    out_row = bn * TILE + tm
    out_col = bm * TILE + tn
    dst[out_row, out_col] = sA[tn, tm]


@cute.jit
def transpose(src: cute.Tensor, dst: cute.Tensor, swizzled: cutlass.Constexpr[bool]):
    M, N = src.shape
    transpose_kernel(src, dst, swizzled).launch(
        grid=(M // TILE, N // TILE, 1),
        block=(TILE, TILE, 1),
    )


def benchmark(swizzled: bool, n_iter: int = 100):
    M = N = 4096
    src = torch.randn(M, N, device="cuda", dtype=torch.float32)
    dst = torch.empty(N, M, device="cuda", dtype=torch.float32)

    # Warmup.
    for _ in range(5):
        transpose(cute.make_tensor_from_torch(src), cute.make_tensor_from_torch(dst), swizzled)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iter):
        transpose(cute.make_tensor_from_torch(src), cute.make_tensor_from_torch(dst), swizzled)
    end.record()
    torch.cuda.synchronize()

    ms = start.elapsed_time(end) / n_iter
    bytes_xferred = 2 * M * N * 4
    gbps = bytes_xferred / (ms * 1e6)
    return ms, gbps


if __name__ == "__main__":
    ms_no, gbps_no = benchmark(swizzled=False)
    ms_yes, gbps_yes = benchmark(swizzled=True)
    print(f"unswizzled: {ms_no:.3f} ms  {gbps_no:6.1f} GB/s")
    print(f"swizzled:   {ms_yes:.3f} ms  {gbps_yes:6.1f} GB/s")
    print(f"speedup: {ms_no / ms_yes:.2f}x")
```

Expected on H100: ~2–4× speedup with swizzle. On T4: 1.5–2× (bank conflicts are still real, but the memory subsystem is slower so the absolute gap is smaller).

Things to notice:

1. **`cutlass.Constexpr[bool]`** marks `swizzled` as compile-time. The IR specializes — two distinct compiled kernels, no runtime branch.
2. **`cute.make_smem_tensor(..., swizzle=Swizzle(3,4,3))`** — this is where submodule 02's algebra meets the kernel. The XOR pattern from `cute_algebra.py` is the same XOR the hardware applies.
3. **`cute.arch.barrier()`** is `__syncthreads()`. Without it the transposed read races the cooperative write.

## The `@kernel`/`@jit` rules, restated

| Rule | Reason |
|---|---|
| Python → `@cute.jit`: ok | Host code, runs on CPU |
| Python → `@cute.kernel`: error | Can only be launched from a `@jit` function |
| `@cute.jit` → `@cute.kernel.launch(...)`: ok | The driver issues the launch |
| `@cute.kernel` → `@cute.kernel.launch(...)`: error | No nested launches; use dynamic parallelism via a different API |
| `@cute.kernel` → `@cute.jit`: error | No host calls from device |

If you forget the boundary you'll get a clear error message at JIT time. Trigger each error once on purpose; the messages are how you'll debug your kernel structure later.

## Profiling

The CuTe-DSL JIT emits PTX you can inspect:

```python
print(transpose_kernel.get_ptx(...))   # exact API may evolve; check the docs
```

For real profiling, `nsys profile` and `ncu` work on CuTe-DSL kernels exactly as they do on Triton or hand-written CUDA. The kernel name in the trace will be a mangled `cute_kernel_*`.

## What you should be able to do next

- Write a CuTe-DSL kernel from scratch that loads a tile, does pointwise math in registers, writes the tile back.
- Recognize the `@kernel`/`@jit` error messages.
- Apply a swizzle to a SMEM allocation and measure the bank-conflict win.
- Read [`hopper/dense_gemm.py`](https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/hopper/dense_gemm.py) and identify the kernel boundary, the SMEM allocation, the TMA setup, the MMA call. You don't need to understand it all yet — submodule 04 is where you build it.

## References

- [CUTLASS examples/python/CuTeDSL/](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL) — official examples.
- [CuTe DSL Basics — Chris Choy](https://chrischoy.org/posts/cutedsl-basics/) — a clean hands-on walkthrough.
- [An applied introduction to CuTeDSL — Simon Veitner](https://veitner.bearblog.dev/an-applied-introduction-to-cutedsl/).
- [CuTe DSL — NVIDIA CUTLASS Documentation](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl.html).
