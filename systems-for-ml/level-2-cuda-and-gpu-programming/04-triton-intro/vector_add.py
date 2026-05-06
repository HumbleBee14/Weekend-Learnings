"""
The Triton hello world. Same vector add as Topic 2's CUDA C++ version, in 30 lines.

Run:
    pip install triton torch
    python vector_add.py
"""

import torch
import triton
import triton.language as tl


@triton.jit
def vector_add_kernel(
    a_ptr,                  # pointer to A
    b_ptr,                  # pointer to B
    c_ptr,                  # pointer to C (output)
    n,                      # total number of elements
    BLOCK_SIZE: tl.constexpr,  # how many elements per program (compile-time const)
):
    """One program (= one thread block) handles BLOCK_SIZE consecutive elements."""
    pid = tl.program_id(axis=0)                            # which program am I?
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # range of element indices
    mask = offsets < n                                     # boundary mask
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    c = a + b
    tl.store(c_ptr + offsets, c, mask=mask)


def vector_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Driver: allocate output, compute grid shape, launch the kernel."""
    assert a.shape == b.shape and a.is_cuda and b.is_cuda
    c = torch.empty_like(a)
    n = a.numel()

    BLOCK_SIZE = 1024
    # The grid is a tuple of integers (or a callable returning one). Here it's 1D.
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    vector_add_kernel[grid](a, b, c, n, BLOCK_SIZE=BLOCK_SIZE)
    return c


def main():
    if not torch.cuda.is_available():
        raise SystemExit("Needs CUDA.")

    n = 1 << 22  # 4M elements
    a = torch.randn(n, device="cuda")
    b = torch.randn(n, device="cuda")

    c = vector_add(a, b)
    expected = a + b
    print(f"correctness: {torch.allclose(c, expected)}")

    # Quick benchmark
    import time
    for _ in range(3):  # warmup
        vector_add(a, b)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(100):
        vector_add(a, b)
    torch.cuda.synchronize()
    ms_per = (time.perf_counter() - t0) * 10  # 1000ms / 100 iters

    bytes_moved = 3 * n * 4   # 2 reads + 1 write, fp32
    bw_gbps = bytes_moved / 1e9 / (ms_per / 1000)
    print(f"per-call: {ms_per:.3f} ms,  bandwidth: {bw_gbps:.0f} GB/s")


if __name__ == "__main__":
    main()
