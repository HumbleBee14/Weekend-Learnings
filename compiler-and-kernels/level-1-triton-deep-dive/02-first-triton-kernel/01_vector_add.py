"""
01 — Vector add in Triton.

The simplest possible kernel. Read it line by line.

What you should observe when you run this:
  - Correctness: max diff vs torch is 0.0 (exact)
  - Throughput: roughly bound by HBM bandwidth. On T4 ~250 GB/s observed;
    on H100 ~2.5 TB/s. Vector add reads 2 * N floats and writes N floats,
    so it's a perfect memory-bandwidth probe.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def vector_add_kernel(
    x_ptr,           # *const float* — pointer to input tensor x
    y_ptr,           # *const float* — pointer to input tensor y
    out_ptr,         # *float*       — pointer to output tensor
    n_elements,      # int           — total number of elements
    BLOCK_SIZE: tl.constexpr,  # int — known at compile time; size of the tile each program handles
):
    # Which program am I? (We launched a 1D grid of `ceil(n_elements / BLOCK_SIZE)` programs.)
    pid = tl.program_id(axis=0)

    # Where does my tile start, and what are the indices I'm responsible for?
    # `tl.arange(0, BLOCK_SIZE)` produces a vector [0, 1, ..., BLOCK_SIZE-1].
    # We add the program's tile offset to get the global indices.
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Mask off out-of-bounds indices (the last program's tile may run past the tensor).
    mask = offsets < n_elements

    # Load. `mask=mask` suppresses the load on out-of-bounds lanes; `other=0.0`
    # is what those lanes see instead. We don't use `other` here because we mask
    # the store too, but it would matter if we did arithmetic on the loaded vector.
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # The actual work. Elementwise add. This compiles to one FFMA per lane.
    out = x + y

    # Store. Mask suppresses out-of-bounds writes.
    tl.store(out_ptr + offsets, out, mask=mask)


def vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Python wrapper that launches the kernel."""
    assert x.is_cuda and y.is_cuda and x.shape == y.shape
    out = torch.empty_like(x)
    n_elements = x.numel()

    # Grid size: enough programs to cover all elements. `triton.cdiv(a, b) == (a + b - 1) // b`.
    # The `meta` lambda receives the kernel's constexprs; we use it to size the grid based on BLOCK_SIZE.
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    vector_add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=1024)
    return out


# -------- Correctness + benchmark --------

def main():
    torch.manual_seed(0)
    device = "cuda"

    # Test on a few shapes, including a non-power-of-2 to verify masking.
    for n in (1024, 8192, 1_000_003, 1 << 22):
        x = torch.randn(n, device=device, dtype=torch.float32)
        y = torch.randn(n, device=device, dtype=torch.float32)

        out_triton = vector_add(x, y)
        out_torch = x + y

        max_diff = (out_triton - out_torch).abs().max().item()
        print(f"n={n:>10}  max_diff={max_diff:.2e}  {'OK' if max_diff == 0 else 'WRONG'}")
        assert max_diff == 0, "Vector add should be bit-exact"

    # Benchmark on a large shape — this measures memory bandwidth, not arithmetic.
    n = 1 << 24  # 16M floats per tensor
    x = torch.randn(n, device=device, dtype=torch.float32)
    y = torch.randn(n, device=device, dtype=torch.float32)

    ms_triton = triton.testing.do_bench(lambda: vector_add(x, y))
    ms_torch = triton.testing.do_bench(lambda: x + y)

    # Bytes moved: 2 loads + 1 store, each is N * 4 bytes for fp32.
    bytes_moved = 3 * n * 4
    gbps_triton = bytes_moved / (ms_triton * 1e-3) / 1e9
    gbps_torch = bytes_moved / (ms_torch * 1e-3) / 1e9

    print()
    print(f"n = {n:,}")
    print(f"  triton: {ms_triton:.3f} ms   {gbps_triton:7.1f} GB/s")
    print(f"  torch : {ms_torch:.3f} ms   {gbps_torch:7.1f} GB/s")
    print()
    print("If you're on T4 you should see ~200–280 GB/s for both.")
    print("If you're on A100 you should see ~1300–1700 GB/s for both.")
    print("If you're on H100 you should see ~2000–2800 GB/s for both.")
    print("torch's eager kernel and ours should be within 5% — there's no clever")
    print("trick available for elementwise add; both are memory-bandwidth-bound.")


if __name__ == "__main__":
    main()
