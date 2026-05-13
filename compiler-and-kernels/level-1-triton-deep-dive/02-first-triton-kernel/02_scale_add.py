"""
02 — Fused scale-add: out = alpha * x + beta * y + bias

Three inputs, two scalar coefficients, plus a bias. Unfused in PyTorch this is
multiple kernels and multiple HBM round-trips. Fused in Triton it's one kernel,
one round-trip — same load/store pattern as vector add, just more math in the middle.

This is the simplest possible demonstration of *kernel fusion*. The lesson:
the difference between fused and unfused is not in the kernel structure — it's
in how many times you cross HBM. Doing more math per byte loaded is free
(until you hit compute limits, which you won't here).

What you should observe:
  - Correctness: max diff vs torch is small (~1e-6 in fp32 due to float rounding order).
  - Throughput: same as vector add — about HBM bandwidth. The extra arithmetic
    is invisible because we're memory-bound either way.
  - If you compare to the unfused PyTorch expression (alpha*x + beta*y + bias),
    PyTorch does this in 4 separate kernels in eager mode — each one round-trips
    through HBM. The Triton kernel does it in 1. On a large tensor, the Triton
    version should be ~3-4x faster than naive eager — entirely from saving HBM traffic.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def scale_add_kernel(
    x_ptr, y_ptr, bias_ptr, out_ptr,
    alpha,           # scalar fp32 — passed by value, lives in a register on every lane
    beta,            # scalar fp32
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    b = tl.load(bias_ptr + offsets, mask=mask)

    # All the math happens in registers. The compiler will likely emit
    # an FMA (fused multiply-add) for each multiplication-then-addition pair.
    out = alpha * x + beta * y + b

    tl.store(out_ptr + offsets, out, mask=mask)


def scale_add(x, y, bias, alpha: float, beta: float):
    assert x.shape == y.shape == bias.shape and x.is_cuda
    out = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    scale_add_kernel[grid](x, y, bias, out, alpha, beta, n_elements, BLOCK_SIZE=1024)
    return out


def main():
    torch.manual_seed(0)
    device = "cuda"

    # Correctness on a few shapes
    for n in (1024, 8192, 1_000_003):
        x = torch.randn(n, device=device, dtype=torch.float32)
        y = torch.randn(n, device=device, dtype=torch.float32)
        bias = torch.randn(n, device=device, dtype=torch.float32)
        alpha, beta = 1.5, -0.7

        out_triton = scale_add(x, y, bias, alpha, beta)
        out_torch = alpha * x + beta * y + bias

        max_diff = (out_triton - out_torch).abs().max().item()
        # fp32 arithmetic, slight differences in op order can give ~1e-6 diffs
        assert max_diff < 1e-5, f"n={n}: max_diff {max_diff} too large"
        print(f"n={n:>10}  max_diff={max_diff:.2e}  OK")

    # Benchmark: fused (one kernel) vs naive eager (multiple kernels)
    n = 1 << 24
    x = torch.randn(n, device=device, dtype=torch.float32)
    y = torch.randn(n, device=device, dtype=torch.float32)
    bias = torch.randn(n, device=device, dtype=torch.float32)
    alpha, beta = 1.5, -0.7

    ms_fused = triton.testing.do_bench(lambda: scale_add(x, y, bias, alpha, beta))
    ms_eager = triton.testing.do_bench(lambda: alpha * x + beta * y + bias)

    # The fused kernel reads 3*N*4 bytes and writes N*4 bytes = 4*N*4 bytes total.
    bytes_fused = 4 * n * 4
    # The eager expression creates intermediate tensors. Conservatively:
    #   alpha*x        -> reads N, writes N
    #   beta*y         -> reads N, writes N
    #   sum1 = (...)+(...) -> reads 2N, writes N
    #   sum1 + bias    -> reads 2N, writes N
    # Total: ~10*N*4 bytes. Some of this hits the L2; the gap usually isn't quite 2.5x but is large.
    gbps_fused = bytes_fused / (ms_fused * 1e-3) / 1e9
    print()
    print(f"n = {n:,}")
    print(f"  fused triton: {ms_fused:.3f} ms   {gbps_fused:7.1f} GB/s (4N bytes)")
    print(f"  eager torch : {ms_eager:.3f} ms   (multi-pass, more bytes moved)")
    print(f"  speedup:      {ms_eager / ms_fused:.2f}x")
    print()
    print("The win comes entirely from doing the math in registers between one load and one store.")
    print("PyTorch eager makes the intermediate tensors and pays HBM for each.")
    print("This is what 'kernel fusion saves HBM round-trips' means in concrete bytes.")


if __name__ == "__main__":
    main()
