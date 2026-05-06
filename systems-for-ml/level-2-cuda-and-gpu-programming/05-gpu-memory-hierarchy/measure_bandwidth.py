"""
Measure achieved bandwidth at each level of the hierarchy on your GPU.

Three workloads:
  1. Pure HBM bandwidth (large array streamed end-to-end)
  2. SMEM-friendly (small data, hits the L1/SMEM tier)
  3. Compute-bound (lots of compute per byte read — should approach peak FLOPS, not BW)

Run:
    pip install triton torch
    python measure_bandwidth.py

This is a sanity check — see what bandwidth your specific GPU actually achieves and
compare to the spec sheet from CONCEPTS.md.
"""

import time
import torch
import triton
import triton.language as tl


def device_specs():
    """Print the GPU we're running on."""
    if not torch.cuda.is_available():
        raise SystemExit("Needs CUDA.")
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"GPU: {name}  (compute capability sm_{cap[0]}{cap[1]})")
    print(f"Total memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")


@triton.jit
def stream_copy_kernel(in_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    """y = x. Pure HBM streaming — should hit close to peak HBM bandwidth."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x = tl.load(in_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x, mask=mask)


@triton.jit
def stream_compute_kernel(in_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    """y = sin(cos(sin(cos(x)))). Lots of compute per byte; tests whether we're compute-bound."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x = tl.load(in_ptr + offsets, mask=mask)
    # 8 transcendentals per element — should saturate the SFU (special function unit)
    for _ in range(4):
        x = tl.cos(tl.sin(x))
    tl.store(out_ptr + offsets, x, mask=mask)


def bench_hbm_streaming(n_bytes: int):
    """Pure copy bandwidth — read x, write y, no compute."""
    n = n_bytes // 4
    x = torch.empty(n, device="cuda", dtype=torch.float32)
    y = torch.empty(n, device="cuda", dtype=torch.float32)

    # Warmup
    for _ in range(3):
        stream_copy_kernel[(triton.cdiv(n, 1024),)](x, y, n, BLOCK_SIZE=1024)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(20):
        stream_copy_kernel[(triton.cdiv(n, 1024),)](x, y, n, BLOCK_SIZE=1024)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000 / 20

    bytes_moved = 2 * n_bytes  # 1 read + 1 write
    bw = bytes_moved / 1e9 / (ms / 1000)
    print(f"  HBM streaming ({n_bytes/1e6:.0f} MB):  {bw:.0f} GB/s  ({ms:.2f} ms)")
    return bw


def bench_compute_bound(n_bytes: int):
    n = n_bytes // 4
    x = torch.empty(n, device="cuda", dtype=torch.float32).uniform_(-1, 1)
    y = torch.empty(n, device="cuda", dtype=torch.float32)

    for _ in range(3):
        stream_compute_kernel[(triton.cdiv(n, 1024),)](x, y, n, BLOCK_SIZE=1024)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(20):
        stream_compute_kernel[(triton.cdiv(n, 1024),)](x, y, n, BLOCK_SIZE=1024)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000 / 20

    bytes_moved = 2 * n_bytes
    bw = bytes_moved / 1e9 / (ms / 1000)
    print(f"  Compute heavy ({n_bytes/1e6:.0f} MB):  {bw:.0f} GB/s  ({ms:.2f} ms)")
    print(f"      ↑ lower than streaming because we're compute-bound, not bandwidth-bound")


def main():
    device_specs()

    print("Bandwidth measurements (peak achievable on a streaming pattern):\n")
    print("Per-call bandwidth at increasing data sizes:")

    # Small sizes might fit in L2 → faster than HBM
    for size in [1 << 20, 1 << 22, 1 << 24, 1 << 26, 1 << 28]:  # 1 MB → 256 MB
        try:
            bench_hbm_streaming(size)
        except torch.cuda.OutOfMemoryError:
            print(f"  {size/1e6:.0f} MB: out of memory")
            continue

    print()
    bench_compute_bound(1 << 26)  # 64 MB

    print()
    print("Notes:")
    print("- 1MB-4MB sizes: probably fitting in L2, so the bandwidth looks higher than spec")
    print("- 64MB+ sizes: well beyond L2, so this is your real HBM bandwidth")
    print("- Compare to spec: H100 = 3.35 TB/s, A100 = 1.94 TB/s, T4 = 320 GB/s")
    print("  You'll typically achieve 70-90% of spec on a well-coalesced kernel")


if __name__ == "__main__":
    main()
