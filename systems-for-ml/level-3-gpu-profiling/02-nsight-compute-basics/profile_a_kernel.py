"""
Two kernels, very different profiles. Use ncu to see why.

  fast_kernel — coalesced row-wise reduction, well-utilized
  slow_kernel — strided access, uncoalesced, terrible bandwidth

Run baseline:
    python profile_a_kernel.py

Profile each with ncu (basic):
    ncu --set basic -k regex:fast_kernel  -c 1 python profile_a_kernel.py
    ncu --set basic -k regex:slow_kernel  -c 1 python profile_a_kernel.py

Profile with full metrics (slow but exhaustive):
    ncu --set full  -k regex:slow_kernel  -c 1 -o slow.ncu-rep python profile_a_kernel.py

Open the .ncu-rep files in the Nsight Compute GUI. Compare the two:
  - Speed of Light page: which is compute-bound, which is memory-bound, which is just bad?
  - Memory Workload Analysis: HBM bandwidth utilization
  - Warp State Statistics: dominant stall reasons (slow_kernel will show "Long Scoreboard")
"""

import torch
import triton
import triton.language as tl


@triton.jit
def fast_kernel(
    x_ptr, out_ptr,
    n_rows, n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Sum each row of x and store the result. Coalesced access pattern:
    threads in a warp read consecutive memory locations.
    """
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    # All threads in the warp read consecutive cols of the same row → coalesced
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0)
    s = tl.sum(x, axis=0)
    tl.store(out_ptr + row, s)


@triton.jit
def slow_kernel(
    x_ptr, out_ptr,
    n_rows, n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Same operation, but with a deliberately bad access pattern: each thread
    in the warp reads a column, not a row. So threads in the same warp end
    up reading addresses that are n_cols apart — completely uncoalesced.
    """
    col = tl.program_id(0)
    rows = tl.arange(0, BLOCK_SIZE)
    mask = rows < n_rows

    # Threads read different rows of the same col → strided, uncoalesced reads
    x = tl.load(x_ptr + rows * n_cols + col, mask=mask, other=0.0)
    s = tl.sum(x, axis=0)
    tl.store(out_ptr + col, s)


def main():
    if not torch.cuda.is_available():
        raise SystemExit("Needs CUDA.")

    n_rows, n_cols = 4096, 4096
    x = torch.randn((n_rows, n_cols), device="cuda", dtype=torch.float32)

    # Output for fast: one value per row. Output for slow: one value per column.
    out_fast = torch.empty(n_rows, device="cuda", dtype=torch.float32)
    out_slow = torch.empty(n_cols, device="cuda", dtype=torch.float32)

    BLOCK = triton.next_power_of_2(max(n_cols, n_rows))

    # Warmup
    for _ in range(3):
        fast_kernel[(n_rows,)](x, out_fast, n_rows, n_cols, BLOCK_SIZE=BLOCK)
        slow_kernel[(n_cols,)](x, out_slow, n_rows, n_cols, BLOCK_SIZE=BLOCK)
    torch.cuda.synchronize()

    # Run each — when ncu is attached and you specified -k regex:, only the
    # matching kernel will be deeply profiled.
    print("Running fast_kernel...")
    fast_kernel[(n_rows,)](x, out_fast, n_rows, n_cols, BLOCK_SIZE=BLOCK)
    torch.cuda.synchronize()

    print("Running slow_kernel...")
    slow_kernel[(n_cols,)](x, out_slow, n_rows, n_cols, BLOCK_SIZE=BLOCK)
    torch.cuda.synchronize()

    print("Done. Profile each with: ncu --set basic -k regex:<name> -c 1 ./this_script")


if __name__ == "__main__":
    main()
