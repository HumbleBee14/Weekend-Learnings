"""
A small live demo: launch a kernel that prints which warp/block/thread it is.

Not really CUDA — Triton, because we don't want to set up a full CUDA toolchain just
to make this point. Same execution model.

Run on Colab T4 (free) or any GPU box:
    pip install triton torch
    python diagram_warps.py
"""

import torch
import triton
import triton.language as tl


@triton.jit
def show_topology_kernel(out_ptr, BLOCK_SIZE: tl.constexpr):
    """
    Each program (= block) writes one row showing:
      - its block id (program_id(0))
      - the thread offsets it owns (BLOCK_SIZE consecutive elements)

    There are no warps in Triton's surface API — the compiler manages them. But under
    the hood, BLOCK_SIZE threads get split into BLOCK_SIZE/32 warps. With BLOCK_SIZE=128,
    each block has 4 warps.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Write each thread's "global thread index" into the output array
    tl.store(out_ptr + offsets, offsets)


def main():
    BLOCK_SIZE = 128  # threads per block; this becomes 4 warps (128/32)
    NUM_BLOCKS = 4
    N = BLOCK_SIZE * NUM_BLOCKS

    out = torch.zeros(N, dtype=torch.int32, device="cuda")
    show_topology_kernel[(NUM_BLOCKS,)](out, BLOCK_SIZE=BLOCK_SIZE)
    out_h = out.cpu().tolist()

    print(f"Launched grid={NUM_BLOCKS} block={BLOCK_SIZE} → {NUM_BLOCKS * BLOCK_SIZE} threads total\n")
    print(f"That's {NUM_BLOCKS * BLOCK_SIZE // 32} warps.\n")

    for b in range(NUM_BLOCKS):
        block_threads = out_h[b * BLOCK_SIZE : (b + 1) * BLOCK_SIZE]
        warps = [block_threads[i * 32 : (i + 1) * 32] for i in range(BLOCK_SIZE // 32)]
        print(f"Block {b}:")
        for w_idx, warp in enumerate(warps):
            print(f"  Warp {w_idx}: threads {warp[0]}..{warp[-1]}  (32 threads in lockstep)")
        print()


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("Needs CUDA.")
    main()
