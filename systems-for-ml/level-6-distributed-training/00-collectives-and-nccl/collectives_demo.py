"""
Demonstrates the four core NCCL collectives on a 2+ GPU host.

Run:
    torchrun --nproc_per_node=2 collectives_demo.py

Set NCCL_DEBUG=INFO to see ring/tree selection and topology output.
"""

import os
import torch
import torch.distributed as dist


def main() -> None:
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # all-reduce: sum of every rank's tensor lands on every rank
    t = torch.full((4,), float(rank + 1), device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    if rank == 0:
        # expected: [1+2+...+world]*ones(4)
        print(f"[all_reduce] rank0 sees: {t.tolist()}")

    # all-gather: each rank contributes a tensor; all ranks see the concatenation
    src = torch.full((2,), float(rank), device=device)
    dst = [torch.empty_like(src) for _ in range(world)]
    dist.all_gather(dst, src)
    if rank == 0:
        print(f"[all_gather] rank0 sees: {[x.tolist() for x in dst]}")

    # reduce-scatter: every rank contributes a chunked tensor; sum, split among ranks
    chunks = [torch.full((2,), float(rank * world + i), device=device) for i in range(world)]
    out = torch.empty(2, device=device)
    dist.reduce_scatter(out, chunks, op=dist.ReduceOp.SUM)
    print(f"[reduce_scatter] rank{rank} owns chunk: {out.tolist()}")

    # all-to-all: each rank sends a different slice to each other rank
    send = torch.tensor([rank * 10 + i for i in range(world)], device=device, dtype=torch.float)
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send)
    print(f"[all_to_all] rank{rank} received: {recv.tolist()}")

    dist.barrier()

    # bandwidth quick-check: time a 256 MiB all-reduce
    n = 256 * 1024 * 1024 // 4  # fp32 elements
    big = torch.randn(n, device=device)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(10):
        dist.all_reduce(big)
    end.record()
    torch.cuda.synchronize()
    secs = start.elapsed_time(end) / 1000.0 / 10
    bytes_moved = 2 * (world - 1) / world * (n * 4)
    bw = bytes_moved / secs / 1e9
    if rank == 0:
        print(f"[bw] 256 MiB all_reduce: {secs*1000:.2f} ms  algo-bw {bw:.1f} GB/s")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
