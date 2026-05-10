"""
Sweep all-reduce bandwidth across message sizes and (optionally) across
fast vs slow transports. Produces a CSV you can plot for G10.

Run:
    torchrun --nproc_per_node=2 bw_matrix.py
    # then re-run with NCCL_P2P_DISABLE=1 for the slow-path comparison
"""
import os
import csv
import time
import torch
import torch.distributed as dist


SIZES_MIB = [0.001, 0.01, 0.1, 1, 4, 16, 64, 256, 1024]


def main() -> None:
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    tag = "slow" if os.environ.get("NCCL_P2P_DISABLE") == "1" else "fast"

    rows = []
    for mib in SIZES_MIB:
        n = max(1, int(mib * 1024 * 1024 // 4))
        t = torch.randn(n, device=device)
        # warmup
        for _ in range(3):
            dist.all_reduce(t)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        iters = 20 if mib < 64 else 5
        for _ in range(iters):
            dist.all_reduce(t)
        end.record()
        torch.cuda.synchronize()
        secs = start.elapsed_time(end) / 1000.0 / iters
        bus_bytes = 2 * (world - 1) / world * (n * 4)
        bw = bus_bytes / secs / 1e9
        if rank == 0:
            print(f"{tag:>4s}  {mib:>8.3f} MiB   {secs*1e3:>8.3f} ms   {bw:>7.2f} GB/s")
            rows.append([tag, mib, secs * 1e3, bw])

    if rank == 0:
        path = f"bw_{tag}.csv"
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["transport", "size_mib", "ms_per_call", "algo_bw_gbs"])
            w.writerows(rows)
        print(f"wrote {path}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
