"""
Inject a per-step delay on one rank and measure how it inflates the
synchronous step time across all ranks. Produces p50/p95/p99 step times
for G11 of Project 3.

Run:
    torchrun --standalone --nproc_per_node=2 straggler_inject.py --slow_ms 50
    torchrun --standalone --nproc_per_node=2 straggler_inject.py --slow_ms 0
"""

import argparse
import os
import time
import statistics
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--slow_ms", type=int, default=0, help="extra ms on rank 0 each step")
    p.add_argument("--slow_rank", type=int, default=0)
    p.add_argument("--steps", type=int, default=100)
    args = p.parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(1024, 4096), nn.GELU(), nn.Linear(4096, 1024)
    ).to(device)
    model = DDP(model, device_ids=[local_rank])
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)

    times = []
    for step in range(args.steps):
        x = torch.randn(32, 1024, device=device)
        torch.cuda.synchronize()
        t0 = time.time()
        y = model(x)
        loss = y.pow(2).mean()
        optim.zero_grad(set_to_none=True)
        loss.backward()
        if rank == args.slow_rank and args.slow_ms:
            time.sleep(args.slow_ms / 1000.0)
        optim.step()
        torch.cuda.synchronize()
        dist.barrier()
        times.append((time.time() - t0) * 1000.0)

    # All ranks see the same wallclock since we barrier'd; compute on rank 0
    if rank == 0:
        times = times[10:]  # drop warmup
        p50 = statistics.median(times)
        p95 = statistics.quantiles(times, n=20)[18]
        p99 = statistics.quantiles(times, n=100)[98]
        print(f"world={world}  slow_ms={args.slow_ms} on rank{args.slow_rank}")
        print(f"  p50 step: {p50:.2f} ms")
        print(f"  p95 step: {p95:.2f} ms")
        print(f"  p99 step: {p99:.2f} ms")
        print(f"  mean    : {statistics.mean(times):.2f} ms")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
