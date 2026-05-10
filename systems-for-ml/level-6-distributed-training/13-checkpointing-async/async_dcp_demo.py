"""
Compare sync vs async DCP checkpointing. Reports:
  - training-pause time (the metric that matters)
  - total save time (sync) or future-completion time (async)

Run:
    torchrun --standalone --nproc_per_node=2 async_dcp_demo.py
"""

import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.distributed.checkpoint as dcp


def main() -> None:
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # Big-ish model so the save time is visible
    torch.manual_seed(0)
    model = nn.Sequential(*[nn.Linear(4096, 4096) for _ in range(8)]).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # warm the optimizer state (needs at least one step for Adam moments)
    x = torch.randn(8, 4096, device=device)
    model(x).pow(2).mean().backward()
    optim.step()
    torch.cuda.synchronize()

    state = {"model": model.state_dict(), "optim": optim.state_dict()}

    # ---- sync save ----
    if os.path.exists("./ckpt_sync"):
        os.system("rm -rf ./ckpt_sync")
    torch.cuda.synchronize()
    dist.barrier()
    t0 = time.time()
    dcp.save(state, checkpoint_id="./ckpt_sync")
    torch.cuda.synchronize()
    dist.barrier()
    sync_time = time.time() - t0
    if rank == 0:
        print(f"sync  save: {sync_time*1000:.1f} ms (training paused entire time)")

    # ---- async save ----
    if os.path.exists("./ckpt_async"):
        os.system("rm -rf ./ckpt_async")
    torch.cuda.synchronize()
    dist.barrier()
    t0 = time.time()
    fut = dcp.async_save(state, checkpoint_id="./ckpt_async")
    torch.cuda.synchronize()
    pause_time = time.time() - t0   # training-pause time
    # ... training would resume here ...
    fut.result()  # wait for the background write
    dist.barrier()
    total_async = time.time() - t0
    if rank == 0:
        print(f"async save: pause {pause_time*1000:.1f} ms; total {total_async*1000:.1f} ms")
        print(f"speedup on training-pause: {sync_time/pause_time:.1f}x")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
