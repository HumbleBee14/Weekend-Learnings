"""
Elastic-launch DDP loop with periodic checkpointing. Designed to be killed
mid-step from another shell so you can watch the survivors recover.

Run with elastic launcher:
    torchrun \
        --nnodes=1 --nproc_per_node=2 \
        --rdzv_backend=c10d --rdzv_endpoint=localhost:29500 \
        --rdzv_id=fail_test --max-restarts=3 \
        elastic_train.py

Then in another shell:
    pkill -9 -f elastic_train.py    # kill all
    # or:  kill -9 <pid_of_one_rank>

The agent will restart the dead worker (or, with NCCL_SHRINK_ABORT, the
survivors continue at smaller world).
"""

import os
import time
import signal
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.nn.parallel import DistributedDataParallel as DDP


CKPT_DIR = "./elastic_ckpt"


def maybe_load(model, optim) -> int:
    if not os.path.exists(CKPT_DIR):
        return 0
    state = {"model": model.state_dict(), "optim": optim.state_dict(),
             "step": torch.tensor(0)}
    dcp.load(state, checkpoint_id=CKPT_DIR)
    step = int(state["step"].item())
    if dist.get_rank() == 0:
        print(f"resumed from checkpoint at step {step}")
    return step


def save(model, optim, step) -> None:
    state = {"model": model.state_dict(), "optim": optim.state_dict(),
             "step": torch.tensor(step)}
    dcp.save(state, checkpoint_id=CKPT_DIR)


def main() -> None:
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print(f"world={world} (after rendezvous; may differ if rank dropped)")

    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(1024, 4096), nn.GELU(), nn.Linear(4096, 1024)
    ).to(device)
    model = DDP(model, device_ids=[local_rank])
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)

    start_step = maybe_load(model, optim)

    for step in range(start_step, start_step + 200):
        x = torch.randn(32, 1024, device=device)
        y = model(x)
        loss = y.pow(2).mean()
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        if rank == 0 and step % 5 == 0:
            print(f"[pid {os.getpid()}] step {step}  loss {loss.item():.4f}")
        if step % 20 == 0 and step > 0:
            save(model, optim, step)
            if rank == 0:
                print(f"checkpoint saved at step {step}")
        time.sleep(0.05)  # slow loop so you have time to kill -9

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
