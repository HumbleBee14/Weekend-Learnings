"""
Manual 2-stage pipeline parallelism with a 1F1B-style microbatch schedule.

Hand-rolled (no torch.distributed.pipelining) so the schedule is visible.

Run:
    torchrun --standalone --nproc_per_node=2 pp_demo.py
"""

import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist


def make_block(dim: int = 512) -> nn.Module:
    return nn.Sequential(
        nn.Linear(dim, dim * 4),
        nn.GELU(),
        nn.Linear(dim * 4, dim),
        nn.LayerNorm(dim),
    )


def main() -> None:
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    assert world == 2, "this demo is hand-rolled for 2 stages"
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # Each stage owns 4 blocks
    torch.manual_seed(0)
    stage = nn.ModuleList([make_block() for _ in range(4)]).to(device)

    n_microbatches = 8
    seq, dim = 64, 512
    micro_size = 4
    optim = torch.optim.AdamW(stage.parameters(), lr=1e-4)

    # GPipe-naive: all forwards then all backwards. Easy to read; big bubble.
    def step_naive() -> float:
        torch.cuda.synchronize()
        t0 = time.time()
        activations = []  # list of (input, output) per microbatch on this stage

        for mb in range(n_microbatches):
            if rank == 0:
                x = torch.randn(micro_size, seq, dim, device=device, requires_grad=True)
            else:
                x = torch.empty(micro_size, seq, dim, device=device, requires_grad=True)
                dist.recv(x, src=0)

            h = x
            for blk in stage:
                h = blk(h)

            if rank == 0:
                dist.send(h, dst=1)
            activations.append((x, h))

        # backward in reverse microbatch order
        for mb in reversed(range(n_microbatches)):
            x, h = activations[mb]
            if rank == 1:
                # synthetic loss: L2 of output
                loss = h.pow(2).mean()
                loss.backward()
                # send dX back to stage 0
                dist.send(x.grad, dst=0)
            else:
                grad_out = torch.empty_like(h)
                dist.recv(grad_out, src=1)
                h.backward(grad_out)

        optim.step()
        optim.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        return time.time() - t0

    # 1F1B: alternate F and B once the pipeline is full
    def step_1f1b() -> float:
        torch.cuda.synchronize()
        t0 = time.time()
        forward_q = []  # microbatches whose forward has been done but backward has not

        warmup = world - rank  # stage 0 does world fwds first; stage 1 does world-1
        n_fwd_done = 0
        n_bwd_done = 0

        # warmup forwards
        for _ in range(warmup):
            forward_q.append(do_forward(rank, stage, device, micro_size, seq, dim))
            n_fwd_done += 1

        # steady state
        while n_bwd_done < n_microbatches:
            if n_fwd_done < n_microbatches:
                forward_q.append(do_forward(rank, stage, device, micro_size, seq, dim))
                n_fwd_done += 1
            x, h = forward_q.pop(0)
            do_backward(rank, x, h, device)
            n_bwd_done += 1

        optim.step()
        optim.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        return time.time() - t0

    # warmup
    for _ in range(2):
        step_naive()

    t_naive = step_naive()
    t_1f1b = step_1f1b()

    if rank == 0:
        print(f"GPipe naive : {t_naive*1000:.1f} ms")
        print(f"1F1B        : {t_1f1b*1000:.1f} ms")
        # rough bubble estimate: bubble ≈ (S-1)/(M+S-1) for naive
        S, M = 2, n_microbatches
        bubble = (S - 1) / (M + S - 1)
        print(f"theoretical bubble (S=2, M={M}): {bubble*100:.1f}%")

    dist.destroy_process_group()


def do_forward(rank: int, stage: nn.ModuleList, device, micro_size, seq, dim):
    if rank == 0:
        x = torch.randn(micro_size, seq, dim, device=device, requires_grad=True)
    else:
        x = torch.empty(micro_size, seq, dim, device=device, requires_grad=True)
        dist.recv(x, src=0)
    h = x
    for blk in stage:
        h = blk(h)
    if rank == 0:
        dist.send(h, dst=1)
    return (x, h)


def do_backward(rank: int, x, h, device):
    if rank == 1:
        loss = h.pow(2).mean()
        loss.backward()
        dist.send(x.grad, dst=0)
    else:
        grad_out = torch.empty_like(h)
        dist.recv(grad_out, src=1)
        h.backward(grad_out)


if __name__ == "__main__":
    main()
