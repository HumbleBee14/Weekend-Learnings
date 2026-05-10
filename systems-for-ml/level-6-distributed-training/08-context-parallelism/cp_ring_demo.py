"""
Tiny CP=2 ring-attention skeleton. Doesn't compute real attention scores —
shows the K/V rotation pattern around the ring.

Run:
    torchrun --standalone --nproc_per_node=2 cp_ring_demo.py
"""

import os
import torch
import torch.distributed as dist


def main() -> None:
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # Each device holds 1024 tokens of K/V; full sequence is 1024 * world
    seq_local, dim = 1024, 64
    Q = torch.randn(seq_local, dim, device=device)
    K = torch.randn(seq_local, dim, device=device) + rank
    V = torch.randn(seq_local, dim, device=device) + rank

    # Place K, V into a rolling buffer that we will rotate around the ring
    K_buf, V_buf = K.clone(), V.clone()

    # Online softmax accumulator (FlashAttention style)
    out = torch.zeros_like(Q)
    max_score = torch.full((seq_local,), float("-inf"), device=device)
    sum_exp = torch.zeros(seq_local, device=device)

    next_rank = (rank + 1) % world
    prev_rank = (rank - 1 + world) % world

    for step in range(world):
        # Q attends to (K_buf, V_buf), which currently belongs to rank=(rank-step)%world
        scores = Q @ K_buf.T / (dim ** 0.5)  # (seq_local, seq_local)
        # numerically stable update
        block_max = scores.max(dim=-1).values
        new_max = torch.maximum(max_score, block_max)
        out *= torch.exp(max_score - new_max).unsqueeze(-1)
        sum_exp *= torch.exp(max_score - new_max)
        weights = torch.exp(scores - new_max.unsqueeze(-1))
        out += weights @ V_buf
        sum_exp += weights.sum(dim=-1)
        max_score = new_max

        if step == world - 1:
            break

        # rotate: send our K_buf, V_buf to next_rank; receive from prev_rank
        K_recv = torch.empty_like(K_buf)
        V_recv = torch.empty_like(V_buf)
        # Use batch_isend_irecv to avoid deadlock
        ops = [
            dist.P2POp(dist.isend, K_buf, next_rank),
            dist.P2POp(dist.isend, V_buf, next_rank),
            dist.P2POp(dist.irecv, K_recv, prev_rank),
            dist.P2POp(dist.irecv, V_recv, prev_rank),
        ]
        for h in dist.batch_isend_irecv(ops):
            h.wait()
        K_buf, V_buf = K_recv, V_recv

    out /= sum_exp.unsqueeze(-1)

    if rank == 0:
        print(f"rank{rank}: each device computed attention on its {seq_local} Q tokens against all {seq_local*world} K/V tokens via {world}-step ring")
        print(f"rank{rank}: output norm: {out.norm().item():.3f}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
