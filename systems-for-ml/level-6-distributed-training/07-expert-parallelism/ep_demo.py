"""
Toy EP=2 MoE forward pass. Demonstrates the two all-to-alls per layer:
1) dispatch tokens to expert-owning device
2) combine expert outputs back to origin

Runs on 2 GPUs. 4 experts, 2 per device. K=1 (top-1 routing) for clarity.

Run:
    torchrun --standalone --nproc_per_node=2 ep_demo.py
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist


class Expert(nn.Module):
    def __init__(self, dim: int = 64) -> None:
        super().__init__()
        self.w = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w(x)


def main() -> None:
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    assert world == 2
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    n_experts = 4
    experts_per_device = n_experts // world
    dim = 64
    n_tokens_per_device = 16

    torch.manual_seed(rank)
    local_experts = nn.ModuleList(
        [Expert(dim).to(device) for _ in range(experts_per_device)]
    )
    # Router weights replicated across devices (small, cheap)
    torch.manual_seed(0)
    router = nn.Linear(dim, n_experts).to(device)

    # Local tokens on this rank
    tokens = torch.randn(n_tokens_per_device, dim, device=device)

    # 1. Route locally: which expert id each token wants
    logits = router(tokens)               # (T, E)
    expert_ids = logits.argmax(dim=-1)    # (T,) values in [0, E)

    # 2. Dispatch all-to-all
    # Decide how many tokens go to each device (an expert lives on device = e // experts_per_device)
    target_device = expert_ids // experts_per_device     # (T,) values 0 or 1
    send_counts = [int((target_device == d).sum().item()) for d in range(world)]

    # Reorder local tokens by destination device
    perm = torch.argsort(target_device)
    sent_tokens = tokens[perm]
    sent_expert_ids = expert_ids[perm]

    # Exchange counts to know how much we'll receive
    recv_counts_t = torch.zeros(world, dtype=torch.int64, device=device)
    send_counts_t = torch.tensor(send_counts, dtype=torch.int64, device=device)
    dist.all_to_all_single(recv_counts_t, send_counts_t)
    recv_counts = recv_counts_t.tolist()

    # All-to-all the tokens themselves
    n_recv = sum(recv_counts)
    recv_tokens = torch.empty(n_recv, dim, device=device)
    dist.all_to_all_single(
        recv_tokens, sent_tokens.contiguous(),
        output_split_sizes=recv_counts, input_split_sizes=send_counts,
    )
    recv_expert_ids = torch.empty(n_recv, dtype=expert_ids.dtype, device=device)
    dist.all_to_all_single(
        recv_expert_ids, sent_expert_ids.contiguous(),
        output_split_sizes=recv_counts, input_split_sizes=send_counts,
    )

    # 3. Run local experts on the tokens that landed here
    out = torch.empty_like(recv_tokens)
    for local_idx in range(experts_per_device):
        global_eid = rank * experts_per_device + local_idx
        mask = recv_expert_ids == global_eid
        if mask.any():
            out[mask] = local_experts[local_idx](recv_tokens[mask])

    # 4. All-to-all back: send outputs home
    final_recv = torch.empty(sum(send_counts), dim, device=device)
    dist.all_to_all_single(
        final_recv, out.contiguous(),
        output_split_sizes=send_counts, input_split_sizes=recv_counts,
    )

    # Un-permute back to original token order
    final = torch.empty_like(final_recv)
    final[perm] = final_recv

    if rank == 0:
        print(f"rank{rank}: sent {send_counts}, recv {recv_counts}")
        print(f"rank{rank}: input  norm: {tokens.norm().item():.3f}")
        print(f"rank{rank}: output norm: {final.norm().item():.3f}")
        print("each token visited exactly one expert via two all-to-alls")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
