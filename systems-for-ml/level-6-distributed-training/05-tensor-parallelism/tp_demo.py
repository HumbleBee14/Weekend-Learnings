"""
Apply tensor parallelism to a single MLP and compare with the single-GPU version.

Run:
    torchrun --standalone --nproc_per_node=2 tp_demo.py
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
)


class SwiGLU(nn.Module):
    """A small SwiGLU MLP — same structure as Llama-style blocks."""

    def __init__(self, dim: int = 1024, hidden: int = 4096) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))


def main() -> None:
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    mesh = init_device_mesh("cuda", mesh_shape=(world,), mesh_dim_names=("tp",))

    torch.manual_seed(0)
    mlp = SwiGLU(dim=1024, hidden=4096).to(device)

    if rank == 0:
        print("Before TP:")
        print(f"  w1.weight: {tuple(mlp.w1.weight.shape)}")
        print(f"  w2.weight: {tuple(mlp.w2.weight.shape)}")
        print(f"  w3.weight: {tuple(mlp.w3.weight.shape)}")

    parallelize_module(
        mlp,
        mesh,
        {
            "w1": ColwiseParallel(),
            "w3": ColwiseParallel(),
            "w2": RowwiseParallel(),
        },
    )

    if rank == 0:
        print("After TP (each rank holds a shard):")
        print(f"  w1.weight: {tuple(mlp.w1.weight.shape)}  (DTensor placement: {mlp.w1.weight.placements})")
        print(f"  w2.weight: {tuple(mlp.w2.weight.shape)}  (DTensor placement: {mlp.w2.weight.placements})")

    # Forward — same input on every rank, same output (after the rowwise all-reduce inside w2)
    torch.manual_seed(42)
    x = torch.randn(4, 64, 1024, device=device)
    y = mlp(x)
    if rank == 0:
        print(f"output shape: {tuple(y.shape)}  (matches single-GPU shape)")
        print(f"output mean : {y.mean().item():+.6f}  std: {y.std().item():.4f}")

    # Quick benchmark
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(50):
        _ = mlp(x)
    end.record()
    torch.cuda.synchronize()
    if rank == 0:
        print(f"50 forwards: {start.elapsed_time(end):.2f} ms total ({start.elapsed_time(end)/50:.3f} ms/forward)")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
