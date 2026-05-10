"""
Build a 2-D DeviceMesh and compose FSDP2 + TP on the same model.

Run on 4 GPUs:
    torchrun --standalone --nproc_per_node=4 mesh_compose.py
On 2 GPUs the mesh degenerates to (1, 2) or (2, 1) — both still work.
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
)


class Block(nn.Module):
    def __init__(self, dim: int = 512, hidden: int = 2048) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.w2(torch.relu(self.w1(x))))


class Model(nn.Module):
    def __init__(self, n_blocks: int = 4) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([Block() for _ in range(n_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for b in self.blocks:
            x = b(x)
        return x


def main() -> None:
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # Choose a (dp, tp) shape from world size
    if world >= 4:
        dp, tp = world // 2, 2
    else:
        dp, tp = world, 1

    mesh = init_device_mesh("cuda", (dp, tp), mesh_dim_names=("dp_shard", "tp"))
    if rank == 0:
        print(f"DeviceMesh: ({dp},{tp}) on {world} ranks")
        print(f"  dp_shard sub-mesh: {mesh['dp_shard']}")
        print(f"  tp sub-mesh:       {mesh['tp']}")

    torch.manual_seed(0)
    model = Model().to(device)

    # Apply TP first (per-block)
    if tp > 1:
        for blk in model.blocks:
            parallelize_module(
                blk,
                mesh["tp"],
                {"w1": ColwiseParallel(), "w2": RowwiseParallel()},
            )

    # Then FSDP2 on top, sharding the (already-TP-sharded) parameters across dp_shard
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    if dp > 1:
        for blk in model.blocks:
            fully_shard(blk, mesh=mesh["dp_shard"], mp_policy=mp)
        fully_shard(model, mesh=mesh["dp_shard"], mp_policy=mp)

    # Forward + backward + step
    x = torch.randn(8, 64, 512, device=device)
    y = model(x)
    loss = y.pow(2).mean()
    loss.backward()
    torch.cuda.synchronize()

    if rank == 0:
        print(f"output shape {tuple(y.shape)}  loss {loss.item():.4f}")
        # Sample one weight to confirm both shardings stacked
        w = model.blocks[0].w1.weight
        print(f"blocks[0].w1.weight: shape={tuple(w.shape)}, type={type(w).__name__}")
        if hasattr(w, "placements"):
            print(f"  placements (per mesh dim): {w.placements}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
