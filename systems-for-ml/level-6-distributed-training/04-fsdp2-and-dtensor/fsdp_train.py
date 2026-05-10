"""
Same toy transformer as Topic 02, now sharded with FSDP2 (`fully_shard`).
Uses DeviceMesh, per-block sharding, mixed precision, sharded checkpointing.

Run:
    torchrun --standalone --nproc_per_node=2 fsdp_train.py
"""

import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.utils.data import DataLoader, Dataset, DistributedSampler


class SyntheticTokens(Dataset):
    def __init__(self, n: int = 4096, seq: int = 256, vocab: int = 8192) -> None:
        self.x = torch.randint(0, vocab, (n, seq))

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, i: int) -> torch.Tensor:
        return self.x[i]


class TinyTransformer(nn.Module):
    def __init__(self, vocab: int = 8192, dim: int = 512, nlayers: int = 6) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        layer = nn.TransformerEncoderLayer(dim, nhead=8, dim_feedforward=2048, batch_first=True)
        self.blocks = nn.ModuleList([layer.__class__(dim, 8, 2048, batch_first=True) for _ in range(nlayers)])
        self.head = nn.Linear(dim, vocab, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.emb(x)
        for b in self.blocks:
            h = b(h)
        return self.head(h)


def main() -> None:
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # 1-D DeviceMesh for pure FSDP (extend later for HSDP/TP composition)
    mesh = init_device_mesh("cuda", mesh_shape=(world,), mesh_dim_names=("dp_shard",))

    torch.manual_seed(0)  # match init across ranks
    model = TinyTransformer().to(device)

    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    )

    # Per-block sharding then global wrap
    for block in model.blocks:
        fully_shard(block, mesh=mesh, mp_policy=mp_policy)
    fully_shard(model, mesh=mesh, mp_policy=mp_policy)

    # Optimizer must be created *after* FSDP wrap
    optim = torch.optim.AdamW(model.parameters(), lr=3e-4)
    loss_fn = nn.CrossEntropyLoss()

    ds = SyntheticTokens()
    sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True, drop_last=True)
    loader = DataLoader(ds, batch_size=8, sampler=sampler, num_workers=2, pin_memory=True)

    t0 = time.time()
    n_tokens = 0
    peak = 0
    for epoch in range(1):
        sampler.set_epoch(epoch)
        for step, batch in enumerate(loader):
            batch = batch.to(device, non_blocking=True)
            x, y = batch[:, :-1], batch[:, 1:]
            logits = model(x)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            n_tokens += x.numel() * world
            peak = max(peak, torch.cuda.max_memory_allocated())
            if rank == 0 and step % 10 == 0:
                dt = time.time() - t0
                print(f"step {step:4d}  loss {loss.item():.3f}  tok/s {n_tokens/dt:,.0f}  "
                      f"peak {peak/1e9:.2f} GB")
            if step >= 50:
                break

    # Sharded checkpoint — every rank writes its own shard, no gather
    state = {"model": model.state_dict(), "optim": optim.state_dict()}
    dcp.save(state, checkpoint_id="ckpt_fsdp_step50")
    if rank == 0:
        print("sharded checkpoint written: ckpt_fsdp_step50/")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
