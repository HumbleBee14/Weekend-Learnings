"""
Minimal DDP training loop on a tiny transformer block.
Demonstrates: init_process_group, DistributedSampler, DDP wrapping,
gradient bucketing, profiler trace export.

Run:
    torchrun --standalone --nproc_per_node=2 ddp_train.py
"""

import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler


class SyntheticTokens(Dataset):
    def __init__(self, n_samples: int = 4096, seq_len: int = 256, vocab: int = 8192) -> None:
        self.x = torch.randint(0, vocab, (n_samples, seq_len))

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, i: int) -> torch.Tensor:
        return self.x[i]


class TinyTransformer(nn.Module):
    def __init__(self, vocab: int = 8192, dim: int = 512, nlayers: int = 4) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        layer = nn.TransformerEncoderLayer(dim, nhead=8, dim_feedforward=2048, batch_first=True)
        self.blocks = nn.TransformerEncoder(layer, num_layers=nlayers)
        self.head = nn.Linear(dim, vocab, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.emb(x)
        h = self.blocks(h)
        return self.head(h)


def main() -> None:
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    model = TinyTransformer().to(device)
    model = DDP(
        model,
        device_ids=[local_rank],
        bucket_cap_mb=25,
        gradient_as_bucket_view=True,
        static_graph=True,
    )

    ds = SyntheticTokens()
    sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True, drop_last=True)
    loader = DataLoader(ds, batch_size=8, sampler=sampler, num_workers=2, pin_memory=True)

    optim = torch.optim.AdamW(model.parameters(), lr=3e-4)
    loss_fn = nn.CrossEntropyLoss()

    do_profile = os.environ.get("PROFILE") == "1"
    profiler_ctx = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./trace"),
        record_shapes=True,
        with_stack=False,
    ) if do_profile else None

    if profiler_ctx is not None:
        profiler_ctx.__enter__()

    t0 = time.time()
    n_tokens = 0
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
            if profiler_ctx is not None:
                profiler_ctx.step()
            if rank == 0 and step % 10 == 0:
                dt = time.time() - t0
                print(f"step {step:4d}  loss {loss.item():.3f}  tok/s {n_tokens/dt:,.0f}")
            if step >= 50:
                break

    if profiler_ctx is not None:
        profiler_ctx.__exit__(None, None, None)
        if rank == 0:
            print("trace written to ./trace; open with chrome://tracing or Perfetto")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
