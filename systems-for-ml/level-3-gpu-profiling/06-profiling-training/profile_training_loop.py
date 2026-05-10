"""
Profile a small training loop with torch.profiler. Includes deliberate anti-patterns
so you can find them in the trace:

  1. CPU-bound dataloader (no num_workers, no pin_memory)
  2. Unfused AdamW
  3. Synchronous H2D copy

Run:
    pip install torch
    python profile_training_loop.py

Open the trace at https://ui.perfetto.dev to see the patterns.

After this, modify the script to fix each anti-pattern and re-profile.
The deltas between runs are your case study.
"""

import time
from pathlib import Path

import torch
from torch import nn
from torch.profiler import (
    ProfilerActivity,
    profile,
    record_function,
    schedule,
    tensorboard_trace_handler,
)
from torch.utils.data import DataLoader, Dataset


class SyntheticDataset(Dataset):
    """Deliberately slow __getitem__ to make the dataloader the bottleneck."""

    def __init__(self, size: int = 1000):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Simulate "real" tokenization / preprocessing — pure CPU work
        time.sleep(0.005)  # 5ms per item
        x = torch.randn(1024)
        y = torch.randn(1024)
        return x, y


def make_loader(
    batch_size: int = 64,
    num_workers: int = 0,            # ANTI-PATTERN: 0 means no parallelism
    pin_memory: bool = False,         # ANTI-PATTERN: blocks H2D
):
    ds = SyntheticDataset(size=2000)
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )


def make_model(device):
    return nn.Sequential(
        nn.Linear(1024, 4096),
        nn.GELU(),
        nn.Linear(4096, 4096),
        nn.GELU(),
        nn.Linear(4096, 1024),
    ).to(device)


def train_step(model, batch, optimizer, criterion, device, non_blocking: bool = False):
    x, y = batch
    with record_function("h2d"):
        x = x.to(device, non_blocking=non_blocking)
        y = y.to(device, non_blocking=non_blocking)

    with record_function("forward"):
        out = model(x)

    with record_function("loss"):
        loss = criterion(out, y)

    with record_function("backward"):
        loss.backward()

    with record_function("optim_step"):
        optimizer.step()
        optimizer.zero_grad()


def main():
    if not torch.cuda.is_available():
        raise SystemExit("Needs CUDA.")
    device = torch.device("cuda")

    # ANTI-PATTERN MODE — change these to see the deltas
    USE_FUSED_ADAMW = False
    USE_NUM_WORKERS = 0
    USE_PIN_MEMORY = False
    USE_NON_BLOCKING_H2D = False

    print(f"Config: fused_adam={USE_FUSED_ADAMW}, num_workers={USE_NUM_WORKERS}, "
          f"pin_memory={USE_PIN_MEMORY}, non_blocking={USE_NON_BLOCKING_H2D}")

    model = make_model(device)
    loader = make_loader(num_workers=USE_NUM_WORKERS, pin_memory=USE_PIN_MEMORY)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=USE_FUSED_ADAMW)
    criterion = nn.MSELoss()

    trace_dir = Path("traces")
    trace_dir.mkdir(exist_ok=True)
    sched = schedule(skip_first=2, wait=1, warmup=1, active=3, repeat=1)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=sched,
        on_trace_ready=tensorboard_trace_handler(str(trace_dir)),
        record_shapes=True,
    ) as prof:
        for step, batch in enumerate(loader):
            if step >= 10:
                break
            with record_function(f"step_{step}"):
                train_step(model, batch, optimizer, criterion, device,
                           non_blocking=USE_NON_BLOCKING_H2D)
            prof.step()

    # Summary
    print("\n=== Top ops by CUDA time ===")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

    print(f"\n=== Top ops by CPU time ===")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))

    print(f"\nTrace at {trace_dir.absolute()}/")
    print("Open in https://ui.perfetto.dev to see the timeline.")
    print()
    print("Look for:")
    print("  1. GPU idle gaps between steps — CPU-bound dataloader")
    print("  2. Yellow H2D bars on the critical path — sync copy")
    print("  3. Long sequence of tiny optim kernels — unfused AdamW")
    print()
    print("Change the USE_* flags at the top of this script and re-run to see the deltas.")


if __name__ == "__main__":
    main()
