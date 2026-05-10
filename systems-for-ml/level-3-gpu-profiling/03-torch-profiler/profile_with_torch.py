"""
Profile a small training loop with torch.profiler.

Run:
    pip install torch
    python profile_with_torch.py

Outputs:
  - A formatted table to stdout (sorted by GPU time)
  - traces/<host>_<pid>.<ts>.pt.trace.json for Perfetto visualization

Open the JSON at https://ui.perfetto.dev (drag and drop).
"""

import os
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


def make_model_and_data(device):
    model = nn.Sequential(
        nn.Linear(1024, 4096),
        nn.GELU(),
        nn.Linear(4096, 4096),
        nn.GELU(),
        nn.Linear(4096, 1024),
    ).to(device)
    x = torch.randn(64, 1024, device=device)
    target = torch.randn(64, 1024, device=device)
    return model, x, target


def train_step(model, x, target, optimizer, criterion):
    """One step with record_function annotations at semantic boundaries."""
    with record_function("forward"):
        out = model(x)

    with record_function("loss"):
        loss = criterion(out, target)

    with record_function("backward"):
        loss.backward()

    with record_function("optim_step"):
        optimizer.step()
        optimizer.zero_grad()


def main():
    if not torch.cuda.is_available():
        raise SystemExit("Needs CUDA.")
    device = torch.device("cuda")

    model, x, target = make_model_and_data(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    trace_dir = Path("traces")
    trace_dir.mkdir(exist_ok=True)

    # Schedule:
    #   skip_first=5: first 5 steps are warmup, not recorded
    #   wait=1, warmup=1, active=3, repeat=1: record steps 7, 8, 9
    sched = schedule(skip_first=5, wait=1, warmup=1, active=3, repeat=1)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=sched,
        on_trace_ready=tensorboard_trace_handler(str(trace_dir)),
        record_shapes=True,
        with_stack=True,
    ) as prof:
        for step in range(15):
            with record_function(f"step_{step}"):
                train_step(model, x, target, optimizer, criterion)
            prof.step()  # advances the schedule — without this, schedule stays in "wait"

    # Print top 20 ops by GPU time
    print("\n=== Top ops by CUDA time ===")
    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=20,
        header="Top GPU consumers",
    ))

    print(f"\n=== Trace files written to {trace_dir.absolute()} ===")
    for f in trace_dir.glob("*.json"):
        print(f"  {f.name}  ({f.stat().st_size / 1e6:.1f} MB)")
    print("\nOpen at https://ui.perfetto.dev (drag and drop the .json).")


if __name__ == "__main__":
    main()
