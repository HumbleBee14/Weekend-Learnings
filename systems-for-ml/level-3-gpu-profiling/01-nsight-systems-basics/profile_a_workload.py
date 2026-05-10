"""
Run a small training-style workload with NVTX annotations, designed to be profiled with nsys.

Run it directly to see baseline timing:
    python profile_a_workload.py

Then profile it with Nsight Systems:
    nsys profile -t cuda,nvtx --cuda-graph-trace=node \
        --capture-range=cudaProfilerApi --capture-range-end=stop \
        -o trace.nsys-rep \
        python profile_a_workload.py

Open trace.nsys-rep in the Nsight Systems GUI. Look for:
  - The "forward" / "backward" / "optim" NVTX bars at the top
  - Gaps in the CUDA HW row (GPU idle)
  - The synchronous H2D copy we deliberately included (yellow bar)
"""

import time

import torch
import torch.cuda.nvtx as nvtx
from torch import nn


def make_model_and_data(device):
    model = nn.Sequential(
        nn.Linear(1024, 4096),
        nn.GELU(),
        nn.Linear(4096, 4096),
        nn.GELU(),
        nn.Linear(4096, 1024),
    ).to(device)

    # Create the input on CPU on purpose — forces an H2D copy each step.
    # This is the "synchronous H2D copy" anti-pattern we want to see in the trace.
    x_cpu = torch.randn(64, 1024)
    target = torch.randn(64, 1024, device=device)
    return model, x_cpu, target


def train_step(model, x_cpu, target, optimizer, criterion, device):
    """One step with NVTX annotations at semantic boundaries."""
    with nvtx.range("h2d_copy"):
        # Synchronous H2D copy — visible in trace as a yellow bar.
        # The .to(device) without non_blocking=True forces a sync.
        x = x_cpu.to(device)

    with nvtx.range("forward"):
        out = model(x)

    with nvtx.range("loss"):
        loss = criterion(out, target)

    with nvtx.range("backward"):
        loss.backward()

    with nvtx.range("optim_step"):
        optimizer.step()
        optimizer.zero_grad()


def main():
    if not torch.cuda.is_available():
        raise SystemExit("Needs CUDA.")
    device = torch.device("cuda")

    model, x_cpu, target = make_model_and_data(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    # Warmup — exclude the first few steps from profiling
    for _ in range(5):
        train_step(model, x_cpu, target, optimizer, criterion, device)
    torch.cuda.synchronize()

    # Tell nsys to start recording. With --capture-range=cudaProfilerApi,
    # only the work between start() and stop() ends up in the trace.
    torch.cuda.profiler.start()

    # The "interesting" part — 20 annotated training steps
    with nvtx.range("training_loop"):
        t0 = time.perf_counter()
        for step in range(20):
            with nvtx.range(f"step_{step}"):
                train_step(model, x_cpu, target, optimizer, criterion, device)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

    torch.cuda.profiler.stop()

    print(f"20 steps in {elapsed * 1000:.1f}ms ({elapsed * 50:.1f}ms per step)")
    print("If profiled with nsys, open trace.nsys-rep in the GUI.")
    print("Look for: gaps in the CUDA HW row, yellow H2D bars, NVTX ranges at the top.")


if __name__ == "__main__":
    main()
