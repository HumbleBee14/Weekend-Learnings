"""
The modern PyTorch memory profiling pattern: record allocator history → snapshot → visualize.

Run:
    pip install torch
    python memory_snapshot.py

Outputs:
  - snap.pickle

Then open at https://pytorch.org/memory_viz (drag and drop the pickle).
You'll see:
  - Every allocation with stack trace
  - Peak memory and what was holding it at peak
  - Fragmentation (gaps in the address space)
  - Reserved-but-not-used memory (the allocator's reservation strategy)
"""

import torch
from torch import nn


def main():
    if not torch.cuda.is_available():
        raise SystemExit("Needs CUDA.")
    device = torch.device("cuda")

    # Start recording every alloc/dealloc + a stack trace per event.
    # max_entries caps the buffer; for long runs, dump and restart periodically.
    torch.cuda.memory._record_memory_history(max_entries=100_000)

    # Run something memory-interesting: a model that allocates large activations
    model = nn.Sequential(
        nn.Linear(1024, 16384),
        nn.GELU(),
        nn.Linear(16384, 16384),
        nn.GELU(),
        nn.Linear(16384, 1024),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for step in range(10):
        x = torch.randn(64, 1024, device=device)
        target = torch.randn(64, 1024, device=device)

        out = model(x)
        loss = (out - target).pow(2).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Dump the snapshot
    torch.cuda.memory._dump_snapshot("snap.pickle")

    # Stop recording (otherwise it keeps growing forever)
    torch.cuda.memory._record_memory_history(enabled=None)

    peak_mb = torch.cuda.max_memory_allocated() / 1e6
    reserved_mb = torch.cuda.max_memory_reserved() / 1e6
    print(f"Peak allocated: {peak_mb:.1f} MB")
    print(f"Peak reserved:  {reserved_mb:.1f} MB")
    print(f"Reservation overhead: {reserved_mb - peak_mb:.1f} MB ({(reserved_mb - peak_mb) / peak_mb * 100:.1f}%)")
    print()
    print("Wrote snap.pickle")
    print("Drag it onto https://pytorch.org/memory_viz to visualize.")


if __name__ == "__main__":
    main()
