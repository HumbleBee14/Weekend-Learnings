# 01 — Nsight Systems Basics

## Files

- `CONCEPTS.md` — what nsys is, the four timeline patterns to recognize, the 2026 command line, NVTX annotations, gotchas
- `profile_a_workload.py` — a small training-style workload with NVTX annotations + a deliberate synchronous H2D copy (visible in the trace)

## Setup

You need Nsight Systems installed:

```bash
# Linux (Ubuntu/Debian)
sudo apt install nsight-systems-cli

# Or full install (with GUI):
# Download from https://developer.nvidia.com/nsight-systems

# macOS / Windows
# Use a remote Linux box; the GUI runs locally and opens .nsys-rep files from anywhere.

# Verify
nsys --version  # should report 2026.x.x
```

Free Colab does not have `nsys`. Rent an A100/H100 instance on RunPod or similar (~$1-2/hr).

## Quickstart

```bash
# 1. Install dependencies
pip install torch

# 2. Run baseline (no profiling)
python profile_a_workload.py

# 3. Profile with nsys
nsys profile \
    -t cuda,nvtx \
    --cuda-graph-trace=node \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    -o trace.nsys-rep \
    python profile_a_workload.py

# 4. Open trace.nsys-rep in the Nsight Systems GUI
# Either:
#   - Local: nsys-ui trace.nsys-rep
#   - Remote: scp trace.nsys-rep to your laptop, open it locally
```

## What to look for in the trace

When you open the file in the GUI, four things to spot:

1. **The NVTX row at the top.** You should see `training_loop` containing 20 `step_N` ranges, each with `forward`, `backward`, `optim_step` inside. This is the structural map of your code.

2. **The yellow H2D copy bars.** Inside each `h2d_copy` range you'll see a yellow `cudaMemcpyAsync` bar (or `cudaMemcpy` if Torch took the sync path). This is the cost of moving the input from CPU to GPU each step. Real workloads either pin the memory (`pin_memory=True` on the dataloader) or overlap the copy with compute on a separate stream.

3. **Kernel launch density.** Zoom into one `forward` range. You should see ~5-10 kernels (one per `Linear` + activation). If they're tightly packed with no gaps, the GPU is busy. If there are gaps between every kernel, you're CPU-bound on the launch path.

4. **Stream usage.** Click on a kernel and check the "Stream" property. Most likely everything's on stream 7 (PyTorch's default). All serial. To overlap H2D with compute you'd put the copy on a separate stream.

## Try

- **Add `non_blocking=True`** to the `.to(device)` call. Re-trace. The H2D copy now goes async — and starts overlapping with the previous step's optim if you also pin the source tensor.
- **Increase the batch size to 256.** Re-trace. Compare kernel duration vs the gap-between-kernels ratio. Bigger batches → kernels longer → CPU launch overhead becomes a smaller fraction.
- **Replace the model with a single `nn.Linear(1024, 1024)`.** Now each step is dominated by launch overhead, not compute. The trace looks completely different — gaps everywhere.
- **Wrap the `train_step` body with `@torch.compile`.** Re-trace. The kernel count drops dramatically because Inductor fuses pointwise ops. Per-step time should drop too.

## Gotchas you'll hit

- `nsys: command not found` — install Nsight Systems CLI
- Trace file is huge — bound it via `--capture-range=cudaProfilerApi`, or shorten the loop
- Permissions error — `sudo modprobe nvidia NVreg_RestrictProfilingToAdminUsers=0` once per boot
- GUI is too slow on the trace file — that's normal for >500MB traces. Reduce capture window.
- "I see no NVTX ranges" — ensure you passed `-t nvtx` (not just `-t cuda`)

## Where this goes

You now have a timeline. Topic 02 (Nsight Compute) zooms into a single kernel from this trace and tells you *why* it's slow at hardware level. Topic 03 (PyTorch profiler) shows the lighter-weight version of what we just did.
