# 06 — Profiling Training

## Files

- `CONCEPTS.md` — what's different from inference (backward, optim, dataloader, NCCL, checkpointing); the 5 common findings; MFU vs Goodput; HTA; FlightRecorder
- `profile_training_loop.py` — small training loop with three deliberate anti-patterns; toggle flags at the top to compare baselines vs fixed

## Quickstart

```bash
pip install torch
python profile_training_loop.py
# → traces/*.json   open at https://ui.perfetto.dev
```

## The four versions to run

The script has four flags at the top. Run the script with each combination:

```python
# Run 1: anti-pattern baseline
USE_FUSED_ADAMW = False
USE_NUM_WORKERS = 0
USE_PIN_MEMORY = False
USE_NON_BLOCKING_H2D = False

# Run 2: fix dataloader
USE_NUM_WORKERS = 4
USE_PIN_MEMORY = True

# Run 3: fix H2D copy
USE_NON_BLOCKING_H2D = True

# Run 4: fix optimizer
USE_FUSED_ADAMW = True
```

After each run, save the trace and look at:
- Total step time (printed in the table)
- Top ops by CUDA time
- Top ops by CPU time
- The Perfetto timeline

You should see step time drop with each fix:

```
Run 1 (all anti-patterns):     ~120 ms/step
Run 2 (workers + pin):         ~70 ms/step    (dataloader fixed)
Run 3 (+ non-blocking H2D):    ~65 ms/step    (small win)
Run 4 (+ fused AdamW):         ~55 ms/step    (optim fixed)
```

The order matters — fix the biggest bottleneck first. CPU-bound dataloader is usually that.

## What to spot in each trace

```
Run 1 — anti-pattern baseline:

GPU stream: [step 0] [        ] [step 1] [        ] [step 2] [        ]
                       ↑ huge gap        ↑ huge gap        ↑ huge gap
CPU thread: [DataLoader.__getitem__ ............]
                ↑ pure CPU, 5ms per item × 64 = 320ms per batch

Diagnosis: dataloader-bound. GPU is idle most of the time.
```

```
Run 2 — workers + pin:

GPU stream: [step 0][step 1][step 2][step 3]   ← packed, no gaps
CPU thread: [worker 0][worker 1][worker 2][worker 3]   ← parallel

Diagnosis: dataloader keeps up; GPU busy.
```

```
Run 4 — all fixed:

step_4 row in the trace:
  forward: 5ms
  backward: 10ms
  optim_step: 1.5ms   ← was 8ms before fused AdamW

Diagnosis: balanced training step.
```

## Try

- **Make the model bigger.** With a tiny model, the dataloader anti-pattern dominates. With a 1B-parameter model, the compute dominates and dataloader matters less. The bottleneck shifts with scale.
- **Run with `torch.compile(model)`.** Fwd+bwd kernels fuse and shrink. Re-profile.
- **Multi-GPU**: run with `torchrun --nproc-per-node=2 profile_training_loop.py` (you'd need to adapt the script for DDP). Now NCCL allreduce shows up. Profile both ranks. Use HTA for cross-rank analysis.
- **Add deliberate work imbalance**: make rank 0 do more work than rank 1. Watch HTA spot the straggler.

## Distributed extension

For real multi-GPU training profiling:

```python
# In your DDP/FSDP training script
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(skip_first=10, wait=1, warmup=1, active=3, repeat=1),
    on_trace_ready=tensorboard_trace_handler(f"./traces/rank_{rank}"),
    record_shapes=True,
) as prof:
    for step, batch in enumerate(loader):
        train_step(model, batch, optimizer, criterion)
        prof.step()
        if step >= 16:
            break
```

After the run, you have `traces/rank_0/`, `traces/rank_1/`, etc. Use HTA:

```python
from hta.trace_analysis import TraceAnalysis
analyzer = TraceAnalysis(trace_dir="./traces/")
print(analyzer.get_idle_time_breakdown())
print(analyzer.get_comm_comp_overlap())
```

For NCCL hangs (different problem), enable Flight Recorder:

```bash
TORCH_NCCL_TRACE_BUFFER_SIZE=2000 \
TORCH_NCCL_ENABLE_TIMING=1 \
TORCH_NCCL_DUMP_ON_TIMEOUT=1 \
torchrun --nproc-per-node 8 train.py
```

## Where this goes

Topic 07 is the full case study workflow. You'll take everything from Topics 01-06 — nsys, ncu, torch.profiler, HTA, the roofline framework — and apply them to a deliberately slow PyTorch model. End-to-end: profile → hypothesize → fix → measure delta. The capstone.
