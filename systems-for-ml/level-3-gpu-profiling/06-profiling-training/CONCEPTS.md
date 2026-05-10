# 06 — Profiling Training

## What's different from inference profiling

Inference (Topic 05) is mostly memory-bound and prefill+decode-shaped. Training adds:

- **Backward pass** — typically 2× the cost of forward
- **Optimizer step** — non-trivial, often 5-15% of step
- **Data pipeline** — dataloaders, tokenization on CPU
- **Multi-GPU collectives** — allreduce, allgather, reduce-scatter dominate at scale
- **Checkpointing** — periodic, slow, blocks training

The bottleneck in training is rarely the model forward — it's almost always one of the four extras above.

## The five common findings in real training

### 1. Dataloader-bound — the silent killer

Symptom in trace:
```
GPU stream: [step 1][        ][step 2][        ][step 3][        ]
                     ↑ idle      ↑ idle       ↑ idle
CPU thread: [        DataLoader.__getitem__        ]
```

GPU is idle waiting for the next batch. Almost every training run starts here. Diagnostic: in `torch.profiler` look at total CPU time vs GPU time per step. If CPU is the longer pole, you're dataloader-bound.

Fixes (in order of impact):
- `num_workers=N` (4-8 is usually enough)
- `pin_memory=True` for faster H2D
- `persistent_workers=True` (avoids worker re-spawn each epoch)
- `prefetch_factor=4`
- GPU-side data loading (DALI, NVIDIA's accelerated loader)
- Sequence packing (avoid padding waste)
- Streaming data (Mosaic StreamingDataset for pretraining)

If after all of this you're still dataloader-bound, the data pipeline is just too slow — pre-tokenize, write packed binary files, mmap them.

### 2. Allreduce-dominated at scale

Symptom in trace (multi-GPU):
```
rank 0: [forward][backward][allreduce─────][optim]
rank 1: [forward][backward    ][allreduce───][optim]
                                     ↑
                                 30-60% of step time

```

NCCL bars in the trace consume more time than compute. Diagnostic: HTA's "communication breakdown" or just visual inspection.

Fixes:
- Bucket sizing tuning (`DDP(bucket_cap_mb=...)` or FSDP equivalents)
- Gradient accumulation (fewer allreduces per step)
- ZeRO/FSDP shard size tuning
- Switch from DDP to FSDP if memory allows it
- 5D parallelism rebalancing (Level 6 of curriculum)
- Better topology — TP only across NVLink, not across IB

### 3. Optimizer step too slow

Symptom: in the table, `optim_step` takes 10-20% of total step time.

Diagnostic: long sequence of small kernels, often `_amp_foreach_non_finite_check_and_unscale` (mixed-precision overhead) and per-parameter Adam updates.

Fix: `torch.optim.AdamW(fused=True)` collapses thousands of tiny per-tensor kernels into one. Often a 5-10× speedup of the optim step.

### 4. FSDP communication-overlap fails

Symptom: in FSDP, the next forward shouldn't start until the previous backward's allgather completes — but ideally those should overlap. When they don't, you see:

```
compute stream: [fwd_layer_N───][fwd_layer_N+1]
comm stream:    [unshard_N+1...]                    ← didn't overlap

```

Diagnostic: `record_param_comms` in PyTorch 2.9+ exposes the FSDP-specific events. Look for unshard/reshard not running in parallel with compute.

Fix: increase the number of overlapping prefetch slots (`limit_all_gathers=False`, more prefetch), check whether the comm stream has enough work to hide.

### 5. Checkpoint stalls

Symptom: every N steps, a multi-second pause where everything stops.

Diagnostic: in the trace, see a blocking `cudaMemcpyAsync(D2H)` followed by file I/O.

Fix: async DCP (`torch.distributed.checkpoint` async mode), or peer replication (each rank checkpoints to a peer's disk in parallel).

## The MFU vs Goodput vocabulary

Two metrics that matter at scale:

**MFU (Model FLOPs Utilization)**:
```
MFU = achieved FLOPs / peak FLOPs
```
"Good" MFU in 2026: 40-55% for dense H100 training, 30-40% for MoE. Frontier (TorchTitan, Litespark) reports 60-89% on small-scale dense, dropping at >128 GPUs.

**Goodput**:
```
Goodput = useful training throughput / peak
        = (elapsed_steps × tokens_per_step) / (wall_time × peak_throughput)
```
Accounts for failures, restarts, checkpointing overhead. At 10k+ GPUs, goodput is typically 50-70% of MFU because of interruptions.

Modern training systems track *both*: MFU as the kernel-efficiency metric, Goodput as the cluster-efficiency metric. Don't conflate them.

## HTA — multi-rank trace analysis

`torch.profiler` produces one JSON per rank. For multi-rank analysis, use HTA (Holistic Trace Analysis):

```python
from hta.trace_analysis import TraceAnalysis

analyzer = TraceAnalysis(trace_dir="./traces/")

# Per-rank GPU idle time — find the straggler
analyzer.get_idle_time_breakdown()

# Comm vs compute overlap — how well is FSDP hiding allreduce?
analyzer.get_comm_comp_overlap()

# Frequent kernel patterns — which kernels dominate
analyzer.get_frequent_cuda_kernel_patterns()

# Trace diff — compare two runs (before vs after optimization)
analyzer.compare_traces("./traces_before/", "./traces_after/")
```

The key feature for distributed training: HTA aligns NCCL collectives across ranks, so you can spot the slow rank (straggler) immediately.

## FlightRecorder — for hangs, not perf

Different tool for a different problem. **NCCL hangs**, where one rank silently waits forever, used to be untouchable. Flight Recorder captures a per-rank ring buffer of recent collectives:

```bash
TORCH_NCCL_TRACE_BUFFER_SIZE=2000 \
TORCH_NCCL_ENABLE_TIMING=1 \
TORCH_NCCL_DUMP_ON_TIMEOUT=1 \
torchrun --nproc-per-node 8 train.py
```

On hang or timeout, each rank dumps its collective history. The analyzer color-codes mismatched collectives across ranks → instant "rank 3 called all_reduce while everyone else called all_gather."

This is part of training profiling now, but for a different failure class than performance.

## The case study workflow for training

Same as inference, with one extra step at the start:

1. **Establish single-GPU baseline** — make sure the model works and measure tokens/sec at single-GPU.
2. **Scale to multi-GPU** — measure tokens/sec/GPU. The scaling efficiency tells you how comm-bound you are.
3. **Profile** — `torch.profiler` with NCCL events, multi-rank HTA.
4. **Identify regime** — dataloader, communication, optimizer, or compute.
5. **Apply ONE fix** — one variable at a time.
6. **Re-profile, re-measure, document.**
7. **Repeat.**

## The deliverable

A profiling report for a training loop:

```
reports/
└── training-profile.md
```

Should include:
- Single-GPU baseline (tokens/sec, MFU)
- Multi-GPU baseline (tokens/sec/GPU, scaling efficiency)
- Diagnosis (which of the 5 findings — dataloader, comm, optim, FSDP, checkpoint)
- Top 3 kernels by time
- Communication-vs-compute overlap percentage
- Goodput (if you simulated failures)

## Pitfalls

1. **Profiling without warmup.** First N steps include compilation, JIT, allocator warmup. Always `skip_first=10` or more.
2. **Forgetting to `prof.step()`.** Schedule never advances, no output.
3. **Measuring only on rank 0.** In multi-rank, the slowest rank determines step time. Profile all ranks.
4. **Trusting MFU alone.** A run with high MFU but frequent restarts has terrible Goodput. Both matter.
5. **Profiling at small scale and extrapolating.** Comm patterns scale non-linearly. Profile at the actual scale you'll deploy.
6. **Treating dataloader as already optimal.** Most teams' dataloaders are 10-30% better with one config tweak.

## References

- PyTorch profiler — https://docs.pytorch.org/docs/stable/profiler.html
- HTA / TraceInsight — https://github.com/facebookresearch/HolisticTraceAnalysis
- Flight Recorder blog — https://pytorch.org/blog/flight-recorder-a-new-lens-for-understanding-nccl-watchdog-timeouts/
- TorchTitan paper (production LLM pretraining) — https://arxiv.org/html/2410.06511v1
- Stas Bekman — ML engineering performance — https://github.com/stas00/ml-engineering/blob/master/training/performance/README.md
- Goodput metric (Google) — https://cloud.google.com/blog/products/ai-machine-learning/goodput-metric-as-measure-of-ml-productivity
- Automated trace collection — https://pytorch.org/blog/automated-trace-collection/
