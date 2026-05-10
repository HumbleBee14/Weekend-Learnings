# 05 — Profiling Inference

## Files

- `CONCEPTS.md` — the five places time goes in inference, vLLM's profiler integration, prefill vs decode isolation, common findings (decode is memory-bound, Python overhead, allreduce dominance, padding waste)
- `profile_mini_serve.py` — client-side driver that hits your Level-1 server with three workload shapes (prefill-dominant, decode-dominant, mixed) so you can profile each separately

## Quickstart

Two terminals:

```bash
# Terminal A: start the Level-1 mini-serve (or vLLM)
cd ../../level-1-inference-serving/03-request-batching
uvicorn server:app --workers 1 --port 8000

# Terminal B: send load
cd ../../../level-3-gpu-profiling/05-profiling-inference
pip install torch httpx
python profile_mini_serve.py
```

For a profiled run, restart the server under one of the profilers:

```bash
# nsys
nsys profile -t cuda,nvtx --cuda-graph-trace=node \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    -o mini_serve.nsys-rep \
    uvicorn server:app --workers 1

# Or torch.profiler — embed the schedule API in mini-serve's batcher loop
```

Run `python profile_mini_serve.py` while the server is profiling.

## What you should see across the three workloads

```
prefill_dominant: 32 requests at concurrency 16
  total wall time: 8.2s          ← compute-bound, batches well
  median:          1800ms
  p99:             2100ms

decode_dominant: 32 requests at concurrency 16
  total wall time: 18.1s         ← memory-bound, slower per token
  median:          4200ms
  p99:             5100ms

mixed_realistic:
  total wall time: 11.5s
  median:          2400ms
  p99:             4800ms        ← long tail from head-of-line blocking
```

In the trace files:

- **prefill_dominant** — long matmul kernels, high SOL, kernel launches packed tight
- **decode_dominant** — many short kernels, gaps for Python overhead, memory SOL near peak
- **mixed_realistic** — head-of-line blocking visible: short requests waiting in batches behind long ones

## The deliverable

This topic produces `reports/profiling-mini-serve.md` for Project 1. Structure (see CONCEPTS.md):

1. Setup details
2. Annotated nsys trace screenshot
3. Top 5 kernels by time (from torch.profiler table)
4. Dominant kernel's roofline placement (AI + SOL from ncu)
5. Diagnosis (compute / memory / overhead-bound)
6. Predicted impact of Level 4's paged KV + continuous batching

The last point is the test: can you predict the throughput delta from the profile? If yes, you understand the system. If no, re-read the trace.

## Try

- **Profile vLLM on the same workloads.** It should win on the decode-dominant case (paged KV, continuous batching). Compare to your mini-serve numbers — that delta is what Level 4 will close.
- **Add `--python-sampling=true`** to nsys. See exactly which Python frames are running while the GPU waits.
- **Find the head-of-line blocking** in the mixed workload. Look for short requests sitting in the queue while a long one finishes. NVTX-annotate request ID for clarity.
- **Profile with concurrency 1 vs 16.** Concurrency 1 should be Python+kernel-launch dominated. Concurrency 16 should saturate the GPU.

## Where this goes

Topic 06 is the training analog of this — same toolset applied to a training loop. Topic 07 is the full case study where you profile, hypothesize, fix, measure delta. The artifact this topic produces (`profiling-mini-serve.md`) is the *evidence* that justifies Level 4's optimization choices.
