# 05 — Profiling Inference

## The applied version of Topics 01-04

Take what you've learned (nsys, ncu, torch.profiler, roofline) and apply it to a real LLM serving workload — your `mini-serve` from Level 1, or a vLLM deployment.

Goal by end of topic: identify the slowest component of an LLM inference pipeline using measured data, predict what fix would help, and have the artifact (`profiling-mini-serve.md`) that justifies Level 4's optimizations.

## The five places time goes in inference

```
Request arrives
      ↓
1. Tokenization (CPU)
      ↓
2. Prefill (compute-bound)
      ↓
3. Decode (memory-bound, autoregressive)
      ↓
4. Detokenization (CPU)
      ↓
5. Network/Python overhead (between every step)
      ↓
Response leaves
```

Each has a typical bottleneck:

| Phase | Typical bottleneck | Tool to find it |
|---|---|---|
| Tokenization | Python single-threaded | torch.profiler (look for CPU bars) |
| Prefill | Compute-bound matmul (attention QKV, MLP) | ncu on the matmul kernel |
| Decode | Memory-bound (reads weights, low AI) | ncu shows memory SOL near 100% |
| Detokenization | Python | torch.profiler |
| Python overhead | GIL, scheduler logic | nsys (CUDA HW gaps while CPU is busy) |

## vLLM's built-in profiling

vLLM has first-class profiler integration as of 2025. Two paths:

### (a) torch.profiler

```bash
VLLM_TORCH_PROFILER_DIR=/tmp/vllm_traces vllm serve meta-llama/Llama-3.1-8B-Instruct
```

In another terminal:
```bash
curl -X POST http://localhost:8000/start_profile
# send some test requests
curl -X POST http://localhost:8000/stop_profile
```

Trace lands in `/tmp/vllm_traces`. Open in Perfetto.

### (b) Nsight Systems

```bash
nsys profile \
  --trace-fork-before-exec=true \
  --cuda-graph-trace=node \
  --capture-range=cudaProfilerApi --capture-range-end=repeat \
  -o vllm_trace \
  vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --profiler-config.profiler cuda

# In another terminal, send the load
```

`--trace-fork-before-exec=true` is critical — vLLM spawns worker processes; without this they're invisible.
`VLLM_WORKER_MULTIPROC_METHOD=spawn` env var helps too.

## Isolating prefill from decode

These two phases have completely different profiles. Profile them separately.

```bash
# Prefill-dominant: long input, almost no output
vllm bench serve --input-len 4096 --output-len 1 --num-prompts 64

# Decode-dominant: short input, long output
vllm bench serve --input-len 1 --output-len 256 --num-prompts 64 --max-concurrency 64
```

Profile each. You'll see:

- **Prefill profile** — long matmul kernels, high SM utilization, kernel-launch density tight. Compute-bound.
- **Decode profile** — many short kernels per token, memory throughput near peak, gaps between steps as Python schedules. Memory-bound, with overhead.

## Common findings in real LLM serving

Patterns you'll see when you actually do this:

### 1. Decode is memory-bound, period

Profile a decode workload. Look at `ncu` for the attention or MLP kernel during decode:

```
Compute SOL: ~10%
Memory SOL: ~85%
Achieved bandwidth: 2.8 TB/s on H100 (out of 3.35)
```

That's the memory wall. The fix isn't a better kernel — it's reducing data: quantization (FP8/FP4), KV cache compression, paged KV.

### 2. Python overhead dominates at small batch sizes

In `nsys`, with `--python-sampling=true`:

```
GPU stream: [k1] [k2]   [k3] [k4]   [k5] ...
Python:     [scheduler....][sampler...][detokenize...]
                ↑ GPU idle here while Python runs
```

For batch=1 decode at high token rate, **15-30% of step latency is Python**. vLLM v1's scheduler rewrite cut this; SGLang's "overlap scheduler" hides it. The fix: CUDA graphs to capture the model forward, leaving only scheduling in Python.

### 3. The famous `cudaEventSynchronize` blow-up

Red Hat's vLLM profiling case study (https://developers.redhat.com/articles/2026/03/09/5-steps-triage-vllm-performance) found 84.7% of CUDA API time was synchronization. Diagnosis: the Python scheduler synchronizes more than necessary, gating GPU work. Fix: pipelining.

You'll see this as a tall `cudaEventSynchronize` bar in the CUDA API row of `nsys`.

### 4. AllReduce dominates at TP≥4 without NVLink

Tensor parallelism splits each layer across N GPUs. Each layer ends with an allreduce to combine partial outputs. With NVLink, allreduce is fast. On PCIe or cross-node IB, it dominates.

In a TP=8 trace on PCIe, you might see:

```
Per-layer breakdown:
  Compute: 0.5 ms
  AllReduce: 3.2 ms    ← 6× more than compute!
```

The fix: tensor-parallel only across NVLink boundaries; pipeline-parallel across nodes.

### 5. Padding waste in prefill

When you batch requests of varied lengths, naive padding pads everything to max length. A 50-token prompt batched with a 5000-token prompt wastes 4950 tokens of compute on the short one.

Visible in the trace: long prefill matmul kernels even though most requests are short. Fix: chunked prefill (vLLM V1 default), or per-request prefill scheduling.

## What you'll build

A profiling report for `mini-serve` (Project 1's deliverable from Level 1):

```
mini-serve/
└── reports/
    ├── week1.md                    # from Level 1
    └── profiling-mini-serve.md     # NEW — from this topic
```

Structure of `profiling-mini-serve.md`:

1. **Setup** — model, batch size, sequence length, GPU, what you profiled
2. **Timeline screenshot** — annotated nsys trace showing prefill vs decode vs Python
3. **Top 5 kernels by time** — from `torch.profiler` table
4. **The dominant kernel's roofline placement** — AI and SOL from `ncu --set full`
5. **Diagnosis** — which regime (compute / memory / overhead) and what the next fix is
6. **Predicted impact of Level 4's fixes** — what should change after paged KV + continuous batching

The last point is the key. Level 4 will replace your naive batcher with paged KV. *Predict* the throughput delta from this profile, then verify after Level 4. That's how a real perf engineer thinks.

## What to look for in your specific `mini-serve`

You wrote this in Level 1. Known issues by design:

1. **Static batching** — head-of-line blocking should show up as kernels running well below capacity when one slow request holds the batch
2. **Padding waste** — variable-length inputs padded to max
3. **No continuous batching** — fast users wait for slow ones
4. **No prefix caching** — repeated prefixes recomputed

Each of these has a fingerprint in the profiler. Find them.

## Pitfalls

1. **Profiling a single request.** Doesn't show batching dynamics. Always profile under realistic concurrency (16+ users via Locust).
2. **Forgetting warmup.** First few requests include CUDA context init, kernel JIT, compile. Discard.
3. **Profiling the warm path only.** Cold start (first request after model load) is its own regime; profile separately.
4. **Trusting GPU util %.** "GPU is 95% utilized" can hide lots of sins — most of that 95% might be a stalled kernel (high warp activity but not producing work). Look at SOL, not utilization.
5. **Micro-benchmarking instead of system profiling.** A kernel that's 30% faster in isolation might be 3% faster end-to-end if it's not on the critical path. Profile the system, not the kernel, for real impact.

## References

- vLLM profiling guide — https://docs.vllm.ai/en/stable/contributing/profiling/
- Red Hat: 5 steps to triage vLLM performance (Mar 2026) — https://developers.redhat.com/articles/2026/03/09/5-steps-triage-vllm-performance
- Red Hat: Profiling vLLM on RHEL — https://developers.redhat.com/articles/2025/10/16/profiling-vllm-inference-server-gpu-acceleration-rhel
- SGLang benchmark and profiling — https://github.com/sgl-project/sglang/blob/main/docs/developer_guide/benchmark_and_profiling.md
- ROCm profiling DeepSeek-V3 on SGLang — https://rocm.blogs.amd.com/software-tools-optimization/kernel-analysis-deep/README.html
