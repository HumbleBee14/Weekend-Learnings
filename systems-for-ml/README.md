# Systems for ML — LLM Infrastructure & Performance Engineering

This is NOT PyTorch. This is about the **systems** built on top of PyTorch to train, serve, and optimize LLMs at scale.

**Prerequisite:** Complete `python-pytorch/` Levels 1–7 first. You need to have built a transformer, trained it, and profiled it before touching this.

## Who needs this

If you want to work on:
- LLM inference infrastructure
- Training platforms (RLHF/RLXF systems)
- GPU performance engineering
- ML platform / MLOps at scale

...this is the curriculum.

## Mindset shift

```
python-pytorch/:     "How does the model work?"
systems-for-ml/:     "How do we make it fast, cheap, and reliable at scale?"
```

## Weekly Roadmap

### Week 1 — Inference Serving (FastAPI + model serving)

Build a real LLM inference server — accept prompts, return generated tokens.

| # | Topic | What you build |
|---|-------|---------------|
| 01 | inference-server-basics | FastAPI endpoint that loads your MiniGPT and serves `/generate` |
| 02 | streaming-tokens | Server-Sent Events (SSE) — stream tokens as they generate (like ChatGPT) |
| 03 | request-batching | Batch multiple user requests into one forward pass for throughput |
| 04 | latency-vs-throughput | Measure and graph: batch size vs latency vs throughput tradeoffs |

**Outcome:** You have a working LLM API server. You understand why batching matters.

**Compute:** CPU only. Your MiniGPT from Level 4 is tiny enough.

---

### Week 2 — CUDA & GPU Programming

Stop treating the GPU as a black box. Write actual GPU kernels.

| # | Topic | What you build |
|---|-------|---------------|
| 01 | cuda-mental-model | Threads, blocks, warps, SMs — how GPUs actually execute code |
| 02 | first-cuda-kernels | Vector addition, elementwise ops in CUDA C++ (via Colab) |
| 03 | matrix-multiply | Naive → tiled → shared memory matmul — see why tiling matters |
| 04 | triton-intro | Write same kernels in Triton (Python, not C++) — much easier |
| 05 | gpu-memory-hierarchy | HBM vs L2 vs SRAM — why Flash Attention is a memory optimization, not a compute one |

**Outcome:** You understand GPU execution model. You can write a kernel.

**Compute:** Colab T4 (free) for CUDA. Triton needs GPU.

---

### Week 3 — LLM Optimization Techniques

Take your inference server and make it fast.

| # | Topic | What you build |
|---|-------|---------------|
| 01 | quantization-basics | int8 / fp16 inference — reduce memory, increase throughput |
| 02 | kernel-fusion | Why separate ops are slow, what fused kernels look like |
| 03 | kv-cache-optimization | PagedAttention concept — how vLLM manages KV cache memory |
| 04 | speculative-decoding | Use a small draft model to speed up a large model's generation |
| 05 | continuous-batching | Serve multiple users with different sequence lengths efficiently |

**Outcome:** You can explain (and partially implement) every major LLM optimization trick.

**Compute:** Mix of CPU (concepts) and Colab GPU (benchmarks).

---

### Week 4 — GPU Profiling & Bottleneck Analysis

Profile real workloads and learn to diagnose performance issues.

| # | Topic | What you build |
|---|-------|---------------|
| 01 | nsight-systems-basics | Capture and read a GPU timeline (kernel launches, memory transfers) |
| 02 | compute-vs-memory-bound | Determine which bottleneck your code has and how to fix each |
| 03 | profiling-inference | Profile your inference server end-to-end, find the slowest component |
| 04 | profiling-training | Profile a training loop — discover where time goes (data loading? attention? comms?) |
| 05 | optimization-case-study | Take a slow model, profile it, optimize it, measure improvement |

**Outcome:** You can profile any PyTorch workload and identify the bottleneck.

**Compute:** Colab GPU for Nsight. CPU profiling works locally.

---

### Week 5 — Distributed Training Frameworks

Scale beyond one GPU using third-party frameworks.

| # | Topic | What you build |
|---|-------|---------------|
| 01 | deepspeed-zero | ZeRO Stage 1/2/3 — partition optimizer states, gradients, weights |
| 02 | model-parallelism | Split a model across GPUs when it doesn't fit on one |
| 03 | pipeline-parallelism | Split model into stages, overlap forward/backward across stages |
| 04 | fsdp | PyTorch's native Fully Sharded Data Parallel (ZeRO-3 alternative) |
| 05 | ray-basics | Ray for distributed job scheduling and multi-node training |

**Outcome:** You understand how 70B+ models are trained across GPU clusters.

**Compute:** Conceptual understanding on CPU. Real runs need multi-GPU (Colab Pro or RunPod for short bursts).

---

### Week 6 — ML Platform Design (Capstone)

Build a mini version of what Scale AI / OpenAI infra teams actually build.

| # | Topic | What you build |
|---|-------|---------------|
| 01 | platform-architecture | Design doc: training → evaluation → serving pipeline |
| 02 | training-job-scheduler | Submit training jobs, track metrics, handle failures |
| 03 | evaluation-pipeline | Automated eval: run benchmarks after training, compare runs |
| 04 | model-registry | Save, version, and load model checkpoints |
| 05 | mini-rlxf | End-to-end: SFT → reward model → RLHF pipeline orchestration |

**Outcome:** You've built a tiny version of what Scale's RLXF platform does.

**Compute:** All CPU. This is system design + orchestration code.

---

## How to use compute wisely

```
Your daily workflow:
  90% LOCAL (CPU)  → design, code, test logic, write orchestration
  10% GPU BURSTS   → profile, benchmark, validate on real hardware

Free GPU options:
  Google Colab (T4)  → CUDA kernels, profiling, short training runs
  Kaggle             → longer sessions, similar hardware

Paid (use sparingly, $10-30/month):
  RunPod / Lambda    → multi-GPU experiments (Week 5 only)
```

## Capstone Project: MiniLLM RLXF System

After completing all 6 weeks, you will have built:

```
┌──────────────────────────────────────────────┐
│              MiniLLM RLXF System             │
│                                              │
│  ┌──────────┐   ┌──────────┐   ┌─────────┐   │
│  │ Training │──→│ Eval     │──→│ Serving │   │
│  │ Pipeline │   │ Pipeline │   │ Server  │   │
│  └──────────┘   └──────────┘   └─────────┘   │
│       │                             │        │
│  DeepSpeed       Benchmarks    FastAPI +     │
│  ZeRO            Auto-compare  Batching +    │
│                                Streaming     │
│                                              │
│  ┌───────────────────────────────────────┐   │
│  │          GPU Profiling Layer          │   │
│  │    torch.profiler + Nsight + custom   │   │
│  └───────────────────────────────────────┘   │
└──────────────────────────────────────────────┘
```


