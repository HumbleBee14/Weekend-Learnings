# Systems for ML — LLM Infrastructure & Performance Engineering

This is NOT PyTorch. This is about the **systems** built on top of PyTorch to train, serve, and optimize LLMs at scale.

**Prerequisite:** Complete `python-pytorch/` Levels 1–7 first. You need to have built a transformer, trained it, and profiled it before touching this.

## Who needs this

If you want to work on:
- LLM inference infrastructure (vLLM / SGLang / TensorRT-LLM / TGI / llama.cpp)
- Training platforms (RLHF / RLXF / post-training systems)
- GPU performance engineering (CUDA, Triton, kernel work)
- ML platform / MLOps at scale (multi-tenant serving, autoscaling, cost)
- **Local / on-device intelligence** (Apple Silicon + MLX, GGUF, agentic IDEs)
- AI Performance / AI Systems Engineer roles

You'll also leave with **awareness** of the compiler stack (MLIR / LLVM / `torch.compile` internals) — enough to decide whether to specialize there as a separate, deeper track later.

...this is the curriculum. These are the same job postings you see at Anthropic, OpenAI, Meta, Databricks, Together, Fireworks, Anyscale, Baseten, Modal, plus every Fortune 500 building internal AI platforms — *and* the wave of local-first / on-device AI startups (Ollama, LM Studio, agentic IDE companies, private-AI vendors) hiring for the same skills minus the datacenter.

## The two paths the field is splitting into

```
Frontier scale (datacenter)        Local scale (on-device)
────────────────────────           ─────────────────────────
vLLM / SGLang / TRT-LLM            llama.cpp / MLX / GGUF
H100 / B200 clusters               M-series unified memory, consumer GPUs
3D parallelism, NCCL, FSDP         Aggressive quantization (4/3/1.58-bit)
$/Mtok at fleet scale              Zero-latency, zero-API-cost agents
RLHF at scale                      QLoRA + local DPO personalization
```

Most curricula teach only the left column. This one covers both, because the job market now hires for both — and the same engineer who can tune vLLM on an H100 is the one a local-AI startup wants tuning llama.cpp on an M5 Max.

## Mindset shift

```
python-pytorch/:     "How does the model work?"
systems-for-ml/:     "How do we make it fast, cheap, reliable, and observable at scale?"
```

The job is rarely "make it work." The job is: 50ms p99, 10× cheaper than GPT-4, never goes down, ships a new fine-tune every week, runs on the GPUs we can actually buy.

## Pedagogy — why the order matters

1. **Build it yourself before using the real thing.** You hand-roll an inference server in Week 1, then your *own* paged KV cache manager in Week 4 — so that when vLLM appears in Week 5, you've effectively reimplemented its core data structure and feel exactly which problem each flag solves.
2. **Understand the substrate before optimizing.** GPU mental model and profiling come *before* optimization tricks — you can't optimize what you can't measure.
3. **Evaluate everything you change.** Quantization without `lm-eval-harness` is guessing. Every optimization week ends with a quality check.
4. **Treat ML systems as distributed systems.** Most "ML infra" failures are networking, data pipeline, or queueing failures wearing an ML costume. Week 6 covers interconnects, tail latency, and failure injection alongside parallelism. Week 7 covers backpressure, multi-tenant fairness, and request hedging.
5. **Production-shape the capstone.** Week 7 is observability, autoscaling, routing, fairness, and cost — what platform teams actually own.

## Weekly Roadmap

### Week 1 — Inference Serving (build your own)

Build a real LLM inference server — accept prompts, return generated tokens. No vLLM yet. You need to feel the pain it solves.

| # | Topic | What you build |
|---|-------|---------------|
| 01 | inference-server-basics | FastAPI endpoint that loads your MiniGPT and serves `/generate` |
| 02 | streaming-tokens | Server-Sent Events (SSE) — stream tokens as they generate (like ChatGPT) |
| 03 | request-batching | Batch multiple user requests into one forward pass for throughput |
| 04 | latency-vs-throughput | Measure and graph: batch size vs latency vs throughput tradeoffs |
| 05 | load-testing | Locust / k6 — generate concurrent load, measure TTFT, ITL, p50/p99 |
| 06 | local-first-touch | Run Ollama + llama.cpp locally — same prompts, see the *other* end of the spectrum |

**Outcome:** You have a working LLM API server. You can speak fluently about TTFT (time-to-first-token), ITL (inter-token-latency), tokens/sec, and why naive batching breaks under variable sequence lengths. You've also seen what "local serving" looks like before we deep-dive into it later.

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
| 05 | gpu-memory-hierarchy | HBM vs L2 vs SRAM — why FlashAttention is a memory optimization, not a compute one |
| 06 | flash-attention-walkthrough | Read the FA2 paper, identify each tile/load/recompute step in the kernel |

**Outcome:** You understand GPU execution model. You can write a kernel. You can read a FlashAttention kernel and explain it.

**Compute:** Colab T4 (free) for CUDA. Triton needs GPU.

---

### Week 3 — GPU Profiling & Bottleneck Analysis

Profile real workloads before you try to optimize them. Profiling moved *before* optimization on purpose — every Week 4 fix is justified by a Week 3 measurement.

| # | Topic | What you build |
|---|-------|---------------|
| 01 | nsight-systems-basics | Capture and read a GPU timeline (kernel launches, memory transfers, gaps) |
| 02 | nsight-compute-basics | Per-kernel metrics — occupancy, achieved bandwidth, SM utilization |
| 03 | torch-profiler | PyTorch profiler + Chrome trace viewer for end-to-end traces |
| 04 | compute-vs-memory-bound | Roofline model — diagnose which bottleneck your code has |
| 05 | profiling-inference | Profile your Week 1 server, find the slowest component |
| 06 | profiling-training | Profile a training loop — data loading? attention? comms? |
| 07 | optimization-case-study | Slow model → profile → hypothesis → fix → measure improvement |

**Outcome:** You can profile any PyTorch workload, place it on a roofline, and identify the next bottleneck.

**Compute:** Colab GPU for Nsight. CPU profiling works locally.

---

### Week 4 — LLM Optimization Techniques

Take your inference server and make it fast. Every topic ends with a quality check, not just a throughput number.

| # | Topic | What you build |
|---|-------|---------------|
| 01 | quantization-basics | int8 / fp16 / fp8 inference — reduce memory, increase throughput |
| 02 | advanced-quantization | AWQ, GPTQ, INT4 — when each is right, what they cost in quality |
| 03 | local-quant-formats | GGUF and EXL2 — the formats that make 70B run on a laptop |
| 04 | extreme-quantization | 3-bit, 2-bit, BitNet 1.58-bit — where the frontier of local inference is going |
| 05 | quality-evaluation | `lm-eval-harness` + perplexity + task accuracy — every quant tested |
| 06 | torch-compile | `torch.compile`, graph capture, fusion — let the compiler do it |
| 07 | kernel-fusion | Why separate ops are slow, what fused kernels look like by hand |
| 08 | kv-cache-naive | Implement a contiguous KV cache yourself — feel fragmentation pain firsthand |
| 09 | kv-cache-paged | Build your own paged KV cache manager — pages, block table, free list (mini-vLLM) |
| 10 | kv-cache-eviction | LRU vs sliding-window vs RadixAttention prefix-sharing — benchmark each |
| 11 | long-context-stress | 100K-token workloads — see naive systems collapse, paged systems hold |
| 12 | speculative-decoding | Draft model + acceptance rate — measure both speed and quality drift |
| 13 | continuous-batching | Serve multiple users with different sequence lengths efficiently |
| 14 | structured-output | Outlines / JSON schema — constrained decoding without quality loss |

**Outcome:** You can explain (and partially implement) every major LLM optimization trick. For each, you know the *quality cost*, not just the speedup. You've built your own paged KV cache — meaning you could have written vLLM's core data structure yourself.

**Compute:** Mix of CPU (concepts) and Colab GPU (benchmarks).

---

### Week 5 — Production Inference Engines

The week the original curriculum was missing. Now that you've built a server and know how to optimize, run the real engines and benchmark them against your handmade one.

| # | Topic | What you build |
|---|-------|---------------|
| 01 | vllm-hello-world | Serve Qwen/Llama via vLLM, hit the OpenAI-compatible endpoint |
| 02 | vllm-features | Prefix caching, chunked prefill, tensor parallelism flags |
| 03 | sglang-and-radixattention | Same model on SGLang — when prefix sharing matters most |
| 04 | tgi-and-tensorrt-llm | HuggingFace TGI + TensorRT-LLM — the other two engines you'll see |
| 05 | llama-cpp-deep-dive | GGML internals, CPU + Metal + CUDA backends, when llama.cpp beats vLLM |
| 06 | engine-bake-off | Same model + same load test across all five — write up the differences |
| 07 | multi-lora-serving | Train tiny LoRAs, hot-swap them via vLLM's multi-LoRA endpoint |
| 08 | offline-batch-inference | vLLM offline mode for million-doc scoring jobs |
| 09 | speculative-decoding-in-prod | Enable spec-decode in vLLM, measure end-to-end gain |

**Outcome:** You can pick the right engine for a workload, tune its flags, and serve dozens of fine-tunes off one base model. This is the single most-asked-about week in interviews.

**Compute:** RunPod / Lambda / Vast.ai bursts. Budget ~$30–50 for the week. A single L4 or A10 (≈$0.40/hr) gets you through most of it.

---

### Week 6 — Distributed Training & Networking

Scale beyond one GPU. The hidden lesson: most distributed training failures are *networking* failures or *data pipeline* failures — not compute ones. So we open the box on collectives, interconnects, and data loading first, then layer the framework abstractions on top.

| # | Topic | What you build |
|---|-------|---------------|
| 00 | collectives-and-nccl | All-reduce, all-gather, reduce-scatter — ring vs tree, bandwidth vs latency |
| 01 | interconnects | TCP vs RDMA vs InfiniBand vs NVLink — bandwidth, latency, when each matters |
| 02 | data-parallel-from-scratch | DDP — what does `loss.backward()` actually communicate? |
| 03 | data-loading-and-tokenization | Streaming datasets, sharding, tokenization bottlenecks — why training fails on data, not GPUs |
| 04 | deepspeed-zero | ZeRO Stage 1/2/3 — partition optimizer states, gradients, weights |
| 05 | fsdp | PyTorch's native Fully Sharded Data Parallel (ZeRO-3 alternative) |
| 06 | tensor-parallelism | Split a single layer across GPUs (Megatron-style) |
| 07 | pipeline-parallelism | Split model into stages, overlap forward/backward across stages |
| 08 | 3d-parallelism | Combine TP + PP + DP — how 70B+ models actually train |
| 09 | ray-and-multi-node | Ray for distributed job scheduling and multi-node training |
| 10 | tail-latency-distributed | Stragglers, p99 explosion across nodes — measure and mitigate |
| 11 | failure-injection | Kill a node mid-training and mid-inference — what actually breaks, what recovers |
| 12 | checkpointing-and-elasticity | Async checkpointing, elastic training, resume-from-failure |

**Outcome:** You understand how 70B+ models are trained across GPU clusters. You know which parallelism axis to add when you hit which wall, why interconnect choice dominates the cost-per-token math, and what actually happens when a node dies at hour 47 of a training run.

**Compute:** Conceptual on CPU. Real runs need multi-GPU (Colab Pro or RunPod for short bursts).

---

### Week 7 — ML Platform & Production (Capstone)

Build a mini version of what platform teams at Scale / OpenAI / Anthropic actually build. The original curriculum stopped at scheduler + registry; production also means observability, autoscaling, and cost.

| # | Topic | What you build |
|---|-------|---------------|
| 01 | platform-architecture | Design doc: training → eval → registry → serving pipeline |
| 02 | training-job-scheduler | Submit training jobs, track metrics, handle failures |
| 03 | evaluation-pipeline | Automated eval: run benchmarks after training, compare runs |
| 04 | model-registry | Save, version, promote, and roll back model checkpoints |
| 05 | observability | Prometheus + Grafana — tokens/sec, queue depth, GPU util, KV-cache fill |
| 06 | inference-routing | L4 vs L7 routing, sticky sessions for prefix-cache locality, request hedging |
| 07 | multi-tenant-fairness | Per-tenant quotas, noisy-neighbor isolation, fair queueing across customers |
| 08 | backpressure-and-queueing | Little's Law applied to LLM serving — when to shed load vs queue vs hedge |
| 09 | autoscaling | Scale replicas on `vllm:num_requests_waiting` and queue latency |
| 10 | cost-economics | $/Mtok per engine + quant combo — when to quantize vs scale horizontally |
| 11 | safety-and-abuse | Rate limiting, prompt-injection at the infra layer, output filtering |
| 12 | mini-rlxf | End-to-end: SFT → reward model → RLHF pipeline orchestration |

**Outcome:** You've built a tiny version of Scale's RLXF platform — and unlike most tutorials, yours has dashboards, autoscaling, and a cost model.

**Compute:** All CPU. This is system design + orchestration code. Optionally point it at the Week 5 engines for a real end-to-end demo.

---

### Week 8 — Local & On-Device Intelligence (Apple Silicon, MLX, Personalization)

The other half of the field. Datacenter inference is half the job market; local-first AI (agentic IDEs, private-AI startups, on-device assistants) is the other half. Apple Silicon's unified memory architecture changed what's possible on a laptop — a maxed-out M-series machine now runs 70B-class models with no GPU rack involved.

| # | Topic | What you build |
|---|-------|---------------|
| 01 | unified-memory-mental-model | Why UMA changes the game — no host↔device copies, KV cache lives in shared RAM |
| 02 | mlx-basics | Apple's MLX framework — tensor ops, lazy eval, autograd on Metal |
| 03 | mlx-vs-pytorch | Same model in PyTorch (CUDA) vs MLX (Metal) — measure throughput, memory, latency |
| 04 | metal-shaders | Write a custom Metal kernel — the Apple equivalent of CUDA C++ |
| 05 | local-serving-stack | llama.cpp + LM Studio + Ollama as production-grade local serving |
| 06 | agentic-ide-backend | Build a "canvas-first" local agentic loop — zero API cost, sub-100ms TTFT |
| 07 | qlora-on-device | Quantized LoRA fine-tune on your laptop — personalize a model to your code style |
| 08 | local-dpo | Direct Preference Optimization locally — the model learns your preferences without sending data anywhere |
| 09 | cpu-simd-for-llms | AVX-512 / NEON kernels in llama.cpp — when CPU beats GPU and exactly why |
| 10 | privacy-and-data-residency | Threat model for on-device AI — what local actually buys you vs cloud |

**Outcome:** You can ship a local-first AI product. You understand UMA, MLX, Metal, and the GGUF ecosystem deeply enough to build something like Cursor's local mode or a private agentic IDE.

**Compute:** Your own Mac (M-series strongly recommended). No cloud spend.

---

### Week 9 — Compiler Stack Awareness (High-Level Tour)

> **Scope note:** The point of *this* curriculum is to make you fluent **end-to-end** — train, optimize, serve, scale, ship. Once you have that, the compiler stack is a separate, deeper specialization you can choose to dive into if it interests you.
>
> So this week is **a high-level tour, not a compiler course.** The goal is awareness: when you read a job description that says "MLIR / LLVM experience a plus," or when `torch.compile` does something surprising, you know what's actually happening underneath. If you fall in love with it, that's a *whole separate track* — not something we cram into one week here.

| # | Topic | What you'll be able to explain |
|---|-------|-------------------------------|
| 01 | the-lowering-picture | High-level view: PyTorch graph → IR → hardware code, who does what |
| 02 | torch-compile-internals | Dynamo + Inductor at a conceptual level — how Week 4's `torch.compile` actually works |
| 03 | xla-vs-inductor | The two big compilation paths and where each is used (JAX/XLA vs PyTorch/Inductor) |
| 04 | what-mlir-and-llvm-are | Why an extra IR layer exists, what dialects mean, why ML needed it |
| 05 | accelerator-landscape | Conceptual: how Groq / Cerebras / TPU / Tenstorrent compilers differ from GPU compilers |

**Outcome:** You can read a `torch.compile` trace and roughly follow it down to hardware. You can hold a conversation with a compiler engineer without bluffing. You know enough to *decide* whether you want to go deeper.

**If you want to go deeper** (separate track, not part of this curriculum): a real MLIR/LLVM specialization means writing custom passes, contributing to IREE or Triton, targeting new accelerators. That's months of dedicated study — start with the LLVM Kaleidoscope tutorial and the MLIR Toy tutorial, then read the StableHLO / IREE / Triton source. Roles: **AI Compiler Engineer** at NVIDIA, Apple, AMD, Groq, Cerebras, Tenstorrent, Modular.

**Compute:** Reading and small experiments. CPU is fine.

---

## How to use compute wisely

```
Your daily workflow:
  90% LOCAL (CPU)  → design, code, test logic, write orchestration
  10% GPU BURSTS   → profile, benchmark, validate on real hardware

Free GPU options:
  Google Colab (T4)  → CUDA kernels, profiling, short training runs
  Kaggle             → longer sessions, similar hardware

Paid (use sparingly, $30-100 total for the curriculum):
  RunPod / Vast.ai   → Week 5 engines (~$0.40-1/hr per L4/A10)
  Lambda / RunPod    → Week 6 multi-GPU (~$2-4/hr per A100 pair)
```

Spin down between sessions. Set hard budget alarms. The whole curriculum should cost less than one month of ChatGPT Pro.

## What you'll be able to claim on a resume

After Weeks 1–7:

- "Built a multi-engine LLM serving stack benchmarking vLLM, SGLang, TGI, TensorRT-LLM, and llama.cpp with reproducible load tests."
- "Implemented a paged KV cache manager from scratch — pages, block table, eviction policies — and benchmarked LRU vs sliding-window vs prefix-sharing strategies on 100K-token workloads."
- "Profiled and optimized a transformer inference path from X tok/s to Y tok/s, validated with `lm-eval-harness`."
- "Implemented and compared DDP, FSDP, ZeRO-3, and 3D parallelism on a multi-GPU cluster; characterized interconnect impact (TCP vs RDMA vs NVLink) on training throughput."
- "Designed a mini ML platform with model registry, request hedging, multi-tenant fairness, queue-depth autoscaling, and per-request cost accounting."
- "Ran failure-injection tests across distributed training and inference — measured p99 tail-latency blowup under stragglers and built recovery paths."

After Week 8 (local / on-device):

- "Built a local-first agentic system on Apple Silicon using MLX and llama.cpp — sub-100ms TTFT with zero API cost."
- "Fine-tuned a 7B model on-device using QLoRA + local DPO; benchmarked MLX vs PyTorch on the same workload."
- "Wrote AVX-512 / NEON-aware kernels for CPU inference, characterizing where llama.cpp on CPU beats a small GPU on cost-per-token."

After Week 9 (compiler awareness):

- "Comfortable reading `torch.compile` / Inductor traces and explaining how a PyTorch graph lowers toward hardware; familiar with the MLIR/LLVM ecosystem at a conceptual level."

These map directly to the bullet points in real **LLM Inference Engineer / ML Systems Engineer / AI Performance Engineer** job descriptions — across both datacenter and local-AI companies. *Compiler Engineer* roles (NVIDIA, Apple, AMD, Groq, Cerebras, Tenstorrent, Modular) require a deeper, separate specialization beyond this curriculum.

## Capstone Project: MiniLLM RLXF Platform

After completing all 7 weeks, you will have built:

```
┌────────────────────────────────────────────────────────┐
│                MiniLLM RLXF Platform                   │
│                                                        │
│  ┌──────────┐   ┌──────────┐   ┌──────────────────┐    │
│  │ Training │──→│ Eval     │──→│ Serving (engine) │    │
│  │ Pipeline │   │ Pipeline │   │ vLLM / SGLang    │    │
│  └──────────┘   └──────────┘   └──────────────────┘    │
│       │              │                 │               │
│   FSDP/ZeRO     lm-eval-harness   Multi-LoRA +         │
│   3D-parallel   regression gates  Continuous batch     │
│                                                        │
│  ┌──────────────────────────────────────────────────┐  │
│  │   Observability — Prometheus + Grafana           │  │
│  │   tokens/sec · p99 TTFT · GPU util · $/Mtok      │  │
│  └──────────────────────────────────────────────────┘  │
│                                                        │
│  ┌──────────────────────────────────────────────────┐  │
│  │   Autoscaling — queue-depth driven replica scale │  │
│  └──────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────┘
```

Three pillars (train → eval → serve), instrumented end-to-end, with the same observability and cost discipline real platform teams ship.
