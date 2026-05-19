# Systems for ML — LLM Infrastructure & Performance Engineering

This is NOT PyTorch. This is about the **systems** built on top of PyTorch to train, serve, and optimize LLMs at scale.

**Prerequisite:** Complete `python-pytorch/` Levels 1–7 first. You need to have built a transformer, trained it, and profiled it before touching this.

## Reference texts (read alongside this curriculum)

This repo is the **main guide** — opinionated, project-anchored, LLM-frontier-focused (2026). Three external sources sit next to it as authoritative references. Read the relevant chapter first when starting a new level, then come back here to build the project.

```
  ┌─────────────────────────────────────────────────────────────────┐
  │                                                                 │
  │              this repo (project-first 2026 lab)                 │
  │     build mini-serve, mini-vllm, engine-bakeoff, mini-platform  │
  │                                                                 │
  └────────┬──────────────────┬──────────────────┬─────────────────-┘
           │                  │                  │
   academic foundations   production reality   primary sources
   ──────────────────     ───────────────────  ────────────────
   Reddi (MIT Press)      Kiely (Baseten)      vLLM/SGLang/TRT-LLM
   textbook-grade         practitioner-grade   the actual code
   concepts + Vol 1 / 2   field-current 2026   you'll be running
```

### Primary reference A — *Machine Learning Systems* (Reddi, Harvard / MIT Press 2026)

- **Book:** [mlsysbook.ai](https://mlsysbook.ai/) — free, open-access, MIT Press print edition 2026
- **Repo:** [harvard-edge/cs249r_book](https://github.com/harvard-edge/cs249r_book) — book source + `tinytorch/` + `labs/` + `kits/` + `mlsysim/`
- **Author:** Vijay Janapa Reddi (Harvard CS249r) + community
- **Flavor:** Academic, textbook-grade. Two volumes spanning the entire ML systems lifecycle (single-machine through datacenter-scale).
- **What it gives you we don't:** Systematic coverage of the *foundations* — the ML systems lifecycle, data engineering, frameworks design, hardware acceleration taxonomy, MLOps, responsible AI, sustainable AI, edge intelligence. Vol 1 (Build/Optimize/Deploy, 1–8 GPU); Vol 2 (Scale/Distribute/Govern at production scale).
- **How to use it:** When you want the canonical, citation-grade explanation of a concept (*what is* a roofline? *what is* MLOps? *why does* data engineering matter?). Their `tinytorch/` is a great parallel exercise to your Level 2.

### Primary reference B — *Inference Engineering* (Kiely, Baseten, December 2025)

- **Book:** Philip Kiely, *Inference Engineering*, 259 pages, Dec 2025 (revised April 2026)
- **Local copy:** [`references/Inference-Engineering-Kiely-2025.pdf`](references/Inference-Engineering-Kiely-2025.pdf)
- **Author:** Philip Kiely, Head of Developer Relations at [Baseten](https://baseten.co) — they ship production inference for real customers, real SLAs, real $/Mtok pressure.
- **Flavor:** Practitioner, production-grade. The book that explains how 2026 inference infra is actually built and run, not just modeled.
- **What it gives you we don't:** A field-current narrative across the inference path — model mechanics → hardware (Hopper / Ada / Blackwell / Rubin) → software (vLLM / SGLang / TRT-LLM / Dynamo) → techniques (quant / spec decode / paged KV / disaggregation) → modalities (VLM / embedding / ASR / TTS / image / video gen) → production (autoscaling, multi-cloud, GPU procurement, observability). The 47-page glossary and 26-page curated reading list are genuinely useful references on their own.
- **How to use it:** When you want the *practitioner's view* of why an inference stack looks the way it does. Cross-cuts Levels 1, 4, 5, and 7 — read the corresponding chapter before each.
- **Caveat:** Section 7.6 is "Production Inference with Baseten" — fair given the author, but read it as a vendor case study, not neutral comparison.

### Primary reference C — Inference engine source (the systems themselves)

- **vLLM:** [vllm-project/vllm](https://github.com/vllm-project/vllm) + the PagedAttention paper (Kwon et al. 2023)
- **SGLang:** [sgl-project/sglang](https://github.com/sgl-project/sglang) + RadixAttention paper
- **TensorRT-LLM:** [NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
- **llama.cpp:** [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)
- **NVIDIA Dynamo / llm-d:** the disaggregated-serving production frontier
- **NVIDIA Triton Inference Server:** [triton-inference-server/server](https://github.com/triton-inference-server/server) — the framework-agnostic outer wrapper

These are not "supplementary reading" — they are the systems you'll be benchmarking and deploying. The READMEs in `level-5-production-engines/` link the exact source files to read.

### Mapping — both books → this repo

| Topic | Reddi (academic) | Kiely (practitioner) | This repo (lab) |
|---|---|---|---|
| Inference engineering as a discipline | — | Ch 0, Preface | Top-level README, Level 1 intro |
| TTFT / ITL / latency percentiles / online vs offline | — | Ch 1 (§1.4) | Level 1 Topic 04, 05 |
| LLM mechanics (attention, MoE, KV) | Vol 1 DL Primer | Ch 2 (§2.1–2.2) | python-pytorch/ prereq + Level 4 |
| Arithmetic intensity / ops:byte / roofline | Vol 1 Benchmarking | Ch 2 (§2.4) | Level 3 Topic 04 (roofline) |
| GPU architectures (Hopper / Ada / Blackwell / Rubin) | Vol 1 AI Acceleration | Ch 3 | Level 2 + Level 9 Topic 05 |
| MIG / multi-GPU instances | — | Ch 3 (§3.3) | Level 7 Topic 10 (autoscaling), Level 6 |
| Local inference (desktop/mobile) | Vol 2 Edge Intelligence | Ch 3 (§3.5) | Level 8 (full level) |
| CUDA / kernels / fusion | Vol 1 AI Acceleration | Ch 4 (§4.1) | Level 2 + sibling `compiler-and-kernels/` |
| PyTorch / model file formats / ONNX / TRT runtime | Vol 1 AI Frameworks | Ch 4 (§4.2) | Level 5 Topic 13 |
| Inference engines (vLLM / SGLang / TRT-LLM) | Vol 1 Serving | Ch 4 (§4.3) | **Level 5 Topics 01–07** |
| NVIDIA Dynamo | Vol 2 Distributed Inference | Ch 4 (§4.4) | Level 5 Topic 09 |
| Benchmarking / profiling | Vol 1 Benchmarking | Ch 4 (§4.5) | Level 3 + Level 5 Topic 07 (bake-off) |
| Quantization (FP8 / FP4 / INT4 / approaches) | Vol 1 Efficient AI | Ch 5 (§5.1) | **Level 4 Topics 01–05** |
| Speculative decoding (draft/Medusa/EAGLE/n-gram) | Vol 1 Model Optimizations | Ch 5 (§5.2) | Level 4 Topic 13 + Level 5 Topic 12 |
| Prefix caching / KV reuse / cache-aware routing | — | Ch 5 (§5.3) | Level 4 Topic 11 + Level 7 Topic 06 |
| KV cache placement / tiering / LMCache | — | Ch 5 (§5.3.2) | Level 7 Topic 12 |
| Long context handling | — | Ch 5 (§5.3.4) | Level 4 Topic 12 |
| Tensor / expert / multi-node parallelism (inference) | Vol 2 Distributed Training | Ch 5 (§5.4) | Level 5 + Level 6 (training side) |
| Disaggregated prefill/decode (incl. Dynamic) | Vol 2 Distributed Inference | **Ch 5 (§5.5)** | **Level 5 Topic 08–09** |
| VLM / omni-modal | — | Ch 6 (§6.1) | Level 5 Topic 14 |
| Embedding models | — | Ch 6 (§6.2) | Level 5 Topic 13 (ORT bench) |
| ASR / TTS / speech-to-speech | — | Ch 6 (§6.3–6.4) | *not currently covered — Kiely fills this* |
| Image generation (DiT, few-step, kernel opts) | — | Ch 6 (§6.5) | *not currently covered — Kiely fills this* |
| Video generation (context parallelism for video) | — | Ch 6 (§6.6) | *not currently covered — Kiely fills this* |
| Containerization / NIMs / dependency mgmt | Vol 1 MLOps | Ch 7 (§7.1) | Level 7 (cross-cutting) |
| Autoscaling / cold starts / scale-to-zero | Vol 2 Ops at Scale | Ch 7 (§7.2) | **Level 7 Topics 10–11** |
| Routing / load balancing / queueing | — | Ch 7 (§7.2.3) | Level 7 Topic 06–08 |
| Multi-cloud / GPU procurement / geo LB / reliability | Vol 2 Ops at Scale | Ch 7 (§7.3) | Level 7 Topic 13 (cost) + CAPACITY-PLANNING.md |
| Zero-downtime deploy / cost estimation / observability | Vol 1 MLOps + Vol 2 | Ch 7 (§7.4) | Level 7 Topic 05, 13 |
| Client-side latency / streaming / async inference | — | Ch 7 (§7.5) | Level 1 Topic 02 (streaming) |
| Training side (FSDP, 3D/5D parallelism, RLHF) | Vol 2 Distributed Training | *not covered* | Level 6 (full level) |
| Edge / TinyML / on-device (Apple Silicon depth) | Vol 2 Edge Intelligence | Ch 3 §3.5 (brief) | **Level 8 (full level)** |
| Compiler stack (MLIR / Inductor / StableHLO) | Vol 1 AI Acceleration | *not covered* | Level 9 + sibling `compiler-and-kernels/` |

### The rule of thumb

- **Reddi** for the *concept* (citation-grade framing of what a thing is and why it matters)
- **Kiely** for the *production view* (what an inference engineer actually does with the thing in 2026)
- **This repo** for the *lab* (build it, break it, measure it, ship a report)

If you can't explain a topic in textbook framing, defend it in practitioner framing, *and* implement it in project framing, you don't know it yet. The three views are different muscles.

### When to read which (by level)

| Level | Primary read before starting |
|---|---|
| Level 1 (Inference Serving) | **Kiely Ch 0–1** (the discipline + the prerequisites) — then build |
| Level 2 (CUDA/GPU) | Reddi *AI Acceleration* + Kiely Ch 3 (hardware) |
| Level 3 (Profiling) | Reddi *Benchmarking* + Kiely §4.5 |
| Level 4 (LLM Optimization) | **Kiely Ch 5** (techniques) + Reddi *Efficient AI / Model Optimizations* |
| Level 5 (Production Engines) | **Kiely Ch 4 + Ch 5.5** (engines + disaggregation) — strongest fit |
| Level 6 (Distributed Training) | Reddi Vol 2 *Distributed Training* (Kiely doesn't cover training) |
| Level 7 (ML Platform) | **Kiely Ch 7** (production) + Reddi Vol 2 *Ops at Scale* |
| Level 8 (Local / On-device) | Reddi *Edge Intelligence* + Kiely §3.5 (brief) — your repo is the depth here |
| Level 9 (Compiler Awareness) | Reddi *AI Acceleration* — Kiely doesn't go here |

---

## Who needs this

This curriculum is for the curiosity-driven learner who wants to *understand* how modern LLM systems actually work — by building each layer themselves.

What you'll explore:
- LLM inference infrastructure (vLLM / SGLang / TensorRT-LLM / llama.cpp / NVIDIA Dynamo / llm-d)
- Distributed training (FSDP2, Megatron-Core, torchtitan; RLHF / GRPO / DPO with verl / OpenRLHF / NeMo-RL)
- GPU performance engineering (CUDA, Triton, CUTLASS / CuTe DSL)
- Production-shaped platforms (KubeRay, vLLM Production Stack, KEDA, Envoy AI Gateway, OpenTelemetry GenAI)
- **Local / on-device intelligence** (Apple Silicon + MLX, M5 Neural Accelerators, Foundation Models framework, agentic local stacks)

You'll also leave with **awareness** of the compiler stack (MLIR / LLVM / `torch.compile` internals) — enough to decide whether to dive into that world separately.

> **On languages.** The curriculum is Python-first. Rust shows up in Level 7 because it's increasingly the production reality for gateways and routers. C++ remains essential for kernel work. Pick up what the project needs, when it needs it.

The point of this curriculum isn't a credential — it's that *you actually understand the substrate* the field runs on. The same depth that makes a system fast is what makes it interesting.

## The two paths the field is splitting into

```
Frontier scale (datacenter)               Local scale (on-device)
────────────────────────                  ─────────────────────────
vLLM / SGLang / TRT-LLM / Dynamo / llm-d  llama.cpp / MLX / vLLM-MLX / Ollama
H100 / B200 / GB300 NVL72                 M5 Pro/Max/Ultra (Neural Accelerators)
5D parallelism (TP+PP+DP+EP+CP), FSDP2    KV cache quantization (4-bit), MoE
NCCL 2.27 + SHARP + Communicator Shrink   Foundation Models framework + adapters
$/Mtok at fleet scale, FinOps for AI      Zero-latency, zero-API-cost agents
GRPO/PPO/DPO with verl/OpenRLHF/NeMo-RL   QLoRA + local DPO/GRPO personalization
KV-cache-aware routing, NIXL, LMCache     exo / Thunderbolt 5 distributed Macs
```

Most curricula teach only the left column. This one covers both, because *they're the same systems lessons* in different costumes — paged KV cache on H100 and on M5 Max are the same data structure with different bandwidth budgets, and the engineer who understands one understands the other.

## Mindset shift

```
python-pytorch/:     "How does the model work?"
systems-for-ml/:     "How do we make it fast, cheap, reliable, and observable at scale?"
```

The hard part is rarely "make it work." The hard part is: 50ms p99, 10× cheaper than GPT-4, never goes down, ships a new fine-tune every week, runs on the GPUs we can actually buy.

## Pedagogy — why the order matters

1. **Build it yourself before using the real thing.** You hand-roll an inference server in Week 1, then your *own* paged KV cache manager in Week 4 — so that when vLLM appears in Week 5, you've effectively reimplemented its core data structure and feel exactly which problem each flag solves.
2. **Understand the substrate before optimizing.** GPU mental model and profiling come *before* optimization tricks — you can't optimize what you can't measure.
3. **Evaluate everything you change.** Quantization without `lm-eval-harness` is guessing. Every optimization week ends with a quality check.
4. **Treat ML systems as distributed systems.** Most "ML infra" failures are networking, data pipeline, or queueing failures wearing an ML costume. Week 6 covers interconnects, tail latency, and failure injection alongside parallelism. Week 7 covers backpressure, multi-tenant fairness, and request hedging.
5. **Production-shape the capstone.** Week 7 is observability, autoscaling, routing, fairness, and cost — what platform teams actually own.
6. **Test what you build.** ML systems testing is its own subject — see [`TESTING.md`](TESTING.md) for the seven layers (schema, numerical correctness, tolerance equivalence, property-based, integration, load, quality regression). Each level's projects reference it.

## Projects & Deliverables

The weekly topic tables below are the **learning content**. The list here is the **artifact track** — what you actually ship as standalone repos with reports, the kind of work a curious reader can open and follow end-to-end.

Four projects across the nine weeks. Not one per week. Each project absorbs 2–3 weeks of learning, builds on the previous one, and ends with a written report. By the end you have one coherent system, not nine disconnected tutorials.

### How each project is structured

Every project follows the same loop. Skip any step and you're back to tutorial-grade work.

1. **Build the working version.** A baseline that runs end-to-end on the happy path.
2. **Break it on purpose.** Each project has an explicit *failure modes you induce* list. You don't wait for breakage — you cause it (long contexts, concurrency spikes, node failures, traffic skew, quantization drift).
3. **Measure the breakage.** Each project has a required-graphs list. Every graph answers one tradeoff question. Every graph is captioned **Setup → Observation → Insight**.
4. **Fix or characterize.** Either fix the failure and re-measure the delta, or document why this specific wall is the right reason a real system (vLLM, FSDP, Ray) exists. Both outcomes are valid; vague hand-waving is not.
5. **Ship the artifact.** Repo folder with code, `reports/` directory with graphs and writeup, and a README structured as a short systems paper (Problem → Architecture → Experiments → Findings → Tradeoffs).

### The four projects

| # | Project | Weeks | What it is | Real-world parallel |
|---|---------|-------|------------|---------------------|
| 1 | **`mini-serve` + `mini-vllm`** | 1–4 | Your hand-rolled FastAPI inference server, then your own paged KV cache + continuous batching layer dropped into it | The internal data structure of vLLM / SGLang |
| 2 | **`engine-bakeoff`** | 5 | Reproducible benchmark harness comparing vLLM, SGLang, TGI, TensorRT-LLM, and llama.cpp on identical workloads | The eval doc every inference team writes before picking an engine |
| 3 | **`mini-platform`** | 6–7 | Distributed training of a small model + production serving stack with router, autoscaler, observability, multi-tenant fairness, cost dashboard | A miniature of what platform teams at Scale / Anthropic / Databricks own |
| 4 | **`local-agent`** | 8 | Apple Silicon local-first agentic loop using MLX + llama.cpp, with QLoRA personalization and local DPO | The on-device backend of Cursor's local mode, or any private-AI startup |

Week 9 (compiler tour) is reading-shaped. Deliverable is a short writeup tracing one model through `torch.compile` / Inductor — no separate project repo needed.

### How they chain

```
mini-serve (W1)  ──►  mini-vllm KV cache (W4)  ──►  drops into mini-serve
                                                          │
                              engine-bakeoff (W5)  ◄──────┤  (your server is one of the entries)
                                                          │
              distributed-trainer (W6) ──► trained model  │
                                                          ▼
                                                   mini-platform (W7)
                                                   serves the W6 model via the
                                                   best W5 engine, with router +
                                                   autoscaler + observability
                                                   from this project

                              local-agent (W8) ──► parallel track, Apple Silicon
```

The chain is the point. By Project 3 you can say: *"Here's a platform. The model was trained with the W6 setup. It's served by vLLM, which I know cold because in W5 I benchmarked it against four others, and in W4 I implemented its core data structure myself."* That answer is the difference between `used vLLM` and `understands vLLM`.

### Per-project required graphs and break-it list

Each project section below names exactly what to break and what to graph. These aren't optional — they're the deliverable.

#### Project 1 — `mini-serve` + `mini-vllm`

**Break it on purpose:**
- Mixed sequence lengths in a static batch (watch GPU sit idle on padding)
- 100 concurrent requests with no batching (watch tail latency explode)
- 100K-token prompt with a contiguous KV cache (watch OOM or fragmentation)
- LRU vs FIFO vs sliding-window eviction under prefix-sharing workload (watch cache hit rate diverge)

**Required graphs:**
- G1: batch size vs throughput vs p99 latency — the classic systems tradeoff curve
- G2: request latency CDF (p50 / p95 / p99 / p999) at fixed concurrency
- G3: context length (1K → 128K) vs TTFT — prefill cost dominance, memory-bound behavior
- G4: KV cache hit rate vs latency under prefix-sharing workload
- G5: eviction policy comparison (LRU / FIFO / length-aware) — latency or cache miss rate

**Outcome artifact:** repo with `mini-serve/`, `mini-vllm/`, `reports/project1.md` containing all five graphs with Setup/Observation/Insight captions.

#### Project 2 — `engine-bakeoff`

**Break it on purpose:**
- Same model, same prompts, same hardware — but each engine with its default flags vs tuned flags
- Long-context workload that exposes KV cache strategy differences
- Prefix-heavy workload (chatbot-style) that exposes RadixAttention's edge in SGLang
- Constrained-memory scenario (smaller GPU than the model nominally needs)
- **Cross-substrate cost run:** llama.cpp on CPU vs the same model on a small GPU — find the workload regime where CPU actually wins on $/Mtok

**Required graphs:**
- G6: TTFT bar chart per engine, split by short prompt (128 tok) vs long prompt (4K tok)
- G7: throughput (tokens/sec) per engine on identical workload
- G8: GPU memory usage vs context length, per engine
- G9: cost per million tokens per engine + quantization combination — *include CPU-only llama.cpp as one of the rows*

**Outcome artifact:** `engine-bakeoff/` repo + `reports/bakeoff.md` written as a short systems-paper-style eval doc. This is exactly the document an inference team writes before adopting an engine — one of the two strongest artifacts in the curriculum.

#### Project 3 — `mini-platform`

**Break it on purpose:**
- Kill a node mid-training (watch what FSDP / Ray do)
- Inject a straggler node into distributed inference (watch p99 explode)
- Skew traffic 90% to one replica (watch noisy-neighbor effects across tenants)
- Push queue depth past autoscaler threshold (verify scale-up actually happens before SLA breach)
- Run a model regression — same engine, new checkpoint that's 5% worse on `lm-eval-harness` (verify the regression gate blocks deploy)
- Trigger a cold start during peak load (watch the first request after a scale-up event)
- Starve the GPU on the data side: undersized tokenizer pool, tiny prefetch (watch GPU util collapse while loss curves stall)
- Swap the scheduler from FCFS to a priority/SJF-batching variant on the same workload

**Required graphs:**
- G10: training throughput vs interconnect type (TCP vs RDMA vs NVLink — even simulated)
- G11: p99 latency timeline with node-failure event marker at t=30s
- G12: p99 latency vs traffic skew (% of requests routed to hottest replica)
- G13: queue depth vs latency — the curve your autoscaler reads, with a Little's-Law (L = λW) validation overlay
- G14: cost vs scaling strategy (vertical = bigger GPU vs horizontal = more replicas)
- G15: cold-vs-warm request latency, with the scale-up reaction window annotated — shows why the autoscaler must fire ahead of demand
- G16: scheduling policy comparison (FCFS vs priority vs SJF-style batching) on identical workload — measure p99, fairness, throughput
- G17: tokenization throughput vs training step throughput — the data-pipeline ceiling that bottlenecks the GPUs

**Outcome artifact:** `mini-platform/` with `training/`, `serving/`, `routing/`, `observability/`, `eval/`, plus `reports/platform.md` written as a systems paper. This is the capstone — it stitches every previous week together.

#### Project 4 — `local-agent`

**Break it on purpose:**
- Same agentic task in cloud-API mode vs local mode — measure latency, cost, and privacy posture side by side
- Push context past unified-memory limits on your specific Mac — characterize where local breaks
- Run the same model in MLX vs `llama.cpp` (Metal backend) vs PyTorch (MPS) — same prompts, three numbers
- Personalize via QLoRA on your own writing/code — measure quality drift on general tasks (the catastrophic-forgetting check)

**Required graphs:**
- G18: TTFT and tokens/sec on Apple Silicon: MLX vs llama.cpp vs PyTorch MPS
- G19: memory pressure curve as context grows on unified-memory hardware
- G20: quality before/after on-device QLoRA — task accuracy on personalized task vs general benchmarks

**Outcome artifact:** `local-agent/` repo + `reports/local.md`. A standalone, demonstrable system — the kind of project that's genuinely useful as a personal tool *and* showcases real systems thinking.

### What "good" looks like for the writeups

Each `reports/*.md` should follow this structure — borrowed from how systems papers are written, not how blog posts are:

```
1. Problem statement       (1 paragraph: what tradeoff this project explores)
2. System architecture     (one diagram, labeled)
3. Experiments             (what you ran, on what hardware, with what workload)
4. Key findings            (numbered, each one a single quantitative claim)
5. Tradeoffs               (latency vs throughput vs cost vs quality vs complexity)
6. What would change at 10× scale  (the question senior engineers always ask)
```

A finding looks like: *"Continuous batching reduced p99 latency by 38% under mixed-length workloads at 64 concurrent users; throughput improved 2.1× with no quality regression on `lm-eval-harness`."* Not: *"Continuous batching is faster."*

---

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

**Project:** This week is the first half of **Project 1 (`mini-serve`)**. By Friday you should have the baseline server running and the load-test harness producing G1 and G2 from the Project 1 graph list.

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

**Project:** This week closes out **Project 1**. Drop your paged KV cache into `mini-serve` to create `mini-vllm`, then run the full break-it list (long context, eviction policy comparison, prefix-sharing workload). Ship `reports/project1.md` with all five required graphs.

**Compute:** Mix of CPU (concepts) and Colab GPU (benchmarks).

---

### Week 5 — Production Inference Engines

The week the original curriculum was missing. Now that you've built a server and know how to optimize, run the real engines and benchmark them against your handmade one.

| # | Topic | What you build |
|---|-------|---------------|
| 01 | vllm-hello-world | Serve Qwen/Llama via vLLM, hit the OpenAI-compatible endpoint |
| 02 | vllm-features | Prefix caching, chunked prefill, tensor parallelism flags |
| 03 | sglang-and-radixattention | Same model on SGLang — when prefix sharing matters most |
| 04 | tensorrt-llm | TensorRT-LLM PyTorch flow + FP8/NVFP4; NIM as the customer-facing wrapper |
| 05 | llama-cpp-deep-dive | GGML internals, Metal/CUDA/CPU backends, when llama.cpp beats vLLM |
| 06 | mlc-llm | Compiler-based, cross-platform; WebGPU and heterogeneous hardware |
| 07 | engine-bake-off | Same model + same load test across all five — write up the differences |
| 08 | disaggregated-inference | Prefill/decode split, NIXL transport — standard, not novel in 2026 |
| 09 | dynamo-and-llmd | NVIDIA Dynamo + llm-d (CNCF Sandbox 2026) — the production frontier |
| 10 | multi-lora-serving | Train tiny LoRAs, hot-swap them via vLLM's multi-LoRA endpoint |
| 11 | offline-batch-inference | vLLM offline mode for million-doc scoring jobs |
| 12 | speculative-decoding-in-prod | EAGLE-3 in vLLM V1, measure end-to-end gain |
| 15 | triton-inference-server | Framework-agnostic outer wrapper — host vLLM + ORT + TRT in one server; ensemble graphs; the foundation of NVIDIA NIM |

**Outcome:** You can pick the right engine for a workload, tune its flags, and serve dozens of fine-tunes off one base model. This week is where the inference-engineering field's center of gravity sits.

**Project:** This week *is* **Project 2 (`engine-bakeoff`)**. Use the load harness from Project 1 to drive identical workloads against vLLM, SGLang, TGI, TensorRT-LLM, and llama.cpp. Ship `reports/bakeoff.md` with G6–G9 and a written recommendation: *"For workload X on hardware Y, use engine Z because…"*

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

**Project:** This week is the first half of **Project 3 (`mini-platform`)**. Train a small model with FSDP, run the failure-injection list, and ship G10 (training throughput vs interconnect) and G11 (p99 timeline with node-failure marker). The trained checkpoint becomes the model your Week 7 platform serves.

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
| 08 | backpressure-and-queueing | Little's Law (L = λW) — derive it, then *validate it* against your own system metrics; when to shed load vs queue vs hedge |
| 09 | scheduling-policies | FCFS vs priority vs SJF-style batching heuristics — measure how the choice shapes p99 |
| 10 | autoscaling | Scale replicas on `vllm:num_requests_waiting` and queue latency |
| 11 | cold-start-and-warmup | Model load time, warm vs cold request latency, why your scale-up must fire *before* the SLA breach |
| 12 | cost-economics | $/Mtok per engine + quant combo — when to quantize vs scale horizontally |
| 13 | safety-and-abuse | Rate limiting, prompt-injection at the infra layer, output filtering |
| 14 | mini-rlxf | End-to-end: SFT → reward model → RLHF pipeline orchestration |

**Outcome:** You've built a tiny version of Scale's RLXF platform — and unlike most tutorials, yours has dashboards, autoscaling, and a cost model.

**Project:** This week closes out **Project 3 (`mini-platform`)**. Stitch the W6 trained model + the best-from-bakeoff engine + your router + autoscaler + Prometheus dashboards into one running system. Run the remaining break-it list (traffic skew, queue-depth threshold, regression gate, cold-start during peak, scheduler swap, data-pipeline starvation) and ship G12–G17 plus the final `reports/platform.md` written as a systems paper.

**Compute:** All CPU. This is system design + orchestration code. Optionally point it at the Week 5 engines for a real end-to-end demo.

---

### Week 8 — Local & On-Device Intelligence (Apple Silicon, MLX, Personalization)

The other half of the field. Datacenter inference is one side; local-first AI (agentic IDEs, private-AI tooling, on-device assistants) is the other. Apple Silicon's unified memory architecture changed what's possible on a laptop — a maxed-out M-series machine now runs 70B-class models with no GPU rack involved.

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

**Project:** This week *is* **Project 4 (`local-agent`)**. Build the agentic loop, run the cloud-vs-local comparison, characterize where unified memory breaks, and ship G18–G20 with `reports/local.md`. A standalone artifact you can actually use day-to-day, with the systems lessons made explicit.

**Compute:** Your own Mac (M-series strongly recommended). No cloud spend.

---

### Week 9 — Compiler Stack Awareness (High-Level Tour)

> **Scope note:** The point of *this* curriculum is to make you fluent **end-to-end** — train, optimize, serve, scale, ship. Once you have that, the compiler stack is a separate, deeper specialization you can choose to dive into if it interests you.
>
> So this week is **a high-level tour, not a compiler course.** The goal is awareness: when `torch.compile` does something surprising, when you see "MLIR" or "StableHLO" mentioned in a paper or repo, you know what's actually happening underneath. If you fall in love with it, that's a *whole separate track* — not something we cram into one week here.

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

## What you'll actually be able to do

After Weeks 1–7:

- Build a multi-engine LLM serving stack and benchmark vLLM, SGLang, TensorRT-LLM, llama.cpp, and your own implementation against each other on reproducible workloads.
- Implement a paged KV cache manager from scratch — pages, block table, eviction policies — and reason through why it beats a contiguous cache on long contexts and prefix-sharing workloads.
- Profile a transformer inference path with Nsight Systems, Nsight Compute, and PyTorch Profiler; place kernels on a roofline; identify whether you're compute-bound, memory-bound, or latency-bound.
- Train a model with FSDP2 + DeviceMesh on multi-GPU; sketch when each axis of 5D parallelism (TP/PP/DP/EP/CP) is needed; characterize how interconnect choice (TCP / RDMA / NVLink) shapes throughput.
- Stand up a mini production platform — KV-cache-aware routing, KEDA autoscaling on `vllm:num_requests_waiting`, OpenTelemetry GenAI observability, weighted-fair-queueing for multi-tenant fairness, per-(engine×quant×hardware) cost dashboards.
- Inject real failures (node deaths, stragglers, traffic skew, queue-threshold breaches, cold starts during peak load, regression-gate triggers) and recover correctly.

After Week 8 (local / on-device):

- Build a local-first agentic system on Apple Silicon using MLX and llama.cpp with sub-100ms TTFT and zero API cost.
- Fine-tune a 7B model on-device using QLoRA + local DPO/GRPO and verify quality with `lm-eval-harness`.
- Reason about CPU SIMD / AMX / SME2 paths and characterize the workload regimes where llama.cpp on CPU beats a small GPU.
- Use Apple's Foundation Models framework with a custom adapter — the on-device 3B path.

After Week 9 (compiler awareness):

- Read a `torch.compile` / Inductor trace and roughly follow it down through Triton → PTX → SASS.
- Hold a real conversation about MLIR, StableHLO, IREE, and the difference between GPU / TPU / Groq / Cerebras compiler stacks.
- Decide, with eyes open, whether to dive into the compiler-engineering world separately.

These are skills the field actually exercises every day. The point isn't to collect them — it's to *understand the substrate* well enough that future tools and architectures aren't mysterious. New engines come and go; paged attention, continuous batching, prefix caching, and the bandwidth hierarchy don't.

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
