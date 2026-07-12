# Reddi *Machine Learning Systems* — Grounded Index

Verified July 2026 against the live book source ([harvard-edge/cs249r_book](https://github.com/harvard-edge/cs249r_book), `dev` branch) and the official instructor curriculum at [mlsysbook.ai/instructors](https://mlsysbook.ai/instructors/). Use this file instead of guessing chapter names — the root README's mapping table predates it.

**Versions at time of writing:**

| Volume | Title | Version | Print |
|---|---|---|---|
| Vol I | *Introduction to Machine Learning Systems* | v0.7.1 | MIT Press **2026** |
| Vol II | *Machine Learning Systems at Scale* | v0.2.1 | MIT Press **2027** |

Vol II is an early draft (v0.2.x) — chapters will churn. Re-verify this index every few months; the site blocks scrapers, but the repo's `book/quarto/contents/vol{1,2}/` folders and `instructors/*.qmd` files are always fetchable.

**Local PDFs:** `Machine-Learning-Systems-Vol1-Reddi.pdf`, `Machine-Learning-Systems-Vol2-Reddi.pdf` (this folder). Fresh copies: [Vol 1](https://mlsysbook.ai/vol1/assets/downloads/Machine-Learning-Systems-Vol1.pdf) · [Vol 2](https://mlsysbook.ai/vol2/assets/downloads/Machine-Learning-Systems-Vol2.pdf).

---

## The ecosystem (more than a book)

The project ships four integrated pillars plus supporting material:

| Pillar | What it is | Link |
|---|---|---|
| **Read** | The two-volume textbook | [mlsysbook.ai](https://mlsysbook.ai) |
| **Build** | TinyTorch — 20 modules, build a framework from scratch | [mlsysbook.ai/tinytorch](https://mlsysbook.ai/tinytorch/) |
| **Explore** | Interactive Marimo labs (simulated hardware, tradeoff discovery) | [mlsysbook.ai/labs](https://mlsysbook.ai/labs/) |
| **Deploy** | Hardware kits (Arduino / Seeed / RPi+Coral) | [mlsysbook.ai/kits](https://mlsysbook.ai/kits/) |

Supporting:

- **Instructor syllabi** — the official 16-week week-by-week curricula for each volume: [Foundations](https://mlsysbook.ai/instructors/foundations-syllabus.html) · [Scale](https://mlsysbook.ai/instructors/scale-syllabus.html) · [Course map](https://mlsysbook.ai/instructors/course-map.html)
- **Slides** — Beamer decks per chapter with speaker notes: [vol1](https://mlsysbook.ai/slides/vol1.html) · [vol2](https://mlsysbook.ai/slides/vol2.html)
- **StaffML** — interview-style drills, L2 vocabulary through L6+ architecture chains and mock interviews: [mlsysbook.ai/staffml](https://mlsysbook.ai/staffml/)
- **MLSys·im** — analytical modeling engine behind the labs (memory bottlenecks, network saturation, scheduling at scales you can't rent): [mlsysbook.ai/mlsysim](https://mlsysbook.ai/mlsysim/)

Their pedagogical loop — *Theory → Build → Simulation → Reality* — is the same loop as this repo's *read → build → break → measure*, so the pillars slot in per level rather than as a separate track.

**The Iron Law** is the book's unifying frame; every optimization in both volumes maps to a term of it:

```
T  ≈  D_vol / BW  +  O / (R_peak · η)  +  L_lat
      ─────────     ────────────────     ─────
      data volume   compute term         latency
      over bandwidth (peak · efficiency) overhead
```

Quantization attacks `D_vol`; tensor cores raise `R_peak`; pipeline bubbles and GPU starvation lower `η`; kernel-launch and network overheads are `L_lat`. Useful shared vocabulary for the blog posts.

---

## Volume I — *Introduction to Machine Learning Systems* (1–8 GPUs)

Four parts: Foundations → Build → Optimize → Deploy. 16-week course pace, one chapter/week.

| Wk | Part | Chapter | Companion lab | TinyTorch |
|---|---|---|---|---|
| 1 | I Foundations | [Introduction](https://mlsysbook.ai/vol1/introduction/introduction.html) | Lab 00: Architect's Portal | 01 Tensor |
| 2 | I | [ML Systems](https://mlsysbook.ai/vol1/ml_systems/ml_systems.html) | Lab 01: The Magnitude Gap | 01 cont. |
| 3 | I | [ML Workflow](https://mlsysbook.ai/vol1/ml_workflow/ml_workflow.html) | Lab 02: Workflow Pipeline | 02 Activations |
| 4 | I | [Data Engineering](https://mlsysbook.ai/vol1/data_engineering/data_engineering.html) | Lab 03: The Data Pipeline | 02 cont. |
| 5 | II Build | [Neural Computation](https://mlsysbook.ai/vol1/nn_computation/nn_computation.html) | Lab 04: Computation Graph | 03 Layers |
| 6 | II | [NN Architectures](https://mlsysbook.ai/vol1/nn_architectures/nn_architectures.html) | Lab 05: Architecture Tradeoffs | 04 Losses |
| 7 | II | [ML Frameworks](https://mlsysbook.ai/vol1/frameworks/frameworks.html) | Lab 06: The Dispatch Tax | 05 DataLoader |
| 8 | II | [Training](https://mlsysbook.ai/vol1/training/training.html) | Lab 07: The Training Loop | 06 Autograd |
| 9 | III Optimize | [Data Selection](https://mlsysbook.ai/vol1/data_selection/data_selection.html) | Lab 08: Data Quality | 07 Optimizers |
| 10 | III | [Model Compression](https://mlsysbook.ai/vol1/optimizations/model_compression.html) | Lab 09: Data Selection Paradox | 08 Training |
| 11 | III | [HW Acceleration](https://mlsysbook.ai/vol1/hw_acceleration/hw_acceleration.html) | Lab 10: Compression Paradox | 08 cont. |
| 12 | III | [Benchmarking](https://mlsysbook.ai/vol1/benchmarking/benchmarking.html) | Lab 11: Hardware Roofline | — |
| 13 | IV Deploy | [Model Serving](https://mlsysbook.ai/vol1/model_serving/model_serving.html) | Lab 12: The Benchmarking Trap | — |
| 14 | IV | [ML Operations](https://mlsysbook.ai/vol1/ml_ops/ml_ops.html) | Lab 13: The Tail Latency Trap | — |
| 15 | IV | [Responsible Engineering](https://mlsysbook.ai/vol1/responsible_engr/responsible_engr.html) | Lab 14: Silent Degradation | — |
| 16 | IV | [Conclusion](https://mlsysbook.ai/vol1/conclusion/conclusion.html) | Lab 15: No Free Fairness | Capstone (AI Olympics) |

Backmatter appendices worth knowing: **A** D·A·M taxonomy (Data·Algorithm·Machine diagnostic framework), **E** System Assumptions (napkin-math reference), **F** Glossary.

## Volume II — *Machine Learning Systems at Scale* (clusters → fleets)

Four parts: The Fleet → Distributed ML → Deployment at Scale → The Responsible Fleet. No TinyTorch column (design-focused); modules 09–20 run as an optional advanced track alongside.

| Wk | Part | Chapter | Companion lab |
|---|---|---|---|
| 1 | I Fleet | [Introduction to Scale](https://mlsysbook.ai/vol2/introduction/introduction.html) | Lab 01: The Scale Wall |
| 2 | I | [Compute Infrastructure](https://mlsysbook.ai/vol2/compute_infrastructure/compute_infrastructure.html) | Lab 02: The Interconnect Wall |
| 3 | I | [Network Fabrics](https://mlsysbook.ai/vol2/network_fabrics/network_fabrics.html) | Lab 03: Communication Topologies |
| 4 | I | [Data Storage](https://mlsysbook.ai/vol2/data_storage/data_storage.html) | Lab 04: The Storage Hierarchy |
| 5 | II Distributed | [Distributed Training](https://mlsysbook.ai/vol2/distributed_training/distributed_training.html) | Lab 05: The Parallelism Puzzle |
| 6 | II | [Collective Communication](https://mlsysbook.ai/vol2/collective_communication/collective_communication.html) | Lab 06: AllReduce Physics |
| 7 | II | [Fault Tolerance](https://mlsysbook.ai/vol2/fault_tolerance/fault_tolerance.html) | Lab 07: The Scheduling Trap |
| 8 | II | [Fleet Orchestration](https://mlsysbook.ai/vol2/fleet_orchestration/fleet_orchestration.html) | Lab 08: The Inference Economy |
| 9 | III Deploy | [Performance Engineering](https://mlsysbook.ai/vol2/performance_engineering/performance_engineering.html) | Lab 09: The Optimization Trap |
| 10 | III | [Inference at Scale](https://mlsysbook.ai/vol2/inference/inference.html) | Lab 10: The KV-Cache Memory Wall |
| 11 | III | [Edge Intelligence](https://mlsysbook.ai/vol2/edge_intelligence/edge_intelligence.html) | Lab 11: Edge Thermodynamics |
| 12 | III | [Ops at Scale](https://mlsysbook.ai/vol2/ops_scale/ops_scale.html) | Lab 12: The Silent Fleet |
| 13 | IV Responsible | [Security & Privacy](https://mlsysbook.ai/vol2/security_privacy/security_privacy.html) | Lab 13: The Price of Privacy |
| 14 | IV | [Robust AI](https://mlsysbook.ai/vol2/robust_ai/robust_ai.html) | Lab 14: The Robustness Budget |
| 15 | IV | [Sustainable AI](https://mlsysbook.ai/vol2/sustainable_ai/sustainable_ai.html) + [Responsible AI](https://mlsysbook.ai/vol2/responsible_ai/responsible_ai.html) | Lab 15: The Fairness Budget |
| 16 | — | [Conclusion](https://mlsysbook.ai/vol2/conclusion/conclusion.html) | Lab 16: Fleet Synthesis (capstone) |

Backmatter appendices: **B** C³ taxonomy (fleet diagnostic framework), **D** Communication Foundations, **E** Reliability Foundations, **F** **Inference Foundations — queuing theory + KV cache** (directly relevant to Levels 1, 4, 7), **G** System Assumptions (napkin math).

## TinyTorch — all 20 modules

| # | Module | # | Module |
|---|---|---|---|
| 01 | Tensor | 11 | Embeddings |
| 02 | Activations | 12 | Attention |
| 03 | Layers | 13 | Transformers |
| 04 | Losses | 14 | Profiling |
| 05 | DataLoader | 15 | Quantization |
| 06 | Autograd | 16 | Compression |
| 07 | Optimizers | 17 | Acceleration |
| 08 | Training | 18 | **Memoization (KV caching)** |
| 09 | Convolutions | 19 | Benchmarking |
| 10 | Tokenization | 20 | Capstone |

Modules 01–08 duplicate `python-pytorch/` (skip). Modules 10–19 are the interesting overlap: they track this repo's Level 2–4 build-it-yourself arc almost one-to-one.

---

## Level-by-level: exact chapters, labs, and papers

The definitive per-level reading list. "Case study" papers come from the book's own instructor pairings — they chose well.

| Repo level | Reddi chapters (exact) | Labs worth running | Case-study paper |
|---|---|---|---|
| **L1 Inference Serving** | Vol 1 *Model Serving*; Vol 2 App. F (queuing theory) | V1 Lab 12 (Benchmarking Trap), Lab 13 (Tail Latency Trap) | Yu et al., **Orca** (OSDI '22) |
| **L2 CUDA & GPU** | Vol 1 *HW Acceleration* | V1 Lab 11 (Hardware Roofline); TinyTorch 17 | Jouppi et al., **TPU** (ISCA '17) |
| **L3 Profiling** | Vol 1 *Benchmarking*; Vol 2 *Performance Engineering* | V2 Lab 09 (Optimization Trap); TinyTorch 14, 19 | — |
| **L4 LLM Optimization** | Vol 1 *Model Compression*; Vol 2 *Inference at Scale* | V1 Lab 10 (Compression Paradox), V2 Lab 10 (KV-Cache Memory Wall); TinyTorch 15, 16, **18** | Dettmers et al., **LLM.int8()** (NeurIPS '22) |
| **L5 Production Engines** | Vol 2 *Inference at Scale* (the PagedAttention/continuous-batching chapter) | V2 Lab 08 (Inference Economy) | Kwon et al., **PagedAttention** (SOSP '23) |
| **L6 Distributed Training** | Vol 2 *Compute Infrastructure*, *Network Fabrics*, *Data Storage*, *Distributed Training*, *Collective Communication*, *Fault Tolerance* — the book's core strength | V2 Labs 02–07 (Interconnect Wall, Topologies, Storage Hierarchy, Parallelism Puzzle, AllReduce Physics, Scheduling Trap) | **Megatron-LM** (Shoeybi '20); Narayanan '21; Jeon, **multi-tenant GPU clusters** (ATC '19) |
| **L7 ML Platform** | Vol 1 *ML Operations*; Vol 2 *Fleet Orchestration*, *Ops at Scale* | V2 Lab 12 (Silent Fleet) | Sculley, **Hidden Technical Debt** (NeurIPS '15); Zhao, **GPU cluster failures** (ATC '24) |
| **L8 Local / On-Device** | Vol 2 *Edge Intelligence* | V2 Lab 11 (Edge Thermodynamics); optionally the [TinyML syllabus](https://mlsysbook.ai/instructors/tinyml-syllabus.html) + kits | — |
| **L9 Compiler Awareness** | Vol 1 *ML Frameworks* (dispatch, eager vs graph, fusion) + *HW Acceleration* | V1 Lab 06 (The Dispatch Tax) | Chen et al., **TVM** (OSDI '18) |
| **L10 Design Capstone** | Vol 2 *Conclusion* + both capstone specs (AI Olympics, **Fleet Synthesis**) | V2 Lab 16 (Fleet Synthesis) — a frontier-model infra design doc with quantitative justification, same shape as the Level 10 prompts | [**StaffML**](https://mlsysbook.ai/staffml/) — L5–L6+ architecture chains and mock-interview prompts |

Corrections to the root README's older fuzzy mapping:

- "Vol 1 Serving" for inference engines → the engine-level material (PagedAttention, continuous batching, KV-cache wall) actually lives in **Vol 2 *Inference at Scale***; Vol 1 *Model Serving* covers single-node batching strategies and SLA design.
- "Vol 2 Distributed Inference" → the chapter is titled ***Inference at Scale***.
- "Vol 2 Ops at Scale" → correct, and *Fleet Orchestration* (Slurm/K8s, multi-tenant scheduling, fragmentation cost) is a separate chapter that belongs with it for Level 7.
- Roofline appears twice: Vol 1 *HW Acceleration* introduces it (single node), Vol 2 *Performance Engineering* extends it multi-node with strong/weak scaling.

## What Reddi will not give you

Field-current 2026 specifics — vLLM V1 flags, SGLang RadixAttention, Dynamo/llm-d disaggregation, EAGLE-3, GGUF/NVFP4, MLX — stay with Kiely and the engine source. Reddi is the concepts-and-physics layer: Iron Law, C³ taxonomy, queuing theory, napkin math. Vol 2's early-draft status (v0.2.1) means the inference chapter in particular may lag the frontier by a year; trust Kiely where they disagree on current practice.
