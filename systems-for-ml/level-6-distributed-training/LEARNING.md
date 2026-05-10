# Level 6 — Learning Path

Distributed training reverses the direction of inference: same model, opposite mode. The systems lessons echo Levels 1–5 — throughput vs latency, memory hierarchy, communication-vs-compute overlap — but the constraints flip. Inference cares about per-request tail; training cares about all-rank step time. 17 topics organized into four sub-arcs:

```
Plumbing                   (00, 00b, 01)   collectives, transports, fabrics
Single-axis parallelism    (02-08)         DDP, FSDP, TP, PP, EP, CP
Composition + production   (09-10)         5D mesh, torchtitan / Megatron
Reliability + bridge       (11-15)         stragglers, failure, async ckpt, Ray, RL
```

## The path

| Folder | Time | What you walk away with |
|---|---|---|
| `00-collectives-and-nccl/` | 2-3h | The four collectives, ring vs tree, NCCL 2.27 features, hang debugging |
| `00b-rdma-gpudirect-nixl/` | 1-2h | Transport hierarchy, GPUDirect RDMA, what NIXL adds for inference + checkpoint streaming |
| `01-interconnects/` | 1h | NVLink5 / NVL72 / IB-XDR / Ultra Ethernet, rail-aware topology |
| `02-data-parallel-from-scratch/` | 1-2h | DDP, gradient bucketing, `nccl:all_reduce` overlap |
| `03-data-loading-and-tokenization/` | 2h | Mosaic StreamingDataset, sequence packing, the dataloader ceiling (G17) |
| `04-fsdp2-and-dtensor/` | 2-3h | `fully_shard`, DeviceMesh, HSDP, mixed precision, sharded DCP |
| `05-tensor-parallelism/` | 1-2h | Column/row parallel pair, sequence parallelism, async-TP |
| `06-pipeline-parallelism/` | 2-3h | Bubble math, 1F1B → ZB-V → DualPipe, hand-rolled 2-stage |
| `07-expert-parallelism/` | 1-2h | All-to-all routing, no-token-dropping, DeepSeek-V3 frontier MoE |
| `08-context-parallelism/` | 1-2h | Ring vs Striped attention, Dynamic CP (Megatron-Core Jan 2026) |
| `09-5d-parallelism-composition/` | 2h | The decision tree, worked examples for 7B/70B/405B/MoE/long-context |
| `10-torchtitan-or-megatron/` | 4-6h | A real training run; saved checkpoint becomes Level 7's artifact |
| `11-tail-latency-and-stragglers/` | 1-2h | Straggler probability math, p99 step time, Comm Shrink for chronic stragglers (G11) |
| `12-failure-injection/` | 2-3h | Elastic launch + NCCL Comm Shrink + DCP recovery flow |
| `13-checkpointing-async/` | 1-2h | Two-stage offload, training-pause time, peer replication |
| `14-ray-and-multi-node/` | 1-2h | Ray Train, KubeRay, multi-node orchestration |
| `15-rl-post-training-bridge/` | 1h | Trainer + rollout engine architecture; bridge to Level 7's mini-rlxf |

Total: ~25-40 hours of focused work, depending on how deep you go on Topic 10.

## What's new in 2026 (deltas vs 2024-2025 content)

The research backing this level surfaced several status changes. Key items in case you saw older material:

- **FSDP2 replaced FSDP1**. `fully_shard` API on `DTensor` is the standard. `FullyShardedDataParallel(...)` tutorials are out of date.
- **3D parallelism became 5D parallelism**. TP + PP + DP + EP + CP is the standard vocabulary across NVIDIA, Meta, Mistral, DeepSeek, Anthropic.
- **torchtitan** (PyTorch official, ICLR 2025) is the canonical PyTorch-native distributed training stack.
- **Megatron-Core / Megatron-Bridge** are the canonical NVIDIA-flavored stack. Bridge ships HF↔Megatron checkpoint conversion.
- **DeepSpeed is legacy** for new pretraining at frontier labs. Niches: CPU/NVMe offload, DeepSpeed-Inference.
- **Zero Bubble Pipeline (ZB-V)** is the practical PP default in 2026 — same memory as 1F1B, near-zero bubble.
- **Dynamic Context Parallelism** (Megatron-Core, Jan 2026) — picks `cp_size` per microbatch.
- **NCCL 2.27+** added SHARP for both NVLink+IB and Communicator Shrink for elastic recovery.
- **NIXL** (NVIDIA Inference Xfer Library, GTC 2025) — KV-cache transfer in disaggregated inference and weight-streaming in RLHF.
- **Async DCP** is the default checkpointing API; sync `torch.save` is deprecated for production runs.
- **Goodput** (Google) is replacing raw MFU as the SLO. Frontier-scale training is judged on goodput now.
- **DeepSeek-V3 DualPipe + auxiliary-loss-free MoE balancing** are the most-cited 2026 systems papers.
- **verl (HybridFlow)** + vLLM/SGLang rollout backend is the production RLHF pattern in 2026.

## What hardware you need

- **Two GPUs minimum** for almost everything. Colab Pro 2×T4, RunPod 2×A10, RunPod 2×A100. ~$2/hr cloud rentals are sufficient.
- **Hopper (H100) ideal** for Topic 10's full torchtitan run with FP8.
- **Multi-node** is required only conceptually for Topics 06 (PP), 07 (EP across nodes), 12 (true elastic recovery). Read those, run the local-only versions of their demos.
- **8 GPU box** unlocks meaningful 5D mesh experiments (Topic 09). Otherwise the 2-D mesh demo is the limit.

## Each topic folder

Same shape as Levels 1-5:

- `CONCEPTS.md` — theory + 2026 state
- One or more code files (`.py`, `.sh`, `.toml`) demonstrating the topic
- `README.md` — quickstart, expected output, things to try

## Project 3 begins here

`mini-platform` (Levels 6–7). Level 6 ships the training half:

```
mini-platform/
├── training/
│   ├── torchtitan-config.toml         (from Topic 10)
│   ├── train.py
│   ├── data/                          (Topic 03 — Mosaic StreamingDataset shards)
│   └── checkpoints/                   (Topic 13 — async DCP output)
└── reports/
    └── training.md                    (Level 6 deliverable)
```

The trained checkpoint is the artifact Level 7 will serve.

Required graphs:
- **G10** — training throughput vs interconnect type (Topic 01)
- **G11** — p99 step time as a function of straggler severity (Topic 11)
- **G12** — failure event annotated on the step-time timeline; time-to-recovery measured (Topic 12)
- **G17** — dataloader throughput vs training throughput; identify the wall (Topic 03)

## After this level

Level 7 (`mini-platform`) loads this trained checkpoint, serves it, and extends the system with a small RLHF rollout loop. Level 8 covers on-device training (QLoRA, GGUF training) — the single-machine analog of what you do at scale here.
