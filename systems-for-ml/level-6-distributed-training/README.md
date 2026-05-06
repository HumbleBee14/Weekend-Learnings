# Level 6 — Distributed Training & Networking

> Outer reference: [`systems-for-ml/README.md`](../README.md) · Project: first half of **Project 3 — `mini-platform`**

## Week goal

Train a model across multiple GPUs *correctly*, then break it on purpose to learn what production teams actually deal with. By Friday you should be able to:

- Train a small model with FSDP2 + DTensor on a multi-GPU host (Colab Pro 2×T4, RunPod 2×A10, or similar).
- Sketch the difference between TP, PP, DP, EP, and CP — the **5D parallelism** vocabulary the field uses in 2026.
- Run a Pipeline Parallelism schedule (1F1B at minimum, ZB-V if time permits) and explain the bubble.
- Inject a node failure mid-training and recover via async DCP + NCCL Communicator Shrink.
- Measure data-pipeline throughput vs training step throughput — the bottleneck most curricula skip.

The trained checkpoint from this week becomes the model your Level 7 `mini-platform` serves.

## Where this fits

- **Comes after:** Levels 1–5. You understand inference end-to-end. Now reverse direction — training is the same model in the opposite mode, and the systems lessons echo: throughput vs latency, memory hierarchy, communication-vs-compute overlap.
- **Comes before:** Level 7 (the trained model lives there). Level 8 (on-device training, QLoRA — single-machine analog of what you do here at scale).
- **Project this feeds:** **Project 3** first half. Ships **G10–G11** (training throughput vs interconnect; p99 timeline with node-failure marker), plus **G17** (data-pipeline throughput).

## 2026 reality check — what changed

- **FSDP1 is deprecated.** FSDP2's `fully_shard` API on `DTensor` is the current standard. Pre-2024 tutorials using `FullyShardedDataParallel(...)` are out of date.
- **3D parallelism → 5D parallelism.** TP + PP + DP + EP (expert) + CP (context) is now the standard vocabulary across the field (NVIDIA, Meta FAIR, Mistral, and others).
- **torchtitan** (PyTorch official, ICLR 2025) is the canonical PyTorch-native distributed training stack. Megatron-Core/Megatron-Bridge is the canonical NVIDIA-flavored stack.
- **DeepSpeed is not gone but is legacy** for new pretraining at frontier labs. Niches: CPU/NVMe offload, inference (DeepSpeed-Inference), DeepSpeed-Chat (mostly displaced by TRL/verl/OpenRLHF).
- **Zero Bubble Pipeline (ZB-V)** is the practical PP default in 2026 — same memory as 1F1B, near-zero bubble.
- **Dynamic Context Parallelism** (Megatron-Core, Jan 2026) — picks `cp_size` per microbatch for variable-length workloads.
- **NCCL 2.27+** added SHARP for both NVLink+IB and **Communicator Shrink** for elastic recovery.
- **NIXL** (NVIDIA Inference Xfer Library, GTC 2025) — async KV-cache transfer, used by Dynamo for disaggregated inference. Included here because the same primitives power training-checkpoint streaming.
- **Goodput** (Google's term) is replacing raw MFU as the SLO of choice — it accounts for failures, restarts, and stragglers. Frontier-scale training is judged on goodput now.

## Topic-by-topic deep dive

| # | Topic | What you build |
|---|-------|---------------|
| 00 | collectives-and-nccl | All-reduce / all-gather / reduce-scatter; ring vs tree algorithm; NCCL 2.27+ features; debugging hangs |
| 00b | rdma-gpudirect-nixl | The transport layer underneath NCCL and disaggregated KV transfer — RDMA, GPUDirect, NIXL |
| 01 | interconnects | NVLink 5 / NVL72 / IB XDR / Ultra Ethernet — the 2026 fabric |
| 02 | data-parallel-from-scratch | DDP — what `loss.backward()` actually communicates |
| 03 | data-loading-and-tokenization | Mosaic StreamingDataset, sequence packing, the data-pipeline ceiling |
| 04 | fsdp2-and-dtensor | `fully_shard`, DeviceMesh, the per-parameter sharding model |
| 05 | tensor-parallelism | Megatron-style intra-layer split |
| 06 | pipeline-parallelism | 1F1B → interleaved → ZeroBubble |
| 07 | expert-parallelism | EP for MoE — DeepSeek-V3 style |
| 08 | context-parallelism | CP + Ring/Striped Attention; Dynamic-CP for variable lengths |
| 09 | 5d-parallelism-composition | When to reach for each axis |
| 10 | torchtitan-or-megatron | Pick one, train a small model end-to-end |
| 11 | tail-latency-and-stragglers | p99 step-time blowup; mitigation |
| 12 | failure-injection | Kill a node mid-step, recover via Comm Shrink |
| 13 | checkpointing-async | DCP async save, peer replication, Goodput math |
| 14 | ray-and-multi-node | Job orchestration, world-size management |
| 15 | rl-post-training-bridge | Brief — vLLM/SGLang as rollout backend; sets up Level 7 |

### 00 — `collectives-and-nccl`

**The four collectives that dominate.**
- **All-reduce** — every rank contributes a tensor; every rank gets the sum. Used in DP gradient sync.
- **All-gather** — every rank contributes; every rank gets the concatenation. Used in FSDP forward (gather sharded weights).
- **Reduce-scatter** — every rank contributes; the sum is split, each rank gets one shard. Used in FSDP backward (gradient reduction).
- **All-to-all** — every rank sends a different tensor to every other rank. Used in expert parallelism (route tokens to expert-owning ranks).

**Implementation matters.**
- **Ring algorithm** — bandwidth-optimal, latency proportional to N (number of ranks). Default for large messages.
- **Tree algorithm** — latency proportional to log(N), bandwidth-suboptimal. Default for small messages.
- **NCCL 2.27 SHARP** — switch-offloaded reductions (NVLink + IB switches do the math). Frees SMs and scales past 1024 GPUs.

**NCCL vs OpenMPI — what you'll see in the wild.** OpenMPI is the HPC-world equivalent of NCCL: same collective primitives (MPI_Allreduce, MPI_Allgather, MPI_Alltoall), much older (1990s heritage), CPU-first with GPU support added later. Its place in 2026:
- **NCCL dominates LLM work** — every modern training/inference framework (PyTorch DDP/FSDP, DeepSpeed, Megatron, vLLM TP) uses NCCL on NVIDIA hardware.
- **OpenMPI shows up in HPC-adjacent ML** — molecular dynamics + ML, scientific computing pipelines, older training codebases that pre-date NCCL maturity. Some non-NVIDIA systems still default to MPI.
- **Hybrid setups exist** — MPI for process spawning + NCCL for GPU collectives. `mpirun -np 8 python train.py` launches 8 processes, each then initializes a NCCL communicator. This is a common pattern in labs that came from HPC.
- **What to remember**: same collective vocabulary (all-reduce, all-gather, etc.). If you see `MPI_Allreduce` in code, mentally substitute "this is what NCCL would do for a GPU tensor." The algorithms (ring, tree) are the same; the implementation differs.

**NCCL 2.27 Communicator Shrink.** Drop a failed/unwanted GPU from a comm dynamically. "Default" mode for planned reconfig; "Error" mode for fault recovery. This is the foundation under elastic training in 2026.

**Ring all-reduce — the algorithm.** Why ring is bandwidth-optimal for large messages: with N GPUs and message size M, a ring all-reduce sends `2(N-1)/N · M` bytes per GPU — asymptotically `2M` regardless of N. Each GPU only ever sends to its right neighbor and receives from its left. It runs in `2(N-1)` steps: N-1 reduce-scatter steps + N-1 all-gather steps. The bandwidth-optimal property is what makes ring scale to thousands of GPUs.

**Tree all-reduce.** Latency `O(log N)` instead of `O(N)`. Win for small messages where you're latency-bound. NCCL switches between ring and tree based on message size (and you can override with `NCCL_ALGO=Ring|Tree`).

**Topology detection.** When NCCL initializes it auto-detects the topology — counting NVLinks per pair, identifying rail boundaries, deciding which GPUs to put on which ring. `NCCL_DEBUG=INFO` dumps this. Read it once on a real multi-GPU box; you'll learn more about your system in 5 minutes than from any doc.

**Common NCCL hangs and how to debug them.**
- **Rank ordering mismatch** — rank 0 calls `all_reduce(tensor_A)`; rank 1 calls `all_reduce(tensor_B)` of different shape. NCCL hangs forever. `NCCL_DEBUG=INFO` shows the mismatch.
- **Stuck on different streams** — communication is on stream X but the dependent compute is on stream Y, with no event between them. Hangs invisibly.
- **One rank skipped a collective** (early return on an exception). Other ranks wait forever. Always wrap your training step in `try/finally` that calls `barrier()` on exit.
- **Network flap** — NCCL doesn't recover from transient IB failures by default. Use Communicator Shrink (NCCL 2.27+) to drop the failed rank.

**Build steps.**
1. Write a small NCCL benchmark (or use `nccl-tests`). Measure all-reduce bandwidth at varying message sizes (1KB to 1GB). Plot it. You'll see the tree → ring crossover.
2. Compare 2-GPU NVLink vs 2-GPU PCIe (if available). The NVLink advantage is order-of-magnitude.
3. Force a hang: deliberately call `all_reduce` with mismatched shapes on rank 0 vs rank 1. Read the `NCCL_DEBUG=INFO` output. Learn to recognize the pattern.
4. Read NCCL env vars: `NCCL_DEBUG=INFO`, `NCCL_ALGO=Ring|Tree`, `NCCL_CROSS_NIC` for rail-aware routing, `NCCL_P2P_DISABLE` for fallback testing.

### 00b — `rdma-gpudirect-nixl`

**What it is.** The transport layer underneath NCCL and underneath KV-cache transfer in disaggregated inference (Level 5 referenced this; here it gets opened up).

**RDMA in one paragraph.** Remote Direct Memory Access. NIC reads/writes remote host memory without involving the remote CPU. The OS kernel is bypassed entirely after queue-pair setup. Latency drops from ~10µs (TCP) to ~1µs. Bandwidth approaches line-rate. The protocol is "verbs" (post send/recv to a queue, poll a completion queue) — different programming model from sockets but the abstraction NCCL builds on.

**GPUDirect RDMA.** RDMA reads/writes *GPU memory directly*, not host memory. The NIC and GPU talk over PCIe/NVLink without staging through DRAM. This is what makes inter-node all-reduce remotely fast. Without it, every GPU-to-GPU transfer would copy GPU→DRAM→NIC→DRAM→GPU.

**The transport hierarchy.**
- **Intra-node, same NVLink domain**: NVLink (1.8 TB/s on Blackwell). NCCL uses `nvlink` transport.
- **Intra-node, separate PCIe**: GPUDirect P2P over PCIe. NCCL uses `pci` transport.
- **Inter-node**: GPUDirect RDMA over IB or RoCE. NCCL uses `net/ib` or `net/socket` transport.

**NIXL (NVIDIA Inference Xfer Library).** A 2025 library specifically for KV-cache transfer in disaggregated inference. Sits above NCCL/UCX/RDMA, exposes a higher-level API: "transfer this list of KV blocks from worker A to worker B." Used by Dynamo, llm-d, vLLM disaggregated mode. The reason it exists: NCCL is designed for collectives (everybody participates); NIXL is designed for unicast point-to-point transfers (prefill worker sends to one specific decode worker), with batching of many small transfers.

**Why this matters for the curriculum.** Disaggregated inference (Level 5 Topic 08) was described abstractly. The mechanics — how bytes actually move from prefill GPU's HBM to decode GPU's HBM across a network — is RDMA + GPUDirect + NIXL. Same primitives are also why NCCL all-reduce is fast in distributed training.

**Build steps (mostly reading).**
1. Read the NCCL transport selection logic — or just run `NCCL_DEBUG=INFO` on a multi-node training job and look for `via NET/IB` vs `via P2P/IPC` in the output.
2. Read the NIXL repo's architecture page: [github.com/NVIDIA/NIXL](https://github.com/NVIDIA/NIXL).
3. Read llm-d's disaggregated inference docs to see how NIXL is used in practice.
4. Read [GPUDirect RDMA docs](https://docs.nvidia.com/cuda/gpudirect-rdma/) — at minimum the architecture diagram.
5. Write 200 words: "How does a KV block move from prefill worker's HBM to decode worker's HBM in a disaggregated vLLM setup?" Trace the path through NIXL, RDMA verbs, GPUDirect.

**You won't write RDMA code this week.** That's specialist. The goal is being able to read NCCL/NIXL output and diagnose why a transfer is slow.

### 01 — `interconnects`

**The 2026 fabric.**

| Layer | Hardware | Bandwidth |
|-------|----------|-----------|
| Scale-up (intra-node, intra-NVLink-domain) | NVLink 5 + NVSwitch | 1.8 TB/s/GPU bidirectional |
| Scale-up (extended) | GB200/GB300 NVL72 | 130 TB/s aggregate, 72 GPUs in one NVLink domain |
| Scale-out (inter-node) | InfiniBand Quantum-X800 (XDR) | 800 Gb/s |
| Scale-out alternate | Spectrum-X 800GbE / Ultra Ethernet | 800 Gb/s, RoCEv2 |

**Why this matters for the curriculum.** Interconnect choice dominates the cost-per-token math. A model that fits in one NVLink domain runs ~10× faster than the same model split across IB. The 5D parallelism axes are chosen partly to keep high-bandwidth communication intra-node and tolerate the lower bandwidth of inter-node.

**Rail-optimized topology.** Each NIC on a node connects to a separate switch ("rail"); NCCL pins flows to the same rail to avoid cross-rail interference. **Rail-only networks** (no inter-rail switching) are deployed at frontier scale to cut cost — it's standard vocabulary in networking-aware ML systems work.

**G10 of Project 3.** Training throughput vs interconnect type. You won't have NVLink 5 on Colab — but you can measure 2-GPU NVLink (if your rented box has it) vs simulated TCP fallback (`NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1`) and graph the delta.

### 02 — `data-parallel-from-scratch`

**What DDP communicates.** When you call `loss.backward()` under DDP, the framework all-reduces gradients across ranks. *That's all*. Forward pass is independent per rank; gradients are the only sync point.

**Build steps.**
1. Take any single-GPU training script (HuggingFace Trainer, plain PyTorch loop). Wrap with `DistributedDataParallel`.
2. Launch with `torchrun --nproc_per_node=2`.
3. `torch.profiler` it — find the `nccl:all_reduce` calls in the trace.
4. Measure step time at world_size=1 vs 2 vs 4 (if you have it). The gap from "linear scaling" is communication overhead.

**Insight to carry.** DDP requires the *whole model* to fit per rank. For models bigger than one GPU, you need FSDP.

### 03 — `data-loading-and-tokenization`

**The data-pipeline ceiling.** GPUs are fast. If your dataloader can't feed them, no model optimization helps. Real production training jobs are dataloader-bound more often than people admit.

**2026 stack.**
- **Mosaic StreamingDataset (MDS)** — drop-in `IterableDataset`, S3/GCS/Azure-backed shards, **elastic determinism** (same sample order regardless of GPU/node count), zstd/brotli compression. Dominant for LLM pretraining outside NVIDIA stack.
- **NVIDIA Megatron-Energon** — multimodal-first dataloader inside NeMo/Megatron-Bridge.
- **WebDataset** — tar-shard format. Losing ground to MDS for LLM use.
- **HuggingFace `datasets` IterableDataset** — fine for fine-tuning, not pretraining.

**Tokenization at scale.** Pre-tokenized shards are standard. Tokenize offline using the Rust `tokenizers` library or RAPIDS-tokenizer. Online tokenization is rare at >70B scale because it bottlenecks the loader.

**Sequence packing.** Concatenate multiple short sequences into one packed sample with cross-sequence attention masking. Hugely improves utilization on variable-length data. Standard in 2026.

**Build steps.**
1. Pick a small dataset (TinyStories, 1B tokens). Tokenize offline into MDS shards.
2. Train a 100M-parameter model with and without sequence packing. Measure tokens/sec, GPU util.
3. **G17 of Project 3:** tokenization throughput (tokens/sec from dataloader) vs training step throughput (tokens/sec consumed by model). Plot both. Where they cross is your wall.

### 04 — `fsdp2-and-dtensor`

**FSDP2 in one paragraph.** Each `nn.Parameter` becomes a `DTensor`, sharded along dim 0 across ranks. Forward pass: all-gather the full parameter shard before the layer runs, drop it after. Backward pass: re-gather, compute, reduce-scatter the gradient. Memory: 1/N of the parameters at rest, full only during the layer's forward/backward.

**Why it replaced FSDP1.**
- Per-parameter sharding (vs FSDP1's flat-parameter chunking) — 7% lower peak memory, ~1.5% higher throughput, deterministic memory.
- Sharded state-dict for free (no all-gather on save).
- Frozen parameters work (LoRA fine-tune composes with FSDP2 cleanly).
- Mixed precision composes cleanly (FP8 weights + BF16 activations).

**DeviceMesh** is mandatory. A multi-dim mesh like `Mesh(("dp_replicate", "dp_shard", "tp", "pp", "cp"))` lets you compose FSDP2, TP, PP, CP on the same model. Each parallelism module takes the relevant sub-mesh.

**Build steps.**
1. Single-host, 2 GPUs. Take your DDP script from Step 02.
2. Replace `DistributedDataParallel(model)` with per-block `fully_shard`:
   ```python
   from torch.distributed.fsdp import fully_shard
   for block in model.transformer_blocks:
       fully_shard(block, mesh=mesh["dp_shard"])
   fully_shard(model, mesh=mesh["dp_shard"])
   ```
3. Train. Measure: peak memory (should be ~1/2 of DDP), throughput (should be close to DDP at this small scale).
4. Save a sharded checkpoint via `torch.distributed.checkpoint.save`. Verify shape.

### 05 — `tensor-parallelism`

**What TP does.** Splits a single matmul across GPUs. The MLP `Y = X·W1·W2` becomes `Y = (X·W1_split)·W2_split` with an all-reduce in the middle. Megatron-style. Bandwidth-heavy — must stay inside the NVLink domain (TP ≤ 8 typically, ≤ 72 inside NVL72).

**Build steps (light).**
1. Use torchtitan or `torch.distributed.tensor.parallel.parallelize_module` to apply TP to a single transformer block.
2. Measure with TP=1 vs TP=2 on the same hardware. TP wins on memory; throughput depends heavily on interconnect.

### 06 — `pipeline-parallelism`

**What PP does.** Splits the model *depth-wise*. GPU 0 has layers 0–7, GPU 1 has 8–15, etc. Microbatches flow through the pipeline. **Bubble** = idle time at the start (filling) and end (draining) of the pipeline.

**Schedules.**
- **GPipe** (2018) — naive, big bubble. Historical only.
- **1F1B / PipeDream-Flush** — alternates one forward, one backward. Memory-efficient. Standard sync schedule.
- **Interleaved 1F1B (Megatron)** — multiple chunks per device, smaller bubble, more comms.
- **Zero Bubble (ZB-V, ICLR 2024)** — split backward into dW (weight grad) and dX (input grad); schedule dW into bubbles. **Practical default in 2026** — same memory as 1F1B, near-zero bubble.
- **DualPipe (DeepSeek-V3, late 2024)** — bidirectional fwd/bwd overlap. Cited heavily in 2026.

**Build steps.** Use torchtitan's PipelineStage. Run 1F1B and (if your version supports it) ZB-V. Compare bubble fractions.

### 07 — `expert-parallelism`

**What EP does.** For MoE models, distribute *experts* across devices. Each token in the batch routes to its top-K experts via all-to-all. Without EP, every device stores every expert (impossible at trillion-param scale).

**When you need it.** Any MoE model with `num_experts × expert_size > one GPU`. DeepSeek-V3 (671B total, 37B active) used EP heavily.

**Build steps (conceptual + small experiment).** Read the Megatron-Core EP docs. If you have access to a small MoE model (Qwen-MoE), run it with EP=2 on 2 GPUs. Otherwise, write a paragraph in your notes explaining the all-to-all routing pattern.

### 08 — `context-parallelism`

**What CP does.** Splits the *sequence dimension* across devices. Ring Attention or Striped Attention pattern — each device holds a slice of the K/V; queries iterate around the ring computing partial attention scores.

**When you need it.** Long-context training. Llama 4 (1M / 10M context), video / DiT pretraining, RAG fine-tuning on document corpora.

**Dynamic Context Parallelism (Megatron-Core, Jan 2026).** Static CP wastes GPU on short sequences in a packed batch. Dynamic-CP picks `cp_size` per microbatch (powers of 2), uses THD packed layout, pre-builds all CP communicator groups. ~1.48× speedup on Llama-13B GitHub data, >35% end-to-end at multi-thousand-GPU scale.

### 09 — `5d-parallelism-composition`

**The decision tree.**
- DP/FSDP: always on.
- TP: when one transformer layer's weights/activations don't fit on one GPU. Bound by NVLink domain.
- PP: when after FSDP+TP, the full model still doesn't fit. Crosses node boundaries.
- EP: triggered by MoE.
- CP: triggered by long sequences (≥32K, often ≥1M).

**Practical compositions.**
- Dense 70B: FSDP + TP + PP.
- MoE 400B: FSDP + EP + PP (+ TP for shared layers).
- Long-context 70B: FSDP + CP + TP.
- MoE long-context: all five.

### 10 — `torchtitan-or-megatron`

**Pick one. Train a small model.** This is where the week converges into a real artifact.

- **torchtitan** — PyTorch-native, simpler install, ICLR 2025 paper, Llama recipes 8B–405B. **Recommended for this curriculum** — closer to what most non-NVIDIA labs use.
- **Megatron-Core / Megatron-Bridge** — NVIDIA-native, more parallelism axes, day-0 model support. Bridge has bidirectional HF↔Megatron checkpoint conversion.

**Build steps with torchtitan.**
1. Clone torchtitan, install deps.
2. Use the Llama 3 8B config but scale down (smaller layers, smaller dataset).
3. Train for 200 steps on 2 GPUs with FSDP2 + (optional) TP=2.
4. Profile with PyTorch Profiler. Identify where time goes (compute, comms, data loading).
5. **Save the checkpoint.** Level 7 will load and serve this.

### 11 — `tail-latency-and-stragglers`

**The straggler problem.** In a sync training step, the slowest rank determines step time. If one GPU runs 10% slower (thermal throttle, ECC retry, mismatched workload), all others wait. At 1024 GPUs with 1% straggler probability per step, ~63% of steps have at least one straggler.

**Mitigation.**
- Detect via per-rank step-time variance.
- For mild stragglers: ignore (stay sync).
- For persistent stragglers: drop the rank via NCCL Communicator Shrink, downscale DP dimension.
- For systemic skew: rebalance work (heavier ranks get fewer microbatches).

**Build steps.** Manually slow one rank (`time.sleep` after backward) by 10%. Measure step-time impact at world_size=2 and 4. Plot p99 step time as a function of straggler severity. **G11 of Project 3.**

### 12 — `failure-injection`

**The 1024-node pattern in 2026.**
- **Async DCP** (`torch.distributed.checkpoint`) — sharded per-rank writes, GPU-to-host offload, host-to-storage offload thread. No more `torch.save` rank-0 gather.
- **Elastic launch** (`torchrun --rdzv`) + **NCCL Communicator Shrink** — drop the failed rank, resume on survivors at smaller world size.
- **Peer replication / in-memory checkpoints** (Gemini, ByteCheckpoint NSDI'25) — peer HBM every few minutes; durable storage every hour.
- **Goodput** as the SLO. A job at 90% MFU but 50% goodput (frequent restarts) is worse than 70% MFU at 95% goodput.

**Build steps.**
1. Train your torchtitan model with async DCP enabled.
2. Mid-training, kill one of the worker processes (`kill -9`).
3. Verify: surviving ranks detect the failure, NCCL Comm Shrink kicks in, training continues at smaller world size from the last saved checkpoint.
4. Document time-to-recovery in `reports/`.

### 13 — `checkpointing-async`

**Why async.** Synchronous checkpointing pauses training. At 70B+ scale, a checkpoint can take minutes — hours of lost compute per day.

**The async pattern.** Two-stage offload: GPU → pinned host buffer (fast, blocks training briefly), pinned host → object storage (slow, runs in background thread).

**Build steps.**
1. Enable DCP async save in torchtitan's config.
2. Time the save: training-pause time vs total-save time. The first should be tiny.
3. Verify resumability: kill mid-save, restart from previous checkpoint.

### 14 — `ray-and-multi-node`

**What Ray is for.** Job orchestration when you can't use SLURM. KubeRay (Ray on Kubernetes) is the cloud-native pattern. Ray Train wraps PyTorch DDP/FSDP for cluster-scheduled training jobs.

**Build steps (light touch — single-machine Ray cluster).**
1. `pip install "ray[default]"`. Start `ray start --head`.
2. Wrap your training function with `ray.train.torch.TorchTrainer`.
3. Submit a 2-worker job to your local cluster.
4. Note the difference: Ray handles process management, DDP setup, fault tolerance.

In production, KubeRay handles the same on Kubernetes. You'll see this again in Level 7.

### 15 — `rl-post-training-bridge`

**Brief context — the inference engine becomes the rollout backend.** RLHF/GRPO/PPO training requires generating completions during training (rollouts). In 2026 production, those rollouts run on **vLLM or SGLang**, not on the training framework itself.

**Frameworks.**
- **verl** (HybridFlow) — increasingly used in production RLHF pipelines.
- **OpenRLHF** — popular open-source.
- **TRL** (HuggingFace) — for smaller-scale.
- **NeMo-RL** — NVIDIA stack.

You don't run RLHF this week. But know the architecture: training framework + rollout engine + reward model + reference model, often colocated or disaggregated. Level 7's `mini-rlxf` topic touches this.

## Project 3 work this week

```
mini-platform/
├── training/
│   ├── torchtitan-config.toml
│   ├── train.py
│   ├── data/                    # Mosaic StreamingDataset shards
│   └── checkpoints/             # async DCP output
└── reports/
    └── training.md              # Level 6 output — feeds final platform report
```

Ships **G10** (interconnect throughput), **G11** (p99 step time with failure marker), **G17** (data-pipeline vs training throughput).

The trained checkpoint is the artifact. Level 7's `mini-platform` loads it and serves it.

## Definition of done

- [ ] You trained a small model with FSDP2 on at least 2 GPUs.
- [ ] You can sketch DeviceMesh-based composition of FSDP2 + TP on a whiteboard.
- [ ] You ran a Pipeline Parallelism schedule (1F1B at minimum) and measured the bubble.
- [ ] You can articulate when each of the 5 parallelism axes is needed.
- [ ] You injected a failure mid-training and recovered via Comm Shrink + DCP.
- [ ] You measured data-pipeline throughput against training throughput and identified the wall.
- [ ] You have a saved checkpoint to hand off to Level 7.

## Resources

- **FSDP2 tutorial** — [docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html).
- **torchtitan paper** — [arxiv.org/abs/2410.06511](https://arxiv.org/abs/2410.06511).
- **torchtitan repo** — [github.com/pytorch/torchtitan](https://github.com/pytorch/torchtitan).
- **Megatron-Bridge** — [github.com/NVIDIA-NeMo/Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge).
- **NCCL 2.27 features** — [developer.nvidia.com/blog/enabling-fast-inference-and-resilient-training-with-nccl-2-27](https://developer.nvidia.com/blog/enabling-fast-inference-and-resilient-training-with-nccl-2-27/).
- **Dynamic Context Parallelism** — [developer.nvidia.com/blog/speeding-up-variable-length-training-with-dynamic-context-parallelism](https://developer.nvidia.com/blog/speeding-up-variable-length-training-with-dynamic-context-parallelism-and-nvidia-megatron-core/).
- **Zero Bubble PP** — [arxiv.org/abs/2401.10241](https://arxiv.org/abs/2401.10241).
- **Mosaic StreamingDataset** — [github.com/mosaicml/streaming](https://github.com/mosaicml/streaming).
- **Async DCP** — [pytorch.org/docs/stable/distributed.checkpoint.html](https://pytorch.org/docs/stable/distributed.checkpoint.html).
- **Goodput (Google)** — [cloud.google.com/blog/products/ai-machine-learning/elastic-training-and-optimized-checkpointing-improve-ml-goodput](https://cloud.google.com/blog/products/ai-machine-learning/elastic-training-and-optimized-checkpointing-improve-ml-goodput).
- **Meta engineering — TP/CP/EP** — [engineering.fb.com/2025/10/17/ai-research/scaling-llm-inference-innovations-tensor-parallelism-context-parallelism-expert-parallelism](https://engineering.fb.com/2025/10/17/ai-research/scaling-llm-inference-innovations-tensor-parallelism-context-parallelism-expert-parallelism/).

## Common pitfalls

1. **Using FSDP1 docs.** Tutorials with `FullyShardedDataParallel(...)` are out of date. Use `fully_shard(...)` from `torch.distributed.fsdp`.
2. **Skipping the dataloader profiling.** "GPU util is 60%" almost always means a starved dataloader. Always measure both throughputs.
3. **Treating TP and DP as interchangeable.** TP communicates per-layer (latency-sensitive, must stay intra-node). DP/FSDP communicates per-step (less sensitive). Mixing them up = giving up half the speed.
4. **Synchronous checkpointing.** Production has been async since 2024. Sync checkpointing blocks the training loop.
5. **No failure injection.** "It works on the happy path" is not a finished week. Kill a process. Watch what happens. Document.
6. **Ignoring Goodput.** MFU is necessary but not sufficient. A run at 80% MFU that survives 14 days beats one at 95% MFU that crashes every 6 hours.

## What you'll be able to do after this week

> Train a small transformer with FSDP2 + DeviceMesh on multi-GPU; characterize 5D parallelism axes (TP/PP/DP/EP/CP) and run a 1F1B pipeline schedule. Implement async DCP checkpointing with NCCL Communicator Shrink for elastic recovery from injected node failures. Profile data pipeline (Mosaic StreamingDataset) against training step throughput, identify and resolve the loader-side bottleneck. Produce a Goodput-aware report on training reliability.
