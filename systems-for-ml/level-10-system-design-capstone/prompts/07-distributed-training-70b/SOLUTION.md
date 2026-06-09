# Prompt 07 — Worked Solution

> The only training-side prompt in the set. Most senior ML systems interviews focus on inference now, but training infra remains a senior-eng skill — and the failure modes are completely different (synchronous, long-running, expensive when wrong).

## 1. Clarifying questions (the first 3 minutes)

1. **Tokens to train on?** 70B Chinchilla-optimal is ~1.4T tokens; 70B compute-optimal-2026 (e.g., Llama-3-70B) is ~15T tokens. Massive difference — 6 weeks on 256 H100 covers ~3-4T tokens at typical MFU. Confirm budget vs. token target.
2. **Data composition mix.** Is it research-iterable (researchers want to tweak the mix during the run)? Or fixed (pre-tokenize once, ship parquet shards)? Affects pipeline design.
3. **Eval cadence during training.** Eval every checkpoint (every 30 min) or just at major milestones? (Affects whether eval has dedicated GPUs or steals from training.)
4. **Restart-from-failure tolerance.** Acceptable to lose 30 min of work on a node failure, or do we need ≤5 min? (Affects checkpoint frequency and async-DCP investment.)
5. **Goodput target source.** Is 85% goodput an explicit ask, or a "we want a good number"? (At 85%, you're tolerating 15% wall-time loss — that's roughly one major failure per week on this cluster size.)
6. **Hardware ownership.** Is the 256 H100 a dedicated reservation we control, or shared with other teams? (Shared changes everything — pre-emption, queue waits, can't pin topology.)

**Assumptions if waved off:** ~3T tokens target, fixed data mix pre-tokenized to parquet, eval at every checkpoint (30 min), ≤30 min restart tolerance, dedicated 256-H100 reservation with topology we own.

## 2. The right answer in one sentence

**256 H100s arranged as 32 nodes × 8 GPUs, running torchtitan with FSDP=32 / TP=8 (2D parallelism inside-node TP, across-node FSDP) plus Float8 mixed precision, async-DCP checkpoint every 30 min, NCCL 2.27+ with SHARP for collective offload, Communicator Shrink for single-node-failure recovery without world restart, Mosaic StreamingDataset for the data pipeline with sequence packing, and Prometheus + Grafana on per-rank goodput.**

This is essentially the published [torchtitan Llama-3.1-70B configuration on 256 H100s](https://arxiv.org/abs/2410.06511): FSDP=32 × TP=8 × Float8 + AsyncTP gives ~12.59% throughput improvement over the FSDP-only baseline. The senior signal is **not inventing a new parallelism config** — the field has converged on this exact shape for this exact model size, and naming the published config shows you've read the work.

## 3. The architecture (whiteboard)

```
   ┌──────────────────────────────────────────────────────────────────┐
   │                256 H100 SXM5 cluster (32 nodes × 8 GPU)          │
   │                                                                  │
   │   Node 1 (8× H100, NVLink5 intra-node, 900 GB/s)                 │
   │   ┌──────────────────────────────────────────────┐               │
   │   │  GPU 0  GPU 1  GPU 2  GPU 3                  │               │
   │   │  GPU 4  GPU 5  GPU 6  GPU 7   ◄── TP=8       │               │
   │   │                                  (intra-node) │               │
   │   └──────────────────────────────────────────────┘               │
   │                            │                                     │
   │              InfiniBand NDR (400 Gb/s per node,                  │
   │              fat-tree topology, 1.6 Tb/s bisection)              │
   │                            │                                     │
   │   Node 2 ──── ... ──── Node 32         ◄── FSDP=32              │
   │                                            (across nodes,        │
   │                                             param shards)        │
   └──────────────────────────────────────────────────────────────────┘
                                │
                                ▼
   ┌──────────────────────────────────────────────────────────────────┐
   │  Storage tier                                                    │
   │  ─ Lustre/WekaIO for active training data (sharded parquet)      │
   │    ~50 TB pre-tokenized corpus, ~12 GB/s aggregate read          │
   │  ─ Object storage (S3/GCS) for checkpoints                       │
   │    full checkpoint ≈ 140 GB (model + opt state + scheduler)      │
   │    sharded async DCP write, peer-replicated                      │
   │  ─ Local NVMe (per node, ~3.84 TB) as warm tier for next         │
   │    checkpoint, also held by peer node                            │
   └──────────────────────────────────────────────────────────────────┘

   ┌──────────────────────────────────────────────────────────────────┐
   │  Data pipeline                                                   │
   │                                                                  │
   │   Mosaic StreamingDataset                                        │
   │   ─ Sharded parquet on Lustre, each rank streams its shard       │
   │   ─ Sequence packing: pack variable-length docs into 8K-token    │
   │     fixed sequences for tensor-aligned batches                   │
   │   ─ Tokenizer pool (off-GPU CPU workers, 32 workers per node)    │
   │   ─ Prefetch depth = 4 batches; if pipeline starves the GPU      │
   │     for >5% of step time, alert and resize tokenizer pool        │
   └──────────────────────────────────────────────────────────────────┘

   ┌──────────────────────────────────────────────────────────────────┐
   │  Control plane                                                   │
   │  ─ Job orchestrator: SLURM (or Ray) for the launch + restart     │
   │    loop                                                          │
   │  ─ Monitoring: Prometheus scrapes every rank — step time, MFU,   │
   │    grad norm, KV inflight, comm time, dataloader idle %          │
   │  ─ Grafana: per-rank step-time heatmap (stragglers visible)      │
   │  ─ Goodput tracker: rolling 24h `useful_compute / wall_time`     │
   │  ─ Failure handler: on NCCL ECCError or rank timeout, trigger    │
   │    Communicator Shrink, mark rank dead, resume from last DCP     │
   │  ─ Async checkpoint writer (own thread, doesn't block step)      │
   └──────────────────────────────────────────────────────────────────┘
```

### Parallelism mapping

```
TP = 8   (intra-node, NVLink5 bandwidth)
        ─ Megatron-style attention/MLP split inside each layer
        ─ All-reduce per layer; small but FREQUENT → needs NVLink

FSDP = 32 (across 32 nodes, InfiniBand)
        ─ Per-parameter sharding via DTensor (FSDP2 API)
        ─ All-gather weights pre-forward; reduce-scatter grads post-backward
        ─ BIG but INFREQUENT → InfiniBand fine; SHARP helps

Total world = TP × FSDP = 8 × 32 = 256 ✓

Local batch  = 16 sequences × 8K tokens = 131,072 tok/rank
Global batch = 16 × 32 = 512 sequences = 4.2M tok/step ← Chinchilla-scale
Step time    = ~3.5 sec (with AsyncTP + Float8 + torch.compile)
Tokens/sec   = ~1.2M tok/s sustained
MFU          = ~52% (Float8 makes pure MFU ill-defined; this is BF16-equivalent)
3T tokens    = 3T / 1.2M / 86400 ≈ 29 days of pure compute
+ goodput overhead: 29 / 0.85 ≈ 34 days wall-time
```

This fits comfortably in the 6-week budget with ~1 week of slack for restarts, eval gaps, and the inevitable mid-run config tweak the researchers want.

## 4. The goodput math (Level 6 Topic 13)

Goodput is the SLO of choice for frontier-scale training. Definition:

```
goodput = useful_compute_wall_time / total_wall_time
        = (time at MFU > threshold)  /  total
```

```
On a 256-H100 cluster, empirical MTBF per node ≈ 12-24 hours
  (mix of GPU ECC errors, NCCL hangs, network blips, host reboots).

256 nodes ÷ 24h MTBF ≈ ~1 node failure per hour cluster-wide.
But most failures are recoverable in seconds (transient NCCL).
True "we lost the run" failures: ~1 / 36 hours typical.

Lost-work budget at 85% goodput on 6 weeks (1008 hours):
  Lost time = 1008 × 0.15 = 151 hours of wall-time
  We can afford ~150 hours of recovery work + idle time

Per-failure cost components:
  ─ Detection lag:           10-60 sec (NCCL timeout, rank doesn't ack)
  ─ Communicator Shrink:     30-60 sec (NCCL 2.27+ feature, dynamic ring resize)
  ─ Resume from last DCP:    2-5 min (DCP shard load + state sync)
  ─ Replay since checkpoint: 0-30 min (worst case = checkpoint right before failure)
  ─ Average failure recovery: ~15 min

Failure budget at 15 min per failure, 150 hour budget:
  = 150 × 60 / 15 = 600 failures permissible over 6 weeks
  At ~1 failure / 36 hours → 28 failures over 6 weeks
  Comfortable margin (600 budget, 28 actual).

The bound that bites first: if mean time to recovery rises (e.g., DCP load
gets slow because object storage degrades), 15 min/failure → 60 min/failure
collapses the budget. MTTR monitoring is mandatory.
```

The senior signal: **named the 85% goodput as a budget, decomposed it into MTBF × MTTR, and verified the design fits.** Not "we'll just have good uptime."

## 5. The hard parts

### 5.1 Async DCP checkpointing — the math that makes 30-min checkpoints free

Naive checkpoint: pause training, sync all ranks, gather sharded weights, write to storage. On 256 H100s with 140GB checkpoint at 12 GB/s storage write = ~12 sec write + ~5 sec sync overhead = **17 sec stalled per checkpoint × 2 per hour = ~34 sec/hour lost to checkpointing**. At 6 weeks that's ~6 hours wasted = ~0.6% goodput hit.

Async DCP (PyTorch's distributed checkpoint, 2024+):
- Forks the save thread; main training continues immediately
- Background thread copies tensors to CPU memory (fast, ~2 sec)
- Slower background thread writes CPU → NVMe → object storage (slow but invisible)
- Main training pays only the CPU-copy cost: ~2 sec stalled, masked under data loading

```
Result: checkpoint stall drops from 17 sec to ~0 (overlapped).
Combined with peer replication (each rank's checkpoint shard held by 1 peer,
making recovery a peer-to-peer NIXL transfer instead of object-store read),
recovery time drops from 5 min to ~30 sec.
```

### 5.2 NCCL 2.27+ Communicator Shrink — the elastic-recovery primitive

Before 2.27: if rank 47 dies mid-step, the whole NCCL world hangs (collective stalls), training process detects timeout (~5 min), then has to **kill the whole world, restart all 256 processes, re-initialize NCCL, re-load checkpoint**. Total recovery: 15-25 min.

With 2.27 Communicator Shrink:
- NCCL detects the failed rank in seconds
- The remaining 255 ranks rebuild the communicator excluding the failed rank
- Training continues on the surviving 255 (now with `world_size = 255`, which the framework has to handle — torchtitan does)
- The failed node is replaced (provisioned in 2-4 min from spare pool)
- Re-joins the world via NCCL Init Add, becomes rank 47 again
- Resumes from last DCP shard

Total recovery: ~5 min, vs. ~25 min pre-2.27. **For a 6-week run with ~28 failures, that's saving ~9 hours of goodput — directly worth tens of thousands of dollars in compute.**

### 5.3 Data pipeline — the bottleneck most curricula skip

```
256 H100s × ~1.2M tok/s sustained = ~310M tokens/sec consumed cluster-wide.
At ~16 bytes/token (Llama-3 tokenizer + parquet overhead): ~5 GB/s read.
This is well below Lustre's ~12 GB/s aggregate — but only if the pipeline is set up right.

Failure modes to design around:
  ─ Tokenizer-bound: too few CPU workers, GPU starves. Fix: 32 workers/node, prefetch 4.
  ─ Hot-shard skew: 1/32 ranks reading a "popular" shard, becomes IO-bottleneck.
    Fix: pre-tokenize + shuffle once, ship deterministically-sharded parquet.
  ─ Sequence-length skew: variable-length docs cause padding waste.
    Fix: SEQUENCE PACKING — concat docs to fill 8K-token sequences, mask
    cross-document attention. 30-50% throughput win.

Monitoring: `dataloader_idle_percent` per rank. Alert at >5%; this is what
"training failed on data, not GPUs" looks like.
```

### 5.4 Float8 — the 2026 free lunch for training

Float8 (Hopper FP8 E4M3/E5M2) mixed precision: weights in BF16 master, forward + backward in FP8, gradient accumulation in FP32. ~30-40% throughput gain vs pure BF16 on H100, similar gain on B200. Quality regression on 70B-class training is <0.5% on Chinchilla-shaped runs, well within noise.

The trap: **Float8 requires correctness work.** Loss scaling, dynamic range tracking, NaN guards. torchtitan ships this end-to-end; rolling your own takes a quarter and probably regresses quality.

```
Pure BF16:       1.0× baseline throughput, 100% well-understood
+ Float8 mixed:  1.3-1.4× throughput, mostly well-understood (2024-2026 work)
+ AsyncTP:       +12.59% on top of Float8 (torchtitan paper)
Combined effect: ~50% more throughput than naive BF16 FSDP
```

### 5.5 What goes wrong specifically at 256 nodes (not 16)

- **NCCL hangs become normal.** At 16 nodes you might see one a month. At 256 you see them weekly. The Communicator Shrink path is exercised continuously.
- **Stragglers.** One node running 2% slower than the rest stalls every step. p99 step time matters more than mean. Auto-eject stragglers via threshold-based detection.
- **Network microbursts.** All-gather bursts hit InfiniBand fabric simultaneously; some nodes' switches see brief congestion. Adaptive routing on the fabric matters.
- **Storage hotspots.** All ranks hit Lustre in lockstep at dataloader-shuffle time. Stagger the shuffle phase per-rank by a few hundred ms.

## 6. Break-it list

| Failure | What happens | Mitigation |
|---|---|---|
| One H100 dies (ECC failure) | Communicator hangs, training halts | NCCL 2.27 Communicator Shrink; replace from spare pool in 2-4 min; resume |
| Whole node dies (host reboot) | Lose 8 GPUs, 1/32 of FSDP shards | Peer-replicated DCP — peer has the shard; recovery via NIXL transfer ~30s |
| Network partition (rack switch flap) | NCCL all-reduce stalls | NCCL_TIMEOUT triggers shrink; auto-eject affected nodes; reroute on healthy paths |
| Researcher wants to change data mix mid-run | Need to swap dataset without losing optimizer state | StreamingDataset supports mid-run shard swap; resume optimizer state; eval-only checkpoint catches quality drift |
| Loss spike (catastrophic gradient explosion) | Loss → NaN, training useless | Grad norm clipping at 1.0; gradient skipping if norm > 10× rolling mean; rewind to checkpoint pre-spike, restart with smaller LR for window |
| Lustre filesystem degrades | Dataloader starves GPUs | Local NVMe cache; have peer-replicated tokenized data; warn at 5% idle, hard-fail at 20% idle |
| Object storage outage (S3) | Can't checkpoint | Buffer to local NVMe; alert; resume async upload when S3 returns; never block training |
| Float8 quality regression | Loss curve diverges from BF16 baseline | Run a control BF16 mini-job in parallel for first 24h; if Float8 main run diverges, fall back to BF16 (~30% slower, recoverable) |
| Spare node pool exhausted | No replacement for failed nodes; world keeps shrinking | Alert at 80% pool depletion; ask infra for more; fail gracefully if pool empty (shrink and continue with smaller world, accept goodput hit) |
| Single rank consistently 2% slower (straggler) | All steps wait for it | Step-time outlier detection; auto-eject and replace within 1h |
| Hardware silently corrupts (silent data corruption) | Loss curve becomes weird, no error | Periodic per-rank gradient checksum (rank → average → diff); CCL parity checks; this is rare but real at this scale |

## 7. What changes at 10× scale (2,560 H100s for 700B+ model)

At 2,560 GPUs, the design changes shape:

**3D parallelism becomes mandatory.** Add PP=4 (Pipeline Parallelism) on top of TP=8 × FSDP. Now 3D: TP=8 × PP=4 × FSDP=80. Zero-Bubble (ZB-V) scheduling for low pipeline bubble. Memory math forces it: 700B in BF16 = 1.4 TB; without PP, FSDP-only doesn't fit.

**Multi-cluster orchestration.** Even one cluster doesn't hold 2,560 GPUs in many DCs. Cross-cluster training over Ultra Ethernet or dedicated dark fiber. NCCL hierarchical collectives (SHARP within cluster + custom inter-cluster reduce).

**Mosaic-style elastic training.** Add/remove nodes mid-run as availability shifts. Trade some efficiency for the ability to opportunistically use spot/preemptible nodes for ~30% cost reduction on the non-critical path (longer-tail data shards).

**Curriculum + mid-run RL post-training.** At 700B you do continued pre-training, then SFT, then RL all in the same orchestrated job. Level 6 Topic 15 (rl-post-training-bridge) becomes a real workstream — vLLM as rollout backend on a sibling cluster.

**Co-designed silicon.** At this scale you can shape the next-gen ask: B200 → B300 → Rubin. Negotiating tensor formats, NVLink bandwidth, HBM capacity directly with NVIDIA. (Real example: xAI's Colossus cluster reportedly co-designed pieces of the H100 SXM5 board layout.)

**Org-shape: dedicated training-infra team.** 5-8 engineers full-time on the training stack alone, separate from inference platform team, separate from model researchers. This is "Anthropic Training Infra" / "OpenAI Pretraining Platform" as a discipline.

## 8. The 30-second summary

> "I'd run torchtitan with FSDP=32 across nodes × TP=8 intra-node × Float8 mixed precision + AsyncTP — the published Llama-3.1-70B-on-256-H100 config that gets ~50% MFU-equivalent. NCCL 2.27 Communicator Shrink for single-node failure recovery without world restart, async DCP checkpoint every 30 min with peer replication, Mosaic StreamingDataset with sequence packing for the data path. Goodput target 85% maps to ~150 hours of recovery budget over 6 weeks; with ~28 expected failures at ~15 min recovery each, we have plenty of margin. ~3T tokens trainable in ~34 wall-clock days. At 10× scale we add PP=4 for 3D parallelism, cross-cluster orchestration, and a dedicated training-infra team."

## What this prompt is really testing

- **Knowing the published config.** FSDP=32 × TP=8 × Float8 on Llama-3.1-70B on 256 H100 is torchtitan's flagship configuration; reciting it shows you've read the work.
- **Goodput as a budget**, not vibes. Decomposing 85% goodput into MTBF × MTTR × failure count.
- **NCCL 2.27 Communicator Shrink** as a specific named feature — separates 2025 candidates from 2026 candidates.
- **Async DCP + peer replication** as a connected pair — not just "we'd checkpoint."
- **Data pipeline as a first-class concern** — "dataloader_idle_percent" as a monitored metric.
- **At 10× scale answer naming 3D parallelism, multi-cluster, co-design** — the seniority signal.

## References

- [Topic 00 — collectives-and-nccl](../../../level-6-distributed-training/00-collectives-and-nccl/) — NCCL 2.27 features
- [Topic 00b — rdma-gpudirect-nixl](../../../level-6-distributed-training/00b-rdma-gpudirect-nixl/) — NIXL for peer-to-peer DCP recovery
- [Topic 04 — fsdp2-and-dtensor](../../../level-6-distributed-training/04-fsdp2-and-dtensor/)
- [Topic 05 — tensor-parallelism](../../../level-6-distributed-training/05-tensor-parallelism/)
- [Topic 09 — 5d-parallelism-composition](../../../level-6-distributed-training/09-5d-parallelism-composition/) — the 2D config detail
- [Topic 10 — torchtitan-or-megatron](../../../level-6-distributed-training/10-torchtitan-or-megatron/)
- [Topic 12 — failure-injection](../../../level-6-distributed-training/12-failure-injection/) — Communicator Shrink in action
- [Topic 13 — checkpointing-async](../../../level-6-distributed-training/13-checkpointing-async/) — async DCP + goodput math
- [NETWORKING-PRIMER.md](../../../level-6-distributed-training/NETWORKING-PRIMER.md) — the fundamentals
- [TorchTitan paper (arxiv 2410.06511)](https://arxiv.org/abs/2410.06511) — the published config
- Reddi *Machine Learning Systems* Vol 2, *Distributed Training* chapter
- Note: Kiely's *Inference Engineering* doesn't cover training — this is one of the few prompts where Reddi is the primary reference.
