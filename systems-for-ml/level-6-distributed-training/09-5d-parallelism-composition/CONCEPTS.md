# 09 — 5D Parallelism Composition

The whole point of the previous topics. Each axis solves one constraint. Stacking them is how frontier-scale training fits at all.

## The five axes

| Axis | What it shards | Frequency | Bandwidth needs |
|---|---|---|---|
| **DP / FSDP** (data parallel / fully-sharded) | input batch + (FSDP) parameters | once per step | inter-node OK |
| **TP** (tensor parallel) | one matmul split across devices | per layer | intra-NVLink only |
| **PP** (pipeline parallel) | model depth | per microbatch boundary | inter-node OK |
| **EP** (expert parallel) | MoE experts | per MoE layer (all-to-all) | intra-NVLink preferred |
| **CP** (context parallel) | sequence dimension | per attention block | intra-NVLink preferred |

Total replication factor: `DP × TP × PP × EP × CP = world_size`.

Each axis lives on a dimension of `DeviceMesh`. The mesh is what makes composition tractable.

## The decision tree

Read top to bottom. Each node activates its axis:

```
Q1: Does the model fit on one GPU?
    Yes → done. No parallelism beyond batch (DDP).
    No → continue.

Q2: Does one transformer layer fit on one GPU after FSDP?
    Yes → just FSDP. Easy mode.
    No → add TP. TP_size limited by NVLink domain (≤8 on Hopper, ≤72 on NVL72).

Q3: Does the full model fit on one node after FSDP+TP?
    Yes → no PP needed. Stay intra-node where possible.
    No → add PP across nodes.

Q4: Is the model MoE?
    Yes → add EP. Replaces some of FSDP's role for the expert weights.
    No → skip.

Q5: Are sequences ≥32K (often ≥1M)?
    Yes → add CP within or across nodes.
    No → skip.
```

## Practical compositions

### Dense 7B-class (e.g., Llama-3-8B at home)

```
DP / FSDP only.
mesh: (world,) with name "dp_shard"
```

Fits per rank. Just FSDP2.

### Dense 70B-class

```
TP=8 (intra-node) × FSDP across nodes
mesh: (n_nodes, 8) with names ("dp_shard", "tp")
```

The model is too big for FSDP alone on 8×H100 — TP shrinks per-rank work, FSDP shards what's left across nodes.

### Dense 405B-class

```
TP=8 × PP=8 × FSDP across DP groups
mesh: (dp, pp, tp) — 3-D
```

PP crosses nodes. TP within node. FSDP across DP groups (each DP group covers one PP+TP rank set).

### Llama-4 / 1M-context training

```
TP=8 × CP=4 × FSDP across DP groups
mesh: (dp, cp, tp)
```

CP shrinks the per-rank sequence chunk. TP shrinks the per-rank attention head set.

### MoE 400B (Mixtral-style)

```
EP=64 × FSDP for non-expert layers × TP=4 for shared layers
mesh: (dp, ep, tp)  with experts pinned to ep dim
```

Experts live on EP-dim ranks. Shared layers (attention, embed, norm) use TP and FSDP normally.

### MoE long-context (DeepSeek-style)

All five axes. The full 5-D mesh.

## DeviceMesh in practice

```python
from torch.distributed.device_mesh import init_device_mesh

# Llama-3-70B style: 8 nodes × 8 GPUs = 64 GPUs
mesh = init_device_mesh(
    "cuda",
    mesh_shape=(8, 1, 1, 8, 1),   # (dp_replicate, dp_shard, pp, tp, cp)
    mesh_dim_names=("dp_replicate", "dp_shard", "pp", "tp", "cp"),
)

# Apply each parallelism to its sub-mesh:
parallelize_module(block, mesh["tp"], plan=...)        # TP within node
fully_shard(block, mesh=("dp_replicate", "dp_shard"))  # HSDP across nodes
# PP and CP would compose similarly
```

## What NOT to do

- **TP across nodes**. Per-layer all-reduce on IB is slow enough to wreck step time. Always keep TP intra-NVLink.
- **EP all-to-all across many nodes** unless the model demands it. The all-to-all dominates step time at high inter-node EP. Keep EP intra-node when feasible (DeepSeek-V3 has EP cross node but pays for it with DualPipe overlap).
- **PP with only 1 microbatch per stage**. Pure bubble. PP requires ≥ S microbatches to amortize.
- **FSDP everywhere**. FSDP's all-gather is bandwidth-heavy at scale. HSDP (sharding within node, replicating across) is the sweet spot at multi-node.

## Worked example: Llama-3-8B on 8×A100

The 8B model fits per A100 (80GB) with BF16 + Adam (~84 GB total — tight). FSDP halves it. Decision: FSDP across all 8, no TP, no PP.

```python
mesh = init_device_mesh("cuda", (8,), ("dp_shard",))
for block in model.blocks:
    fully_shard(block, mesh=mesh)
fully_shard(model, mesh=mesh)
```

That's it. Don't over-engineer.

## Worked example: Llama-3-70B on 64×A100 (8 nodes × 8)

Won't fit per rank even with FSDP. Need TP within node. Across nodes use FSDP.

```python
mesh = init_device_mesh("cuda", (8, 8), ("dp_shard", "tp"))
for block in model.blocks:
    parallelize_module(block, mesh["tp"], llama_tp_plan)
    fully_shard(block, mesh=mesh["dp_shard"])
fully_shard(model, mesh=mesh["dp_shard"])
```

TP=8 inside each node (NVLink); FSDP=8 across nodes (IB).

## Worked example: Llama-3-405B on 256×H100 (32 nodes × 8)

Need PP. Layout: PP=8 (4 layers per stage), TP=8 (intra-node), DP=4 (across DP groups, each covering one PP+TP slice).

```
mesh = init_device_mesh("cuda", (4, 8, 8), ("dp_shard", "pp", "tp"))
```

This is what torchtitan's 405B recipe uses.

## Why goodput, not MFU, is the SLO in 2026

MFU (Model FLOPs Utilization): `actual_flops / peak_flops`. Tells you how efficient one step is.

Goodput (Google): `useful_training_time / wallclock_time`. Accounts for failures, restarts, stragglers, slow checkpoints.

A run at 90% MFU but 50% goodput (frequent restarts) is worse than one at 70% MFU at 95% goodput. Frontier-scale training is judged on goodput. Topic 12 (failure injection) and Topic 13 (async checkpointing) are how you keep goodput high.

Reference: [cloud.google.com/blog/products/ai-machine-learning/elastic-training-and-optimized-checkpointing-improve-ml-goodput](https://cloud.google.com/blog/products/ai-machine-learning/elastic-training-and-optimized-checkpointing-improve-ml-goodput).

## Reference

- torchtitan paper (5D parallelism): [arxiv.org/abs/2410.06511](https://arxiv.org/abs/2410.06511)
- Megatron-Core parallelism docs: [docs.nvidia.com/megatron-core/latest/api-guide/parallel_state.html](https://docs.nvidia.com/megatron-core/latest/api-guide/parallel_state.html)
- DeepSeek-V3 (5D MoE): [arxiv.org/abs/2412.19437](https://arxiv.org/abs/2412.19437)
- Goodput: [cloud.google.com/blog/products/ai-machine-learning/elastic-training-and-optimized-checkpointing-improve-ml-goodput](https://cloud.google.com/blog/products/ai-machine-learning/elastic-training-and-optimized-checkpointing-improve-ml-goodput)
