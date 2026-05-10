# 04 — FSDP2 and DTensor

FSDP2 (`fully_shard`) is the 2026 default for any model that exceeds one GPU's memory. FSDP1 (`FullyShardedDataParallel`) is deprecated. Tutorials that wrap with `FullyShardedDataParallel(...)` are out of date — use `fully_shard(...)` from `torch.distributed.fsdp`.

## What FSDP shards

Each `nn.Parameter` becomes a `DTensor` sharded along dim 0 across the FSDP ranks. At rest, each rank holds `params/N` of every parameter. The full parameter is materialized only briefly:

```
forward of layer L:
    all-gather L's params       ← one collective per layer
    compute L
    drop the gathered params    ← memory back down

backward of layer L:
    all-gather L's params       ← again
    compute backward
    reduce-scatter the gradient ← grad ends up sharded same way as param
    drop the gathered params
```

Memory at rest: `params/N` per rank. Peak memory during a layer: `params/N + one_layer_full_size`. For a 70B model on 8 GPUs that is ~9 GB at rest plus a few hundred MB peak. Fits.

## Why FSDP2 replaced FSDP1

FSDP1 used **flat parameters**: it concatenated all params in a unit, sharded the flat buffer, and reconstructed by views. This worked but had pain points:
- All params in a unit had to share dtype/requires_grad. Frozen-param fine-tuning (LoRA) didn't compose cleanly.
- Mixed precision was awkward (the flat buffer had one dtype).
- Memory was non-deterministic step-to-step.
- Sharded checkpoints required gather/scatter dances.

FSDP2 uses **per-parameter sharding**: each `nn.Parameter` is its own DTensor. Result:
- Frozen params just stay replicated (no shard).
- Mixed precision composes (each param keeps its dtype).
- Memory is deterministic.
- ~7% lower peak memory, ~1.5% higher throughput in published benchmarks.
- Sharded `state_dict` for free.

This is the kind of "API rewrite that pays for itself" story; the new API is also smaller.

## DeviceMesh — the composability machine

`DeviceMesh` is a multi-dimensional grid of ranks. Each parallelism axis (DP-replicate, DP-shard, TP, PP, CP) is a dimension of the mesh.

```python
from torch.distributed.device_mesh import init_device_mesh

mesh = init_device_mesh(
    "cuda",
    mesh_shape=(2, 2, 2),
    mesh_dim_names=("dp_replicate", "dp_shard", "tp"),
)

# slices of the mesh you can pass to each parallelism module
fsdp_mesh = mesh["dp_shard"]
hsdp_mesh = mesh["dp_replicate", "dp_shard"]   # hybrid sharded data parallel
tp_mesh = mesh["tp"]
```

Each parallelism module takes the relevant sub-mesh. FSDP2 takes the `dp_shard` (or `dp_replicate, dp_shard` for HSDP). TP takes the `tp` slice. Composing all of them:

```python
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel, RowwiseParallel

# 1. Tensor-parallelize each block first
for block in model.transformer_blocks:
    parallelize_module(block, tp_mesh, {
        "attn.wq": ColwiseParallel(),
        "attn.wk": ColwiseParallel(),
        "attn.wv": ColwiseParallel(),
        "attn.wo": RowwiseParallel(),
        "mlp.w1": ColwiseParallel(),
        "mlp.w2": RowwiseParallel(),
    })

# 2. FSDP-shard each block on the dp_shard dim
for block in model.transformer_blocks:
    fully_shard(block, mesh=fsdp_mesh)
fully_shard(model, mesh=fsdp_mesh)
```

The two compositions don't fight: TP shards within each block (across `tp_mesh` ranks), then FSDP shards the TP-sharded params again (across `fsdp_mesh` ranks). Total replication factor = `tp × fsdp_world`.

## HSDP — hybrid sharded data parallel

Sharding across all ranks is wasteful when inter-node bandwidth is much lower than intra-node. HSDP shards within a node and replicates across nodes. Communication pattern:
- intra-node (high bandwidth): all-gather + reduce-scatter (the FSDP collectives)
- inter-node (low bandwidth): all-reduce of the gradient at the end (DDP-style)

Mesh: `(replicate=n_nodes, shard=gpus_per_node)`. Parameters fit in `params / gpus_per_node` per rank, and inter-node traffic is one all-reduce per step instead of an FSDP gather-cycle.

## Activation checkpointing composes cleanly

```python
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

for i, block in enumerate(model.transformer_blocks):
    block = checkpoint_wrapper(block)
    fully_shard(block, mesh=fsdp_mesh)
```

The `checkpoint_wrapper` re-runs forward in backward to save activation memory; FSDP handles the parameter dance independently. Stacking is straightforward in FSDP2 — was painful in FSDP1.

## Sharded checkpointing for free

```python
import torch.distributed.checkpoint as dcp

state = {"model": model.state_dict(), "optim": optim.state_dict()}
dcp.save(state, checkpoint_id="ckpt/step_1000")
```

Each rank writes its own shard to disk. No gather. Fast. Topic 13 covers async DCP.

## Mixed precision the right way

```python
from torch.distributed.fsdp import MixedPrecisionPolicy

mp = MixedPrecisionPolicy(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,   # accumulate grads in fp32 for stability
)
fully_shard(block, mesh=fsdp_mesh, mp_policy=mp)
```

`reduce_dtype=fp32` is the safe default for production. `param_dtype=bfloat16` gives you the throughput; the master weights live in fp32 in the optimizer state.

For FP8 + BF16 (Hopper/Blackwell), `param_dtype=torch.bfloat16` and your Linear modules use FP8 internally via `torchao` or `transformer_engine`. FSDP2 handles this; FSDP1 didn't.

## Build steps

1. Reuse the Topic 02 transformer.
2. Replace `DDP(model, ...)` with FSDP2 per-block + global wrap (see `fsdp_train.py`).
3. Train. Measure peak memory (should be ~1/2 of DDP for 2 GPUs at this small scale, much more dramatic for big models).
4. Save a sharded checkpoint via `torch.distributed.checkpoint.save`. Confirm shard files appear.

## Common foot-guns

- **Forgetting the global wrap**. After per-block `fully_shard`, do `fully_shard(model)` to handle the embedding/head and any params not covered by per-block calls. Otherwise embedding stays replicated and you wasted the memory.
- **Param dtype mismatch with optimizer state**. Optimizer must be initialized *after* FSDP wraps. Otherwise it sees full-shape params.
- **Calling `.to(device)` post-FSDP**. Don't. FSDP places params on the right device during the wrap.
- **Forgetting to set the seed before model construction**. With per-rank RNG drift, ranks can build different random init values, causing silent training divergence. Set torch seeds before `nn.Module(...)`.

## Reference

- FSDP2 tutorial: [pytorch.org/tutorials/intermediate/FSDP_tutorial.html](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- FSDP2 design: [github.com/pytorch/pytorch/issues/114299](https://github.com/pytorch/pytorch/issues/114299)
- DTensor: [pytorch.org/docs/stable/distributed.tensor.html](https://pytorch.org/docs/stable/distributed.tensor.html)
- DeviceMesh: [pytorch.org/docs/stable/distributed.html#device-mesh](https://pytorch.org/docs/stable/distributed.html#device-mesh)
- HSDP rationale: [pytorch.org/blog/training-production-ai-models](https://pytorch.org/blog/training-production-ai-models/)
