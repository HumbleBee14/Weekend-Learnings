# 02 — Data Parallel From Scratch

DDP is the simplest piece of distributed training and the foundation everything else compares against. The whole point of this topic: see exactly which bytes move, when, and why.

## What DDP communicates

When you call `loss.backward()` under `DistributedDataParallel`, gradients are all-reduced across ranks. That is the only sync point in a step.

```
forward (independent per rank)
    ↓
backward (independent per rank — but DDP fires hooks)
    ↓
all-reduce gradients (NCCL)         ← the only collective
    ↓
optimizer.step (independent per rank — same grads → same update)
```

Because every rank applies the same averaged gradients to identical model weights, all ranks remain bit-identical (modulo nondeterministic ops). Therefore no parameter sync is needed — just gradient sync.

## Gradient bucketing — the actual implementation

DDP does not all-reduce one tensor per parameter. That would mean thousands of tiny NCCL calls. Instead it groups parameters into **buckets** (default ~25 MB) in *reverse* order of layer registration.

```
backward starts at the last layer
    ↓
gradient for layer N computes
    ↓
DDP hook fires: append grad to current bucket
    ↓
when bucket fills:  enqueue async all-reduce
    ↓
... compute continues for layer N-1, N-2, ...
    ↓
all gradients eventually all-reduced; .wait() at end of backward
```

This is the comms-compute overlap that makes DDP fast. The all-reduce of bucket K happens while backward is computing the gradients for buckets K-1, K-2, .... Effective overhead is `comms − overlap`.

The reverse-order is a heuristic: gradients for the last layers are computed first in backward, and parameters for the first layers are needed last in the next forward — so all-reducing them first hides the latency.

Tunables:
- `bucket_cap_mb=25` — default bucket size. Smaller = more overlap opportunity, more NCCL overhead.
- `gradient_as_bucket_view=True` — flatten grads into bucket buffers; one fewer copy.
- `static_graph=True` — assert the graph is fixed; lets DDP skip hook reordering on every step.
- `find_unused_parameters=True` — needed if some params don't get gradients each step (MoE, branching). Costs an extra pass; avoid if possible.

## What DDP does *not* fix

- **Memory.** Every rank holds the full model + full optimizer state + full gradients. For a 7B BF16 model with Adam, that is 7×2 (params) + 7×2 (grads) + 7×8 (Adam states fp32) = ~84 GB per rank. Doesn't fit on most GPUs. → FSDP exists for this reason.
- **Compute imbalance.** All ranks do identical work. If one batch is heavier than another for some reason (longer sequences, different shapes), the slowest rank dictates step time.
- **Activation memory.** No reduction; full per rank. → activation checkpointing, sequence parallelism.

## What DDP gets right

- **Throughput at small-medium scale.** When the model fits, DDP is hard to beat. Less per-step overhead than FSDP's gather+release.
- **Simplicity.** One wrapper, one collective, one mental model.
- **Composability with TP.** TP shrinks the per-rank model; DDP replicates the shrunk version. `world = TP_size × DP_size`.

## torchrun — the launcher

`torchrun` (the `torch.distributed.run` entrypoint) sets the env vars that `init_process_group` reads:
- `RANK` — global rank
- `LOCAL_RANK` — rank within the node (use this to pick `cuda:LOCAL_RANK`)
- `WORLD_SIZE`
- `MASTER_ADDR`, `MASTER_PORT`

```bash
# single node, 2 GPUs
torchrun --standalone --nproc_per_node=2 train.py

# two nodes, 8 GPUs each, with rendezvous
torchrun --nnodes=2 --nproc_per_node=8 \
    --rdzv_backend=c10d --rdzv_endpoint=node0:29500 \
    --rdzv_id=run42 train.py
```

`--rdzv_backend=c10d` is the elastic rendezvous; combined with `--max-restarts` it gives you re-entry on rank failure (Topic 12).

## Profiling

```python
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=True,
) as prof:
    for _ in range(10):
        train_step()

prof.export_chrome_trace("ddp_trace.json")
```

Open in `chrome://tracing` or [Perfetto](https://ui.perfetto.dev). Look for `nccl:all_reduce` ranges on the comms stream. If they cleanly overlap with backward kernels, you have good overlap. If they form a gap at the end of backward, the last bucket is comms-bound.

## Insight to carry into FSDP

DDP's communication volume per step: `2 × params × bytes_per_param × (N-1)/N` (one all-reduce of all gradients).

FSDP's communication volume: roughly `3 × params × bytes_per_param × (N-1)/N` (one all-gather in forward, one in backward, one reduce-scatter at end). About 1.5× the DDP traffic, but the model fits.

Both numbers are independent of N for ring algorithms. Both are bounded by interconnect bandwidth. This is why Topic 01's bandwidth measurements are not an academic exercise.

## Reference

- DDP design notes: [pytorch.org/docs/stable/notes/ddp.html](https://pytorch.org/docs/stable/notes/ddp.html)
- DDP tutorial (2024 update): [pytorch.org/tutorials/intermediate/ddp_tutorial.html](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- torchrun / elastic: [pytorch.org/docs/stable/elastic/run.html](https://pytorch.org/docs/stable/elastic/run.html)
- PyTorch profiler: [pytorch.org/docs/stable/profiler.html](https://pytorch.org/docs/stable/profiler.html)
