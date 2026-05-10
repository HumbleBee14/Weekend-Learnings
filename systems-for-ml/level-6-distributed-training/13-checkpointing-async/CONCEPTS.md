# 13 — Async Checkpointing

Synchronous checkpointing pauses training. At 70B+ scale a checkpoint can take minutes. Hours of lost compute per day. The 2026 default is async DCP — pause training for ~10 ms, finish writing in the background.

## Why sync hurts

Sync `torch.save` flow:
1. Stop training.
2. Gather all parameter shards to rank 0 (huge collective).
3. Serialize and write to disk/object store.
4. Resume training.

For 70B BF16 + Adam: ~140 GB to gather, ~280 GB to write (with optimizer state). Even at 1 GB/s effective write bandwidth, that's 4–5 minutes per checkpoint.

If you checkpoint every hour, you lose ~7% of training time. Often more — if write bandwidth is shared with other tenants, sync checkpoints can take 10–30 minutes.

## Async DCP architecture

Two-stage offload:

```
training step
    ↓
checkpoint trigger
    ↓
stage 1 (blocking, ~10 ms):
  GPU shards → pinned host buffers (cudaMemcpyAsync)
    ↓
training step resumes
    ↓
stage 2 (background, seconds-minutes):
  pinned host → object storage (separate thread + multi-part upload)
```

Training pause = stage-1 only. Total save time = stage-1 + stage-2, but stage-2 is invisible to the training loop. The trick: the host buffer lives in pinned memory so the GPU→host copy doesn't go through the GPU's main copy stream blocking next step's ops.

The DCP API (`torch.distributed.checkpoint.async_save`) handles both stages:

```python
import torch.distributed.checkpoint as dcp

state = {"model": model.state_dict(), "optim": optim.state_dict()}
fut = dcp.async_save(state, checkpoint_id="ckpt/step_1000")

# training continues here
for step in range(...):
    ...
    if fut.done():
        # save completed
        ...
```

`async_save` returns a future. Underlying mechanism: a dedicated background thread per rank, with a global Future synchronizing completion across ranks.

## Sharded format

DCP writes one file per rank:

```
ckpt/step_1000/
    .metadata
    __0_0.distcp
    __1_0.distcp
    __2_0.distcp
    ...
```

Each rank writes its own shards. Loading is similarly sharded — you can load on a different world size and DCP redistributes (within the constraints of the sharding strategy).

## Peer replication

For ultra-fast resumption, write to peer HBM in addition to (eventually) durable storage. [Gemini (USENIX ATC '23)](https://www.usenix.org/conference/atc23/presentation/wang-zhuang) and [ByteCheckpoint (NSDI '25)](https://www.usenix.org/conference/nsdi25/presentation/wan) implement this.

```
every 5 minutes:   peer-HBM checkpoint  (cheap, fast)
every 1 hour:      durable checkpoint   (slow, persistent)
```

On rank failure, recovery reads from peer HBM if the failure is recent — milliseconds. From durable storage if older — seconds.

In 2026, peer replication is becoming a standard option in training stacks. torchtitan exposes it via experimental config.

## What goes in the checkpoint

- Model parameters (sharded — DTensor metadata preserves the placement)
- Optimizer state (Adam moments, etc. — also sharded)
- LR scheduler state
- Step counter
- DataLoader sample-cursor (Mosaic StreamingDataset elastic-determinism makes this work)
- RNG state (per rank)

DCP handles the first two natively. The rest you stuff in the same `state` dict.

## Common bugs

- **Saving model.module.state_dict() instead of model.state_dict() under DDP/FSDP**: doesn't include the wrapper info you need for elastic resume.
- **Forgetting `dcp.async_save` returns a future**: if you don't `.wait()` on it before the next save, you can race on the host buffer.
- **Checkpointing too rarely**: cheap to save async. Save every 100–500 steps. The cost is ~10ms training pause.
- **Checkpointing without DataLoader cursor**: resume restarts from step 0 of the dataloader. Repeat-data symptom: loss curve has a periodic "low-then-bounce" pattern.

## Build steps

1. Take your torchtitan run from Topic 10. Confirm `async_mode = "async"` in the config.
2. Time the save: Python `time.time()` before `async_save` and immediately after. That is the training-pause time. Should be tens of ms.
3. Wait for the future to complete and time again. That is the total save time. Will be seconds-minutes depending on disk and model size.
4. Verify resumability: kill the run mid-save. Restart. Confirm the previous checkpoint loads cleanly (in-progress save was discarded, prior save is intact).

## Reference

- DCP docs: [pytorch.org/docs/stable/distributed.checkpoint.html](https://pytorch.org/docs/stable/distributed.checkpoint.html)
- DCP async_save: [pytorch.org/tutorials/recipes/distributed_async_checkpoint_recipe.html](https://pytorch.org/tutorials/recipes/distributed_async_checkpoint_recipe.html)
- Gemini paper: [usenix.org/conference/atc23/presentation/wang-zhuang](https://www.usenix.org/conference/atc23/presentation/wang-zhuang)
- ByteCheckpoint paper: [usenix.org/conference/nsdi25/presentation/wan](https://www.usenix.org/conference/nsdi25/presentation/wan)
- Goodput math: [cloud.google.com/blog/products/ai-machine-learning/elastic-training-and-optimized-checkpointing-improve-ml-goodput](https://cloud.google.com/blog/products/ai-machine-learning/elastic-training-and-optimized-checkpointing-improve-ml-goodput)
