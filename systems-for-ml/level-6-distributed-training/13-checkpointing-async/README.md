# 13 — Async Checkpointing

## Files

- `CONCEPTS.md` — why sync hurts, two-stage offload, peer replication, what goes in the checkpoint
- `async_dcp_demo.py` — sync vs async DCP timing, reports training-pause vs total save time

## Quickstart

```bash
torchrun --standalone --nproc_per_node=2 async_dcp_demo.py
```

## Expected output

```
sync  save: 1240.3 ms (training paused entire time)
async save: pause 28.4 ms; total 1290.1 ms
speedup on training-pause: 43.6x
```

The sync save and the async-total are roughly the same — same data, same disk. The training-pause is the meaningful number. 28ms instead of 1240ms means you can checkpoint 50× more often without losing throughput.

## Try

- Increase the model size (more `nn.Linear` blocks). Both numbers grow; the speedup on training-pause stays roughly constant.
- Move the checkpoint to a different filesystem (e.g., `/tmp` on tmpfs vs an NFS mount). The async-total drops dramatically; the training-pause stays the same.
- Save inside a training loop and measure step-time impact with vs without async. Should be unmeasurable with async; visible bump with sync.

## Build steps for full G-set

1. Run a 200-step training loop.
2. Checkpoint sync every 50 steps. Plot step-time histogram. Spikes at 50/100/150/200.
3. Checkpoint async every 50 steps. Plot step-time histogram. No spikes.

## Where this goes

- Topic 12 — failure injection uses async DCP for fast recovery
- Topic 14 — Ray Train wraps this with a job-orchestration layer
- Project 3 — async DCP is part of the deliverable
