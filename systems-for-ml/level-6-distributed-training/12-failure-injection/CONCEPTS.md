# 12 — Failure Injection

The 1024-node pattern: at any meaningful scale, hardware breaks during training. Hours of compute lost per failure if you're not engineered for it. The 2026 toolkit is async DCP + elastic launch + NCCL Communicator Shrink.

## What fails

- **GPU**: ECC-uncorrectable, XID 13/31/79 errors, thermal events that don't recover.
- **NIC / cable**: link flap, RoCE pause storms, IB subnet manager loses sight of a port.
- **Node**: kernel panic, OOM-killer, BMC reset, power event.
- **Storage**: object store throttle, NFS mount lost.
- **Software**: OOM in the training loop, deadlock in the dataloader, dependency upgrade rollout.

Mean time between failures at frontier scale (10K+ GPUs) is hours. At your home cluster (2-8 GPUs) it's days/weeks — but the *patterns* you learn here are what you'll apply at scale later.

## The toolkit

### Async DCP (`torch.distributed.checkpoint`)

Sharded per-rank writes. Two-stage offload: GPU → pinned host buffer (fast, blocks training briefly), pinned host → object storage (slow, runs in a background thread). Topic 13 covers the API.

### Elastic launch (`torchrun --rdzv`)

```bash
torchrun \
    --nnodes=2:8 \                     # min 2, max 8 — elastic range
    --nproc_per_node=8 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=node0:29500 \
    --rdzv_id=run42 \
    --max-restarts=10 \
    train.py
```

Workers join via the rendezvous (c10d-backed). On a worker exit, surviving workers re-rendezvous, the framework calls back to the user code (`on_membership_change`) so it can re-shard the model and resume. With NCCL Communicator Shrink, the survivors don't even need to fully tear down — they shrink the comm and continue.

### NCCL Communicator Shrink (NCCL 2.27+)

```c
ncclCommShrink(parent_comm, ranks_to_drop, num_ranks_to_drop, &new_comm, NCCL_SHRINK_DEFAULT);
```

Two modes:
- `NCCL_SHRINK_DEFAULT` — planned reconfig. The dropped rank participates in the shrink call. Used for graceful drain.
- `NCCL_SHRINK_ABORT` — the dropped rank is dead/unresponsive. Survivors call shrink on their own. Used for fault recovery.

The new comm has fewer ranks and continues without the dead one. PyTorch exposes this via the elastic + NCCL path; on a multi-rank job, when the elastic agent detects a worker exit, it triggers `NCCL_SHRINK_ABORT` on the underlying NCCL comm so the survivors don't hang.

### Peer replication / in-memory checkpoints

[Gemini (USENIX ATC '23)](https://www.usenix.org/conference/atc23/presentation/wang-zhuang) and [ByteCheckpoint (NSDI '25)](https://www.usenix.org/conference/nsdi25/presentation/wan) both keep the *recent* state in peer HBM (much faster to re-fetch than from object storage), with durable storage hit only every ~1 hour. Restart cost goes from minutes (read from S3) to seconds (read from peer HBM).

## The recovery flow

```
t=0       8 ranks training
t=10min   GPU 3 throws XID 79
          NCCL collective on rank 3 fails
          watchdog timer fires on ranks 0-2,4-7
          elastic agent detects rank 3 exit
          → calls ncclCommShrink with NCCL_SHRINK_ABORT, dropping rank 3
          → re-creates DeviceMesh at world=7 (or marks rank 3 spot empty)
          → reloads from last DCP checkpoint into the new world layout
          → resumes
t=10min+~30s  training continues at world=7
```

The "30 seconds" is the goal. To hit it:
- Async DCP must have a recent checkpoint.
- The framework must support world-size re-sharding on load (FSDP2 + DCP does; FSDP1 didn't cleanly).
- The elastic launcher must successfully signal the surviving agents.
- The NCCL shrink must succeed (it does, deterministically, in 2.27+).

## What you cannot recover from cleanly

- **TP rank loss within an NVLink domain.** TP requires every TP rank to participate. Losing one means the layer cannot run. You drop the entire TP group → whole node. If your job is `TP=8 × DP=8`, losing one GPU means losing one entire DP slot, not one TP rank.
- **PP rank loss.** Same — pipeline stages cannot be skipped. Lose a PP rank, lose the entire stage. Effectively lose `TP_size × CP_size × EP_size` GPUs per failed rank.
- **Last DP shard with critical params** — handled by FSDP2 since every shard exists on at least one rank, but if your sharding strategy doesn't replicate, you can't recover.

The practical rule: failures only recover cleanly along the DP/FSDP axis. TP/PP/EP/CP group-failures cost you the whole group.

## DP-axis recovery via Comm Shrink

When a rank goes down in the DP dimension:
1. Survivors detect via NCCL watchdog.
2. Elastic agent issues `NCCL_SHRINK_ABORT`.
3. Reduce-scatter / all-gather collectives now run at `dp_size - 1` ranks.
4. Optimizer's effective batch size shrinks proportionally (you may scale lr or just accept the smaller effective batch).
5. Training continues.

## Goodput accounting

Pre-failure: 8 GPUs × 4 hours = 32 GPU-hours useful.
Failure happens, recovery takes 5 minutes, run continues at 7 GPUs.

Goodput math:
- Wallclock used: 4h0min0s + 5min recovery = 4h5min
- GPU-hours billed: 8 × 4h5min = 32.67
- Useful GPU-hours: 4h × 8 + (next slice) × 7

The percentage delta between failed-with-recovery and failure-free is small if recovery is fast. Async DCP makes recovery fast. If recovery requires reading from S3 (sync DCP), you lose 10–60 minutes per failure — that's the goodput killer.

## Build steps

1. Train your torchtitan model from Topic 10 with `async_mode = "async"` enabled in the config.
2. Mid-training, kill one of the worker processes: `kill -9 <pid>` from another shell, or use `failure_inject.sh`.
3. Verify behavior:
   - Surviving ranks log a NCCL watchdog timeout.
   - Elastic agent (if `--max-restarts > 0`) restarts the dead worker and resumes from checkpoint.
   - Or, with manual Comm Shrink, training continues at smaller world.
4. Document time-to-recovery in `reports/`.

## Reference

- NCCL 2.27 Comm Shrink blog: [developer.nvidia.com/blog/enabling-fast-inference-and-resilient-training-with-nccl-2-27](https://developer.nvidia.com/blog/enabling-fast-inference-and-resilient-training-with-nccl-2-27/)
- torchrun elastic: [pytorch.org/docs/stable/elastic/run.html](https://pytorch.org/docs/stable/elastic/run.html)
- TorchElastic design: [pytorch.org/docs/stable/elastic/design.html](https://pytorch.org/docs/stable/elastic/design.html)
- Gemini (USENIX ATC '23): [usenix.org/conference/atc23/presentation/wang-zhuang](https://www.usenix.org/conference/atc23/presentation/wang-zhuang)
- ByteCheckpoint (NSDI '25): [usenix.org/conference/nsdi25/presentation/wan](https://www.usenix.org/conference/nsdi25/presentation/wan)
- Goodput: [cloud.google.com/blog/products/ai-machine-learning/elastic-training-and-optimized-checkpointing-improve-ml-goodput](https://cloud.google.com/blog/products/ai-machine-learning/elastic-training-and-optimized-checkpointing-improve-ml-goodput)
