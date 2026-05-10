# 12 — Failure Injection

## Files

- `CONCEPTS.md` — what fails, async DCP + elastic launch + Comm Shrink toolkit, what cannot recover cleanly, goodput accounting
- `elastic_train.py` — DDP loop with `dcp.load`/`dcp.save`, designed to be kill-9'd from another shell
- `failure_inject.sh` — launches the loop and kills one worker mid-run; watch recovery

## Quickstart

```bash
bash failure_inject.sh
```

You will see (roughly):

```
[pid 12345] step  0  loss 1.024
[pid 12345] step  5  loss 0.812
checkpoint saved at step 20
killing worker pid 12347
[NCCL] watchdog: collective took longer than 600s timeout
[E] worker rank 1 exited with signal 9
[INFO] elastic agent: re-rendezvous, attempting restart
[INFO] worker rank 1 restarted
resumed from checkpoint at step 20
[pid 12348] step 25  loss 0.640
...
```

The "resumed from checkpoint at step 20" line is the recovery completing. Time-to-recovery from kill to next training step is the metric.

## Try

- Set `--max-restarts=0` and rerun. The whole job dies. This is what happens without elastic launch.
- Increase the checkpoint interval to 100 steps. Re-kill mid-run. Now you lose more progress per failure — see how checkpoint frequency trades off against goodput.
- Kill rank 0 instead of rank 1. NCCL watchdog should still fire on the survivor; behavior identical.
- On a multi-node setup, kill an entire node. The c10d rendezvous re-forms with `nnodes=1` (within the elastic range).
- Compare time-to-recovery with and without async DCP (Topic 13). The async case has fresher checkpoints → less work to redo.

## Build steps for full G11/G12

1. Run a 200-step training loop.
2. Kill rank 1 at step 50.
3. Time recovery: `t_kill → t_first_step_after_resume`.
4. Document in `reports/training.md`. Plot p99 step time across the run with the failure marker.

## Notes on real production

- TorchElastic's c10d rendezvous works for single-node and small multi-node. Frontier-scale uses Kubernetes-native rendezvous (KubeRay, etcd-based) — same model, different rendezvous backend.
- True NCCL Comm Shrink (no full restart, just shrink the communicator) requires NCCL 2.27+ and the appropriate framework wiring. PyTorch is rolling support across 2026.

## Where this goes

- Topic 13 — async DCP variants of `dcp.save` here cut the checkpoint pause to ~10ms
- Topic 14 — Ray Train wraps this elastic + DDP pattern with a job-orchestration layer
- Project 3 G11/G12 use this script as the harness
