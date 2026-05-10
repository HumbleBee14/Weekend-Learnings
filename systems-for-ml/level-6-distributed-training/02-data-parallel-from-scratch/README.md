# 02 — Data Parallel From Scratch

## Files

- `CONCEPTS.md` — what DDP communicates, gradient bucketing, what DDP does not fix
- `ddp_train.py` — minimal DDP loop on a tiny transformer; profiler-ready

## Quickstart

```bash
torchrun --standalone --nproc_per_node=2 ddp_train.py
PROFILE=1 torchrun --standalone --nproc_per_node=2 ddp_train.py
```

`PROFILE=1` writes a Chrome trace under `./trace/`. Open in [Perfetto](https://ui.perfetto.dev/) and look for `nccl:all_reduce` ranges on the comms stream overlapping with backward kernels.

## Expected output

```
step    0  loss 9.124  tok/s 14,200
step   10  loss 7.892  tok/s 41,800
step   20  loss 7.214  tok/s 43,100
step   30  loss 6.877  tok/s 43,200
step   40  loss 6.601  tok/s 43,400
step   50  loss 6.412  tok/s 43,500
```

Numbers will differ by GPU; what matters is loss going down and tok/s being roughly stable after warmup.

## Try

- Time `world_size=1` vs `=2`. Linear scaling is `2×`. The shortfall is the comms tax.
- Set `bucket_cap_mb=1` and re-time. More NCCL calls, less amortization. You will see throughput drop.
- Set `bucket_cap_mb=200`. Fewer calls but worse comms-compute overlap on backward. Often slower than the default.
- Set `static_graph=False`. Watch DDP do more work per step; small but measurable.
- Open the profiler trace and find one bucket's `all_reduce`. Confirm it overlaps with a backward kernel — that is the overlap working.

## Where this goes

- Topic 04 — FSDP2 — replaces DDP for models that don't fit per rank
- Topic 11 — straggler injection — re-uses this script as the workload
- Topic 12 — failure injection — kills a rank mid-DDP and recovers
