# 06 — Pipeline Parallelism

## Files

- `CONCEPTS.md` — bubble math, GPipe → 1F1B → interleaved → ZB-V → DualPipe schedules
- `pp_demo.py` — hand-rolled 2-stage pipeline; runs both GPipe-naive and 1F1B; reports the gap

## Quickstart

```bash
torchrun --standalone --nproc_per_node=2 pp_demo.py
```

## Expected output

```
GPipe naive : 124.3 ms
1F1B        : 96.7 ms
theoretical bubble (S=2, M=8): 11.1%
```

The 1F1B step is shorter because the steady-state region is more compact. With `S=2, M=8` the gap is small; bump stages and microbatches and the gap widens dramatically.

## Try

- Increase `n_microbatches` from 8 to 32. Watch both schedules' bubble fraction shrink.
- Decrease to `n_microbatches=2`. Now the bubble dominates — both schedules run roughly the same.
- Replace this hand-rolled code with `torch.distributed.pipelining.Schedule1F1B` and confirm the times match.
- Read `torch.distributed.pipelining.ScheduleZBVZeroBubble` source. The split of `dW`/`dX` is the part to focus on.

## Where this goes

- Topic 09 — composing PP with FSDP+TP for full 5D
- Topic 10 — torchtitan PP recipe; one config switch swaps the schedule
- Topic 12 — failure injection; PP makes recovery harder (a stage failure stalls the whole pipeline)
