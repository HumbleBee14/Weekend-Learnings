# 01 — Interconnects

## Files

- `CONCEPTS.md` — NVLink 5 / NVL72 / IB-XDR / Ultra Ethernet, rail-optimized topology, why parallelism axes pin to specific links
- `bw_matrix.py` — sweeps all-reduce bandwidth from 1 KiB to 1 GiB; runs once for fast path, once with `NCCL_P2P_DISABLE=1` for slow path. Produces `bw_fast.csv` and `bw_slow.csv` for G10.

## Quickstart

```bash
torchrun --standalone --nproc_per_node=2 bw_matrix.py
NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 \
    torchrun --standalone --nproc_per_node=2 bw_matrix.py
```

## Expected output (2×A10, NVLink)

```
fast     0.001 MiB    0.043 ms     0.05 GB/s
fast     0.010 MiB    0.044 ms     0.46 GB/s
fast     1.000 MiB    0.062 ms    16.79 GB/s
fast    64.000 MiB    1.870 ms    35.71 GB/s
fast  1024.000 MiB   28.420 ms    37.62 GB/s

slow     1.000 MiB    0.621 ms     1.69 GB/s
slow  1024.000 MiB   86.110 ms    12.42 GB/s
```

The asymptote is your bus bandwidth ceiling. Small messages are latency-bound (the curve is flat-low at the left). The crossover from latency-bound to bandwidth-bound is where ring vs tree algorithm choice matters.

## Try

- Plot both CSVs on a log-log axis (size vs bandwidth). The shape of the curve teaches more than the absolute number.
- Set `NCCL_ALGO=Tree` and rerun. Watch the small-message latency drop while large-message bandwidth flattens.
- On a multi-node box, rerun with `NCCL_DEBUG=INFO` and confirm `via NET/IB/0/GDRDMA` in the log. If you see `via NET/Socket`, your peers are not using RDMA.

## Where this goes

- Topic 02 turns these primitives into a DDP step
- Topic 04 (FSDP2) reduces *both* halves (all-gather + reduce-scatter), so doubles the comms traffic — the bandwidth ceiling here is the budget you spend
