# 00 — Collectives and NCCL

## Files

- `CONCEPTS.md` — the four collectives, ring vs tree, NCCL 2.27 features, hang-debug guide
- `collectives_demo.py` — runs all_reduce / all_gather / reduce_scatter / all_to_all, plus a 256 MiB bandwidth measurement
- `hang_demo.py` — deliberately hangs by mismatching shapes; read with FlightRecorder
- `run.sh` — torchrun wrapper with NCCL_DEBUG=INFO

## Quickstart

```bash
bash run.sh
```

You need at least 2 GPUs. Colab Pro 2×T4, RunPod 2×A10, any DGX, or an 8×H100 box all work.

## Expected output (2×A10, NVLink between them)

```
NCCL INFO Channel 00 : 0[3000] -> 1[44000] via NVL/0
NCCL INFO Channel 01 : 0[3000] -> 1[44000] via NVL/1
[all_reduce] rank0 sees: [3.0, 3.0, 3.0, 3.0]
[all_gather] rank0 sees: [[0.0, 0.0], [1.0, 1.0]]
[reduce_scatter] rank0 owns chunk: [2.0, 4.0]
[reduce_scatter] rank1 owns chunk: [4.0, 6.0]
[all_to_all] rank0 received: [0.0, 10.0]
[all_to_all] rank1 received: [1.0, 11.0]
[bw] 256 MiB all_reduce: 5.34 ms  algo-bw 47.9 GB/s
```

If you see `via NET/Socket` on a multi-node run, you have lost RDMA. Investigate before training a real model.

## Try

- `NCCL_ALGO=Tree bash run.sh` — force tree on a 2-rank job. Bandwidth will be similar at 256 MiB; the win is at small messages.
- `NCCL_P2P_DISABLE=1 bash run.sh` — disable peer-to-peer. Bandwidth drops dramatically; you are seeing the SHM fallback.
- `NCCL_DEBUG_SUBSYS=NET bash run.sh` on a multi-node setup — see whether you are getting `GDRDMA` or staging through host memory.
- Run `hang_demo.py` and let it sit for 10 minutes with `TORCH_NCCL_DUMP_ON_TIMEOUT=1`. Open the dumps and identify the mismatched collective.

## Where this goes

Topic 00b opens the transport layer beneath NCCL (RDMA, GPUDirect, NIXL). Topic 02 turns these primitives into a working DDP loop.
