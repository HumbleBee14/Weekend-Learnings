# 00b — RDMA, GPUDirect, NIXL

## Files

- `CONCEPTS.md` — RDMA verbs primer, GPUDirect RDMA, NCCL transport hierarchy, NIXL architecture
- `inspect_transport.sh` — runs a small NCCL job and extracts the chosen transport per channel from `NCCL_DEBUG=INFO` output

## Quickstart

This topic is mostly reading. The one runnable artifact:

```bash
bash inspect_transport.sh
```

On a 2-GPU NVLink box you will see:

```
Channel 00 : 0[3000] -> 1[44000] via NVL/0
Channel 01 : 0[3000] -> 1[44000] via NVL/1
```

On a 2-GPU PCIe-only box:

```
Channel 00 : 0[3000] -> 1[44000] via P2P/IPC
```

On a multi-node IB box (run with a real launcher, not standalone):

```
Channel 00 : 4 -> 0 [send] via NET/IB/0/GDRDMA
```

`GDRDMA` confirms GPUDirect RDMA is in play. If absent, every inter-node tensor stages through host DRAM.

## Try

- Force the slow path: `NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 bash inspect_transport.sh`. Now you should see `via SHM` or `via NET/Socket`. Compare the bandwidth in the parent topic's `bw` line.
- Check whether your kernel has `nvidia_peermem` loaded: `lsmod | grep nvidia_peermem`. Without it, GPUDirect RDMA degrades to host staging on most distros.
- Read the NIXL `transfer` example in `examples/nixl_python_example.py` from the [NIXL repo](https://github.com/NVIDIA/NIXL).

## Where this goes

- Topic 01 — interconnects — gives the bandwidth numbers your transport choice unlocks
- Topic 13 — async checkpointing — uses peer-HBM replication, the same transport story
- Level 5 Topic 08 — disaggregated inference — was the inference-side use of NIXL
