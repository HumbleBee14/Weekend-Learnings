# 00b — RDMA, GPUDirect, NIXL

The transport layer underneath NCCL all-reduce and underneath KV-cache transfer in disaggregated inference. The same primitives power both, which is why this single topic spans training-side checkpointing, inter-node collectives, and inference-side prefill→decode handoff.

## RDMA in one paragraph

Remote Direct Memory Access. A NIC reads/writes remote host memory without involving the remote CPU. After queue-pair (QP) setup the kernel is bypassed entirely on the data path. Latency drops from ~10 µs (TCP) to ~1 µs. Bandwidth approaches line-rate (400/800 Gb/s on modern fabrics). The programming model is "verbs": post a send/recv to a queue, poll a completion queue (CQ). Different from sockets but the abstraction NCCL builds on.

```
TCP send                           RDMA send
─────────                          ────────────
app → kernel buffer                app posts WR to QP
kernel → NIC (sk_buff)             NIC DMAs from app memory
NIC → wire                         NIC → wire
remote NIC → kernel                remote NIC DMAs to app memory
kernel → app (recv)                NIC posts CQE
                                   app polls CQE
                                   (no remote CPU involvement)
```

## GPUDirect RDMA

Standard RDMA targets host (DRAM) buffers. GPUDirect RDMA registers GPU memory as the source/destination. The NIC and GPU talk over PCIe (or NVLink in newer systems via NVLink-C2C) without staging through host DRAM.

```
Without GPUDirect RDMA               With GPUDirect RDMA
──────────────────────               ──────────────────────
GPU HBM                              GPU HBM
  ↓ cudaMemcpy                         ↓
host DRAM (pinned)                   NIC reads HBM directly via PCIe BAR
  ↓ ibv_post_send
NIC                                  NIC → wire
  ↓
wire
```

That extra hop is the difference between 200 GB/s effective and ~25 GB/s. Without GPUDirect RDMA, every inter-node tensor moves through host memory twice (once on send, once on receive).

NCCL detects GPUDirect at init. With `NCCL_DEBUG=INFO` look for:

```
NCCL INFO NET/IB : Using [0]mlx5_0:1/IB ; OOB ib0:10.0.0.1<0>
NCCL INFO Channel 00 : 4 -> 0 [send] via NET/IB/0/GDRDMA
```

`GDRDMA` confirms the path. Without it you would see only `via NET/IB`.

## The transport hierarchy NCCL picks from

| Locality | Transport | NCCL tag |
|---|---|---|
| Same NVLink domain | NVLink, optionally with NVSwitch SHARP | `NVL`, `NVLS` |
| Same node, separate PCIe | GPUDirect P2P over PCIe | `P2P/IPC` |
| Same node, P2P unavailable | Shared host memory + cudaMemcpyAsync | `SHM` |
| Different nodes, IB available | GPUDirect RDMA over IB/RoCE | `NET/IB/GDRDMA` |
| Different nodes, no GDR | RDMA but staged through host | `NET/IB` |
| Different nodes, no IB | TCP sockets | `NET/Socket` |

Going from `NVL` to `NET/Socket` is roughly four orders of magnitude in bandwidth. This is why interconnect awareness is so heavily woven into the parallelism-axis decisions in Topic 09.

## RoCE vs InfiniBand

Both speak verbs. Both can do GPUDirect RDMA. They differ in the underlying L2:

- **InfiniBand**: dedicated fabric, native verbs, deterministic latency, requires Subnet Manager. Quantum-X800 is the 2026 NVIDIA flagship at 800 Gb/s.
- **RoCEv2**: verbs over UDP/IP on Ethernet. Spectrum-X 800GbE is the NVIDIA Ethernet equivalent. Ultra Ethernet Consortium (UEC) standardized lossless-ish Ethernet for AI in 2024–2025.

NCCL works on either. The `NET_PLUGIN` resolves the right verbs path. ML systems work in 2026 increasingly happens on RoCE because Ethernet is cheaper and the consortium-standardized congestion control closed most of the IB latency gap.

## NIXL — the 2025 inference-side transfer library

[NVIDIA Inference Xfer Library](https://github.com/NVIDIA/NIXL) (GTC 2025). Sits above NCCL/UCX/RDMA. Exposes a higher-level API tuned for *unicast point-to-point* transfers of variable-size buffer lists.

Why it exists when NCCL already exists:
- NCCL collectives assume every rank participates. KV-cache transfer in disaggregated inference is one prefill worker → one decode worker. Not a collective.
- NCCL assumes static, registered buffers. KV blocks are allocated dynamically per request.
- NCCL has high per-call overhead optimized away by amortization across thousands of ranks. NIXL is optimized for many small transfers per second.

Used by: NVIDIA Dynamo, llm-d, vLLM disaggregated mode, Ray Serve LLM. Internally talks UCX (which talks verbs), with optimized paths for `cuda_ipc` (intra-node) and GPUDirect RDMA (inter-node).

Architecture:

```
   Application (vLLM, Dynamo, llm-d)
   ┌──────────────────────────────────┐
   │  NIXL agents + memory registry   │
   └────┬─────────┬─────────┬─────────┘
        │         │         │
     UCX       cuda_ipc   POSIX
        │
     verbs (RDMA)
```

A "register memory" call hands NIXL a span of HBM. A "transfer" call says "send this list of (src_offset, dst_offset, len) triples to that other agent." Under the hood it batches into one or a few RDMA writes.

## Why this topic appears in a training curriculum

Two reasons.

1. **Inter-node collectives.** Every NCCL all-reduce that crosses a node boundary is RDMA + GPUDirect underneath. Understanding the fast path is required for diagnosing a slow training step where the comms-overlap math says it should be faster.

2. **Checkpoint streaming.** Async DCP (Topic 13) writes shards to peer HBM and to durable storage. Peer-to-peer HBM transfer at speed = NIXL or direct GPUDirect RDMA. ByteCheckpoint (NSDI '25) and Gemini (USENIX) both use peer replication to amortize checkpointing.

For the disaggregated-inference angle, the same transport ships KV blocks from prefill GPU's HBM to decode GPU's HBM. Level 5 Topic 08 mentioned this abstractly; here is the mechanism.

## You will not write RDMA code this week

Verbs programming is specialist. The point of this topic is reading-level fluency: when NCCL prints `via NET/Socket` instead of `via NET/IB/GDRDMA`, you know what was lost; when a vLLM disaggregated deploy is slow, you know to look at NIXL counters before the model.

## Build steps (mostly reading)

1. Run any multi-GPU job with `NCCL_DEBUG=INFO`. Find the transport line for inter-rank channels. Identify NVL vs P2P/IPC vs NET.
2. Read NIXL repo: [github.com/NVIDIA/NIXL](https://github.com/NVIDIA/NIXL). Skim the README and the "How NIXL works" section.
3. Read [GPUDirect RDMA architecture docs](https://docs.nvidia.com/cuda/gpudirect-rdma/).
4. Read the llm-d disaggregated-inference design notes for how NIXL is used in practice: [llm-d/llm-d](https://github.com/llm-d/llm-d).
5. Write a 200-word note: "How does a KV block move from prefill worker's HBM to decode worker's HBM in a disaggregated vLLM setup?" Trace the path through NIXL → UCX → verbs → GPUDirect RDMA.

## Reference

- NCCL transport layer: [docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)
- GPUDirect RDMA: [docs.nvidia.com/cuda/gpudirect-rdma/](https://docs.nvidia.com/cuda/gpudirect-rdma/)
- NIXL repo: [github.com/NVIDIA/NIXL](https://github.com/NVIDIA/NIXL)
- NIXL announcement (GTC 2025): [developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/)
- UCX: [openucx.org](https://openucx.org/)
- RDMA verbs intro: [rdmamojo.com/2013/01/26/ibv_post_send](http://www.rdmamojo.com/2013/01/26/ibv_post_send/)
- Ultra Ethernet Consortium: [ultraethernet.org](https://ultraethernet.org/)
