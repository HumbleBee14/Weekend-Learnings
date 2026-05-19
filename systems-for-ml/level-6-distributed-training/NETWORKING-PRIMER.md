# Networking Primer for ML Systems

> Read this before Topic 00 if you're coming from a pure-Python / single-machine background. If you've ever debugged a slow `scp`, an `iperf3` run, or a Kafka cluster, you can skip this.

Distributed training is mostly a networking problem dressed in PyTorch syntax. NCCL, RDMA, GPUDirect, NVLink — these are the actual variables that decide whether your 70B run trains at 400 TFLOPS/GPU or 80. This primer gets you to literate in ~20 minutes.

## The four numbers you actually need

Every link in a cluster has four properties. The rest is detail.

| Property | What it means | Typical units | Why ML cares |
|---|---|---|---|
| **Bandwidth** | Bytes per second the link can move at steady state | GB/s | Sets how fast `all-reduce` finishes on big tensors |
| **Latency** | Time before the *first byte* arrives after sending | µs | Sets the floor for small-tensor collectives and tail latency |
| **Topology** | Who is connected to whom, in what shape | (ring / fat-tree / dragonfly) | Decides whether `all-reduce` can use ring algorithm or has to hop |
| **CPU involvement** | Does the OS kernel touch every byte? | (yes / no — "kernel bypass") | Decides whether you saturate the link or hit ~10% of theoretical |

Internalize the **bandwidth ÷ latency** intuition: for a 1 GB tensor at 200 GB/s, the wire takes 5 ms. For a 4 KB control message, latency dominates (the wire takes nanoseconds, but the round-trip is microseconds). Collectives on big weights are bandwidth-bound; gradient sync on small parameters is latency-bound. This is why FSDP shards on parameter granularity — to make every reduction bandwidth-bound.

## The hierarchy in 2026

```
        Fastest, smallest scope ────────────────────► Slowest, largest scope
        ┌────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
        │ HBM    │   │ NVLink   │   │ NVL72    │   │ InfiniB. │   │ Ethernet │
        │ on-die │   │ same node│   │ rack     │   │ same DC  │   │ WAN      │
        ├────────┤   ├──────────┤   ├──────────┤   ├──────────┤   ├──────────┤
        │ 5 TB/s │   │ 900 GB/s │   │ 130 GB/s │   │ 50 GB/s  │   │ 12 GB/s  │
        │ 100 ns │   │ 1 µs     │   │ 2 µs     │   │ 5 µs     │   │ 100 µs   │
        └────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘
        ◄────── 50× ──────► ◄── 7× ──► ◄── 3× ──► ◄────── 4× ──────►
```

Each step is roughly an order of magnitude slower. **You design distributed training to keep traffic at the highest level possible** — TP inside a node (NVLink), DP across nodes (InfiniBand), PP between DC zones if you must. If your collective is hitting Ethernet, something has gone wrong.

## TCP vs RDMA — the single biggest production lever

Standard TCP:
```
GPU memory → CPU memory → kernel TCP stack → NIC → wire → NIC → kernel → CPU → GPU
            (memcpy)     (syscalls, checksums)            (reverse)        (memcpy)
```
Every byte is touched by the CPU. Throughput caps around 25–40 GB/s on the fastest NICs because the kernel becomes the bottleneck. Latency: 30–100 µs per hop.

RDMA (Remote Direct Memory Access):
```
GPU memory → NIC → wire → NIC → GPU memory
            (zero CPU involvement, "kernel bypass")
```
The NIC reads/writes GPU memory directly. Throughput: line rate (200–400 Gb/s on InfiniBand XDR). Latency: 1–5 µs. **GPUDirect RDMA** is the specific NVIDIA flavor that skips even the host-memory bounce.

**Rule of thumb:** on a multi-node training run with TCP, you'll do ~30% of the speed you'd get with RDMA. The cost difference is small at any production scale.

## NCCL — what's actually happening

NCCL is NVIDIA's collective communication library. It implements `all-reduce`, `all-gather`, `reduce-scatter`, `broadcast`, etc. *on top of* whatever transport is available (NVLink, RDMA, TCP fallback). Picture it as the "BLAS of distributed training" — you don't call wire protocols directly, you call NCCL ops and NCCL picks the right algorithm.

Two algorithms you'll meet in Topic 00:

```
RING all-reduce:        TREE all-reduce:
  GPU 0 ──► GPU 1                GPU 0
              │                  /     \
  GPU 3 ◄── GPU 2             GPU 1   GPU 2
                              /  \      \
                           GPU 3 GPU 4   GPU 5
```

- **Ring** = optimal bandwidth for large tensors (each GPU sends `(N-1)/N` of the data exactly once). Good when the tensor is big enough to amortize the ring traversal.
- **Tree** = optimal latency for small tensors (log(N) hops instead of N). Good for control messages and tiny gradients.

NCCL picks for you. You learn to read its choice in profiles.

## GPUDirect — the three flavors

| Variant | What it bypasses | When it kicks in |
|---|---|---|
| **GPUDirect P2P** | CPU bounce between GPUs *on the same node* | Multi-GPU on one box, when NVLink is wired |
| **GPUDirect RDMA** | Host memory entirely, GPU↔NIC↔GPU across nodes | Multi-node with RDMA-capable NICs |
| **GPUDirect Storage** | Host memory between GPU and NVMe | Loading datasets, checkpoint streaming |

**Failure mode to recognize:** if your config is "wrong" (e.g., NIC and GPU on different NUMA nodes, or PCIe lanes shared), GPUDirect silently falls back to CPU-bounce mode and you lose ~5× throughput with no error. `nvidia-smi topo -m` shows you the actual topology.

## The four words that own this story

- **Latency** — first-byte time. Cares about distance, hops, kernel involvement.
- **Bandwidth** — steady-state byte rate. Cares about wire width, encoding efficiency.
- **Bisection bandwidth** — how much traffic crosses an imaginary cut through the cluster. Determines how many GPUs you can do `all-reduce` across without slowdown.
- **Tail latency** — the slowest link or slowest GPU in a synchronous collective sets the entire step time. One straggler stalls the whole world.

If you remember nothing else from this primer, remember: **synchronous distributed training is rate-limited by your slowest link and your slowest GPU.** Every framework optimization (FSDP overlap, ZeRO bucketing, PP scheduling) is ultimately trying to hide latency behind compute. You can't optimize what you don't understand the cost of.

## What to read next

- Topic 00 — `collectives-and-nccl` now makes sense: ring vs tree, bandwidth-bound vs latency-bound, why bucket size matters.
- Topic 00b — `rdma-gpudirect-nixl` is the transport underneath.
- Topic 01 — `interconnects` — NVLink 5, NVL72 rack-scale, IB XDR, Ultra Ethernet (what 2026 fabrics actually look like).
- [Reddi Vol 2 — *Communication*](https://mlsysbook.ai/) chapter — canonical textbook framing.

You don't need to memorize the numbers — you need the mental model that says *"this collective looks slow; is it bandwidth-bound or latency-bound; which step in the hierarchy is it hitting?"* That's all this primer exists for.
