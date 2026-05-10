# 01 — Interconnects

The 2026 fabric, the bandwidth numbers that drive parallelism choices, and the rail-aware topology that keeps NCCL flows from interfering.

## The hierarchy

```
   ┌────────────── one node ──────────────┐
   │  GPU0 ───NVLink5/NVSwitch─── GPU1     │   1.8 TB/s/GPU bidirectional
   │   │  ╲     1.8 TB/s         │         │   intra-NVLink-domain
   │   │   ╲                     │         │
   │  GPU2 ───────────────────── GPU3      │
   └──┬───────────────────────┬───────────┘
      │                       │
      │ rail 0 ──── IB XDR ────┤  800 Gb/s/NIC inter-node
      │ rail 1 ──── IB XDR ────┤  one NIC per GPU = 8 rails on 8-GPU node
      │ ...                    │
      ▼                       ▼
    ┌── leaf switches ──┐   ┌── leaf switches ──┐
            │                       │
            └───── spine ───────────┘
```

## Bandwidth table (2026)

| Layer | Hardware | Bandwidth |
|---|---|---|
| Intra-NVLink-domain | NVLink 5 + NVSwitch (Hopper/Blackwell) | 1.8 TB/s/GPU bidirectional |
| Extended NVLink domain | GB200/GB300 NVL72 | 130 TB/s aggregate, 72 GPUs in one NVLink domain |
| Inter-node IB | InfiniBand Quantum-X800 (XDR) | 800 Gb/s per port |
| Inter-node Ethernet | Spectrum-X 800GbE / Ultra Ethernet | 800 Gb/s, RoCEv2 |
| Cluster-wide rail-only | rail-aligned IB or Ethernet | 8× 800 Gb/s per node |

The intra-NVLink-domain to inter-node ratio is roughly 18:1 — one NVLink hop is 18× the bandwidth of one IB-XDR hop. That ratio is the single most important number when picking parallelism axes.

## NVL72 — what changed

Pre-NVL72: NVLink stayed inside one chassis. Crossing chassis meant going to IB.

NVL72 (GB200): 72 GPUs sit in one NVLink domain, connected by NVLink 5 + NVSwitch silicon spread across the rack. Aggregate 130 TB/s. From a software standpoint the entire 72-GPU rack acts like a single huge node.

Implication for parallelism: a 70B model that used to need TP + PP across IB now fits TP + FSDP entirely intra-NVLink. Step time drops accordingly. The reason TP=8 was the historical ceiling was the NVLink domain size; with NVL72 the ceiling is 72.

## Rail-optimized topology

Each GPU on a node has its own NIC. Each NIC connects to a separate "rail" — a parallel switch fabric that does not share switches with other rails. NCCL pins a flow to one rail to avoid cross-rail interference.

```
node A             node B
GPU0 → NIC0 ─── rail-0 leaf ─── NIC0 → GPU0
GPU1 → NIC1 ─── rail-1 leaf ─── NIC1 → GPU1
GPU2 → NIC2 ─── rail-2 leaf ─── NIC2 → GPU2
...
```

`NCCL_CROSS_NIC=0` enforces strict same-rail; `=2` allows cross-rail when needed (some collectives benefit). At frontier scale, **rail-only networks** drop the inter-rail switching layer entirely — you cannot send rail-0 → rail-1 because the switches do not connect. Cuts cost; constrains software to map collectives along rail boundaries.

Standard vocabulary in 2026 networking-aware ML systems work. If you read "rail-aligned all-reduce" in a paper, this is what they mean.

## Why this drives parallelism choices

- **TP**: per-layer all-reduce, hot path of every forward and backward. Latency-sensitive. Must stay intra-NVLink. Hence `TP ≤ 8` on Hopper, `TP ≤ 72` on NVL72 in principle.
- **DP/FSDP**: per-step communication (reduce-scatter / all-gather). Tolerates IB. Fine across nodes.
- **PP**: send/recv per microbatch boundary. Tolerates IB if microbatches are large. Often crosses node boundaries.
- **EP**: all-to-all per MoE layer. Wants to stay intra-NVLink ideally; tolerates IB at the cost of step time.
- **CP**: ring-attention sends per attention block. Latency-sensitive; usually stays intra-node.

The rule: high-frequency collectives go on high-bandwidth links. Low-frequency ones can cross to IB. This is the entire 5D parallelism placement story compressed.

## Measuring what you have

You will not have NVL72 on Colab. You can still measure the delta between fast and slow paths:

```bash
# Fast path
torchrun --nproc_per_node=2 ../00-collectives-and-nccl/collectives_demo.py

# Force slow path
NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 \
    torchrun --nproc_per_node=2 ../00-collectives-and-nccl/collectives_demo.py
```

The bandwidth gap between the two is your local NVLink/PCIe vs host-staging delta. The same multiplier applies, scaled up, between intra-NVLink-domain and inter-node-IB traffic on real clusters.

## G10 of Project 3

Training throughput vs interconnect type. On 2 GPUs:
- Run a 100M-param FSDP step at NVLink/PCIe full bandwidth.
- Run the same with `NCCL_P2P_DISABLE=1`.
- Plot tokens/sec for each. The gap is the interconnect tax.

You won't see real NVL72 numbers without renting Blackwell. But the *shape* of the curve (linear scaling minus a comms tax that grows with model size) is the same.

## Reference

- NVLink 5 / NVL72: [nvidia.com/en-us/data-center/gb200-nvl72](https://www.nvidia.com/en-us/data-center/gb200-nvl72/)
- Quantum-X800: [nvidia.com/en-us/networking/quantum-x800](https://www.nvidia.com/en-us/networking/quantum-x800/)
- Spectrum-X / Ultra Ethernet: [ultraethernet.org/specifications](https://ultraethernet.org/specifications/)
- Rail-optimized topology paper (HotInfra '23): [arxiv.org/abs/2307.12169](https://arxiv.org/abs/2307.12169)
- NCCL topology + rails: [docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-cross-nic](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-cross-nic)
