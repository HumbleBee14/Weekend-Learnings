# 00 — Collectives and NCCL

The four collectives that show up everywhere in distributed training, the two algorithms NCCL picks between, and the things that go wrong on a real cluster.

## The four collectives

```
all-reduce          all-gather         reduce-scatter      all-to-all
(every→sum)         (every→concat)     (every→split-sum)   (every→every-distinct)

R0: a   ─┐          R0: a   ─┐         R0: [a0,a1] ─┐      R0: [x0,x1]   ─┐
R1: b   ─┤→ a+b+c   R1: b   ─┤→[a,b,c] R1: [b0,b1] ─┤→ R0: a0+b0+c0      R1: [y0,y1]   ─┤→ R0:[x0,y0,z0]
R2: c   ─┘          R2: c   ─┘         R2: [c0,c1] ─┘    R1: a1+b1+c1    R2: [z0,z1]   ─┘   R1:[x1,y1,z1]
                                                            R2: ...                              ...
```

Where each shows up in 2026 training:

| Collective | Where |
|---|---|
| all-reduce | DDP gradient sync at end of backward |
| all-gather | FSDP forward (gather sharded weights), TP column-parallel output |
| reduce-scatter | FSDP backward (gradient reduction into shards) |
| all-to-all | EP token routing, sequence-parallel↔tensor-parallel transition |
| send/recv (point-to-point, not strictly collective) | PP stage handoff |

The mental model: FSDP step = `all_gather(weights) → forward → all_gather(weights) → backward → reduce_scatter(grads)`. DDP step = forward, backward, `all_reduce(grads)`. Different communication patterns, different bandwidth profiles.

## Ring all-reduce — the algorithm

Why ring is the bandwidth-optimal collective for large tensors. Given `N` ranks and message size `M`:

- Reduce-scatter phase: `N-1` steps. Each step sends `M/N` bytes. Total: `(N-1)·M/N` bytes per GPU.
- All-gather phase: `N-1` steps. Same volume. Another `(N-1)·M/N` bytes per GPU.
- Total per GPU: `2(N-1)·M/N` ≈ `2M` for large N.

The key property: data sent per GPU is independent of N. That is what makes ring scale to thousands of GPUs without saturating any single link.

```
Ring of 4 ranks, M=4 chunks. Step 1 of reduce-scatter:

R0 [a0 a1 a2 a3]      R0 sends a3 → R1
R1 [b0 b1 b2 b3]      R1 sends b0 → R2
R2 [c0 c1 c2 c3]      R2 sends c1 → R3
R3 [d0 d1 d2 d3]      R3 sends d2 → R0

After 3 reduce-scatter steps each rank owns one fully-summed chunk.
After 3 all-gather steps every rank has all 4 chunks summed.
```

## Tree all-reduce

Latency `O(log N)` instead of `O(N)`. Wins when message size is small enough that the per-step latency dominates the bandwidth cost. NCCL's heuristic auto-switches at a tunable threshold (`NCCL_ALGO_THRESHOLD`).

```
8-rank tree, reduction phase:

  R0 ── R1     R2 ── R3     R4 ── R5     R6 ── R7
   \_____/      \_____/      \_____/      \_____/
      \____________/             \____________/
              \_________________________/
                       root has full sum
```

Then a broadcast back down. Total time: `2·log(N)·α + 2·log(N)·M/β` where α is latency, β is bandwidth.

NCCL also has `Tree`, `Ring`, `CollnetChain`, `CollnetDirect`, and `NVLS` (NVLink SHARP) algorithms. Force selection with `NCCL_ALGO=Ring|Tree|...`.

## NCCL 2.27 — what's new in 2026

Source: [NCCL 2.27 release blog](https://developer.nvidia.com/blog/enabling-fast-inference-and-resilient-training-with-nccl-2-27/).

- **SHARP for both NVLink and InfiniBand**. Switch silicon does the reduction. SMs free; bandwidth amplified at scale. Toggle: `NCCL_NVLS_ENABLE=1`, `NCCL_SHARP_DISABLE=0`.
- **Communicator Shrink** (`ncclCommShrink`) — drop a rank from a live comm without tearing it down. Two modes:
  - `NCCL_SHRINK_DEFAULT` — planned reconfig. The to-be-dropped rank participates in the shrink call.
  - `NCCL_SHRINK_ABORT` — the dropped rank is dead/unresponsive. Survivors call shrink and continue.
  This is the foundation under elastic training and the cleanest path to recovery from a hung/dead GPU.
- **Symmetric memory + window-based primitives** — lower-overhead small-message paths.
- **PAT (Parallel Aggregated Trees)** — better all-gather/reduce-scatter at medium scales.
- **FlightRecorder dump on hang** — see the debugging section below.

## NCCL vs OpenMPI

OpenMPI is the HPC-world equivalent of NCCL. Same collective vocabulary (`MPI_Allreduce`, `MPI_Allgather`, `MPI_Alltoall`), much older heritage (1990s), CPU-first with GPU support added later via CUDA-aware MPI.

In 2026:
- NCCL dominates LLM training and inference on NVIDIA hardware. Every modern stack — PyTorch DDP/FSDP, DeepSpeed, Megatron-Core, vLLM TP, SGLang — uses NCCL.
- OpenMPI shows up in HPC-adjacent ML: molecular dynamics + ML, climate-model training, scientific computing pipelines, codebases predating NCCL maturity.
- Hybrid pattern is real: `mpirun -np 8 python train.py` for process spawning, then NCCL for GPU collectives. Common in labs that came from HPC.
- Same algorithms. If you read `MPI_Allreduce` in code, mentally substitute "NCCL would do the same on a GPU tensor."

## Topology detection

NCCL probes the system at `ncclCommInitRank` and prints (with `NCCL_DEBUG=INFO`):
- Which links connect which GPUs (NVLink, PCIe, IB)
- Number of NVLinks per pair
- Rail boundaries (which NIC owns which GPU subset)
- Chosen algorithm/protocol per message size

Read this once on a real multi-GPU box. You will learn more about your interconnect in five minutes than from any document.

```
NCCL INFO Channel 00 : 0[3000] -> 1[44000] via NVL/0
NCCL INFO Channel 01 : 0[3000] -> 1[44000] via NVL/1
NCCL INFO 2 NVLinks per pair, ring topology = 0 -> 1 -> 0
```

Two GPUs, both on NVLink, two channels. Compare to:

```
NCCL INFO Channel 00 : 0[3000] -> 1[44000] via SHM/direct/direct
NCCL INFO Channel 00 : 0[3000] -> 1[44000] via P2P/IPC
```

That is the "no NVLink between these GPUs" path. PCIe with peer-to-peer enabled. Order of magnitude slower.

For multi-node:

```
NCCL INFO Channel 00 : 4[3000] -> 0[3000] [send] via NET/IB/0/GDRDMA
```

`GDRDMA` is GPUDirect RDMA. If you see `via NET/Socket` instead, RDMA is not in play and you are sending GPU→host→TCP→host→GPU. Investigate.

## Common hangs and how to debug

NCCL hangs are the bane of distributed training. The collective requires all ranks to call it with matching shapes/dtypes/in-order. If one diverges, the others wait forever.

Patterns:

1. **Rank ordering mismatch.** Rank 0 calls `all_reduce(tensor_A)` while rank 1 calls `all_reduce(tensor_B)` of different shape. NCCL hangs. `NCCL_DEBUG=INFO` shows the mismatch.
2. **Stream desync.** The collective is queued on stream X, the dependent compute on stream Y, no event between them. Hangs invisibly — the kernel is queued but never runs.
3. **Skipped collective.** One rank early-returns from a training step (exception swallowed). The others wait at the next barrier forever. Always wrap the step in `try/finally` with a barrier on exit.
4. **Different number of microbatches across DP ranks.** Last batch is uneven. Use `set_epoch` + DistributedSampler with `drop_last=True` or pad.
5. **Network flap.** NCCL by default does not recover from transient IB failures. Use Communicator Shrink in error mode, or set `NCCL_IB_TIMEOUT` higher and retry at the framework level.
6. **CUDA OOM on one rank only.** That rank dies; others wait. Watchdog should kill the whole job.

### FlightRecorder

NCCL 2.27+ ships a flight recorder. On hang, dump per-rank state:

```bash
NCCL_DEBUG=INFO \
TORCH_NCCL_TRACE_BUFFER_SIZE=20000 \
TORCH_NCCL_DUMP_ON_TIMEOUT=1 \
TORCH_NCCL_DEBUG_INFO_TEMP_FILE=/tmp/nccl_trace \
torchrun --nproc_per_node=8 train.py
```

When a watchdog timeout hits, every rank dumps its in-flight collectives. Diff them. The rank that has fewer collectives queued is the deserter. The collective at the head of every other rank's queue is what they were waiting on.

PyTorch docs: [pytorch.org/docs/stable/distributed.html#flight-recorder](https://pytorch.org/docs/stable/distributed.html#flight-recorder).

## Useful environment variables

```
NCCL_DEBUG=INFO              # always-on for new clusters
NCCL_DEBUG_SUBSYS=ALL        # narrow with INIT,COLL,P2P,NET as needed
NCCL_ALGO=Ring               # force ring (default is auto)
NCCL_PROTO=Simple|LL|LL128   # message protocol; LL/LL128 are low-latency
NCCL_P2P_DISABLE=1           # disable GPU peer-to-peer (forces SHM/host staging)
NCCL_IB_DISABLE=1            # disable InfiniBand (forces TCP)
NCCL_CROSS_NIC=0|1|2         # rail-aware: 0 strict same-rail, 2 cross-rail allowed
NCCL_NVLS_ENABLE=1           # NVLink SHARP
NCCL_BUFFSIZE=8388608        # internal buffer; tune for small-message tail
NCCL_MIN_NCHANNELS=4         # parallel rings
NCCL_SOCKET_IFNAME=ib0       # bind to specific NIC
```

## Reference

- NCCL docs: [docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html)
- NCCL 2.27 blog: [developer.nvidia.com/blog/enabling-fast-inference-and-resilient-training-with-nccl-2-27](https://developer.nvidia.com/blog/enabling-fast-inference-and-resilient-training-with-nccl-2-27/)
- nccl-tests: [github.com/NVIDIA/nccl-tests](https://github.com/NVIDIA/nccl-tests)
- PyTorch FlightRecorder: [pytorch.org/tutorials/prototype/flight_recorder_tutorial.html](https://pytorch.org/tutorials/prototype/flight_recorder_tutorial.html)
- Bandwidth-optimal allreduce paper: [andrew.gibiansky.com/blog/machine-learning/baidu-allreduce/](https://andrew.gibiansky.com/blog/machine-learning/baidu-allreduce/)
