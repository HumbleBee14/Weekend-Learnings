# 15 — Distributed Mac Inference

## Why this exists

A single 64GB Mac fits a 70B at 4-bit comfortably and a 105B MoE if it fits in active-param bandwidth. It does **not** fit Llama-3.1-405B, DeepSeek-V3 (671B), or full-precision frontier checkpoints. The 2026 answer is to wire a few Macs together over Thunderbolt 5 and shard the model across them.

This is not "training across Macs." It is **inference**, pipeline-parallel, where each Mac holds a contiguous range of transformer layers and forwards activations to the next.

## The hardware story

| Link | Bandwidth (full duplex) | Round-trip | Practical for |
|------|------------------------|------------|---------------|
| Thunderbolt 4 | 40 Gb/s | ~50–100 us | Up to ~70B sharded |
| Thunderbolt 5 | 80 Gb/s (boost 120) | ~30–60 us | Up to 405B / 671B |
| 10 GbE | 10 Gb/s | ~150 us+ | Toy demos |
| Wi-Fi 7 | 5–10 Gb/s real | high jitter | Not for serving |

Thunderbolt 5 in a ring topology (Mac A -> Mac B -> Mac C -> Mac A) is the 2026 default for `mlx.distributed` and `exo`. Apple shipped TB5 on M4 Pro/Max+ and across all M5s.

```
        Mac 0                Mac 1                Mac 2
     +---------+           +---------+          +---------+
     | layers  |  TB5 80G  | layers  |  TB5 80G | layers  |
     |  0-31   | --------> | 32-63   | -------> | 64-95   |
     +---------+           +---------+          +---------+
          ^                                          |
          +---------- TB5 80G (ring close) ----------+
     activations forward; KV cache stays local to its layer's Mac
```

## The two stacks worth knowing

### `mlx.distributed`

[MLX `mx.distributed`](https://ml-explore.github.io/mlx/build/html/usage/distributed.html) ships a ring all-reduce over MPI or its own native backend. As of MLX 0.18+ (2026), tensor parallelism and pipeline parallelism for `mlx-lm` ship behind a single launcher:

```bash
mlx.launch --hostfile hosts.txt -n 3 \
  python -m mlx_lm.generate \
  --model mlx-community/Meta-Llama-3.1-405B-Instruct-4bit \
  --prompt "explain raft" --max-tokens 200
```

`hosts.txt` lists each Mac's IP (over the Thunderbolt-bridged subnet — set up "Internet Sharing" or a manual static-IP TB bridge). MLX shards weights at load time; each rank only allocates its slice.

Reference: [MLX distributed docs](https://ml-explore.github.io/mlx/build/html/usage/distributed.html), [Apple research note on M5 distributed inference](https://machinelearning.apple.com/research/exploring-llms-mlx-m5).

### `exo`

[exo-explore/exo](https://github.com/exo-explore/exo) is the more general project: pipeline-parallel inference across heterogeneous devices (Macs, Linux boxes, even iPhones), discovering peers automatically and routing activations through the optimal partition. Backends include MLX (preferred on Apple Silicon), tinygrad, llama.cpp.

Run on each device:

```bash
exo
```

The discovery protocol forms a mesh, partitions a model based on each device's free memory, and exposes an OpenAI-compatible endpoint on `:52415`. Pull a 405B model and it shards itself.

exo's design is closer to "Kubernetes for local inference" than to a tightly-coupled MPI job. Latency is higher than `mlx.distributed` because partitioning is dynamic, but it tolerates heterogeneous hardware.

### llama.cpp RPC backend

Less popular but still relevant: `llama.cpp` has an RPC backend (`-DLLAMA_RPC=ON`) that ships layers across machines. It works on any platform but is bandwidth-naive and slower than the MLX path on Apple Silicon. Useful when you have a mixed-OS pile.

## Pipeline parallelism — what's actually happening per token

```
prompt -> Mac 0 (embed + layers 0..31) -> activations [b, s, h] over TB5 ->
          Mac 1 (layers 32..63)         -> activations [b, s, h] over TB5 ->
          Mac 2 (layers 64..95 + lm_head) -> token logits -> sample on Mac 2
                                              (token id is broadcast back to Mac 0
                                               for the next decode step's input embed)
```

Per **prompt-processing** step you pay one TB5 hop per shard boundary, with the activation tensor `[batch, seq_len, hidden]`. For Llama-405B at fp16: hidden=16384, seq_len=512, batch=1 → 16 MB/hop. At TB5 80 Gb/s = 10 GB/s, that's ~1.6 ms/hop, easily hidden by compute.

Per **decode** step the activation is `[batch, 1, hidden]` ≈ 32 KB/hop — pure latency, ~50 us/hop. Over 3 Macs that's 150 us added to per-token latency. Acceptable. You will measure 35–55 tok/s decode on a well-tuned 3-Mac shard of 405B-4bit.

## KV cache placement

KV cache is **local to each Mac's layers**. Mac 0 stores K/V for its 32 layers; Mac 1 for its 32; etc. This is critical: the KV cache is the bandwidth bottleneck of decode, and you do **not** want to ship it across the wire. Long-context inference benefits proportionally from sharding because each Mac's KV is 1/N the size.

## Topology trade-offs

```
Star (one head + N workers)        Ring (N peers, layers 0..L-1 split)
                                   
       Mac 0 (head)                Mac 0 -> Mac 1 -> Mac 2
      / | | \                                            \
   M1  M2 M3 M4                              <----------+
                                   
   simple, head bottlenecks         even bandwidth, hardest to debug
```

`exo` prefers ring; `mlx.distributed` supports both but ring is typical for ≥3 nodes.

## When this is worth it

- You have **two or three** of the same Mac class (mixing M2 Pro with M4 Max means the M2 Pro is the bottleneck for the whole pipeline).
- You actually need a model that doesn't fit on the biggest single Mac you have. For 70B and below, a single 64GB+ Mac is faster (no inter-node hops).
- You can dedicate the Macs (no other display load on the GPU during inference).
- Thunderbolt 5 cables and ports — TB4 will work for 70B, struggles past 200B.

When it's **not** worth it: most of the time. A single M5 Max 128GB runs 70B at >40 tok/s. The interesting case is the 200B+ frontier.

## Failure modes

1. **Slowest Mac dominates.** Pipeline stalls at the slowest stage. Match hardware.
2. **TB chain length.** A daisy-chained TB5 cable across 3 Macs means hops compound. Use the host's two TB5 controllers to make the ring symmetric.
3. **Thermal throttling.** Sustained load on multiple Macs in a stack with no airflow throttles all of them. Space them out.
4. **Power.** Three M-series boxes pulling 60W each is fine on residential power, but a USB-C charger sharing the TB5 cable is asking for issues. Use dedicated power.
5. **Token sampling location.** If you sample on the head node, Mac N has to ship logits back. Sample on the last stage and broadcast the chosen id. `mlx.distributed` does this correctly; check exo's logs.

## Scope for the week

If you have one Mac, this topic is a read. Read the MLX distributed docs, skim `exo`'s README, write the bandwidth math for the model you'd shard. If you have two Macs, run `exo` across them and serve a 70B 4-bit. If you have three M-series and TB5, try Llama-405B-4bit through `mlx.launch`.

## References

- MLX distributed docs: https://ml-explore.github.io/mlx/build/html/usage/distributed.html
- mlx-examples distributed launchers: https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm
- Apple — exploring LLMs with MLX on M5: https://machinelearning.apple.com/research/exploring-llms-mlx-m5
- exo-explore/exo: https://github.com/exo-explore/exo
- Thunderbolt 5 spec (Intel): https://www.intel.com/content/www/us/en/architecture-and-technology/thunderbolt/thunderbolt-5.html
- llama.cpp RPC backend: https://github.com/ggml-org/llama.cpp/tree/master/examples/rpc
