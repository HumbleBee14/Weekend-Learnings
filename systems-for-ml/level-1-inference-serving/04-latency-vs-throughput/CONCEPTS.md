# 04 — Latency vs Throughput

## The single most important graph in serving

X-axis: batch size (or load).
Y-axis-left: throughput (tokens/sec).
Y-axis-right: p99 latency (ms).

You always get two curves: one rising, one rising-then-exploding. Every serving decision — batch size, autoscaling threshold, engine choice, hardware sizing — is a point on this curve.

If you can't read this graph for your own system, you can't make these decisions for someone else's.

## Why throughput rises with batch size

GPU forward pass cost is dominated by reading model weights from HBM. That cost is paid *once per batch*, not per sequence. Adding sequences to the batch is nearly free until you run out of memory or compute.

So throughput rises sharply, then flattens when one of these happens:
- KV cache fills up (memory-bound) — usually the first wall on consumer GPUs
- Tensor cores saturate (compute-bound) — happens at very high batch sizes on big GPUs
- Padding waste dominates — batch is 32 wide but only 4 sequences are doing useful work

## Why latency rises with batch size

Two effects layered on top of each other:

1. **Bigger batches take longer to compute.** Batch=8 takes ~1.5× the time of batch=1, not 8× — but it's not free.
2. **Queue waiting time goes up.** As you push more concurrent users at the server, requests pile up in the queue waiting for a batch to form. p99 = (queue wait) + (batch run time).

The second effect is what makes p99 *explode* past the knee. Once requests arrive faster than batches can finish, the queue grows unboundedly and tail latency goes through the roof.

## The knee

The "knee" is the inflection point: throughput gains have flattened but latency hasn't blown up yet. That's the operating point you usually want.

For a 0.5B model on CPU with 50-token outputs the knee is often around batch=4–8.
For a 7B on A100 with longer outputs it might be 16–32.
For 70B on H100 it's typically 32–64.

The exact location depends on:
- Model size (bigger model = more memory pressure = lower knee)
- Sequence length (longer = more KV cache = lower knee)
- Hardware (more memory = higher knee)
- Quantization (FP8 saves memory = higher knee)

## SLO-driven sizing

In production you don't pick "the knee." You pick the largest batch size that still meets your latency SLO. Example:

> "p99 must be ≤ 2000ms for any decode of ≤ 100 tokens at concurrency 50."

Work backwards:
1. Sweep batch sizes against the load you expect.
2. Find the largest batch that keeps p99 ≤ 2000ms.
3. That's your operating batch size.
4. If no batch size satisfies the SLO, you need more replicas (or a bigger machine).

This is exactly what KEDA autoscaling does at scale (Level 7).

## Little's Law preview

If you know any two of these you can compute the third:

```
L = λ × W
```

- L: average number of requests in the system
- λ: arrival rate (requests/second)
- W: average time in the system (sec)

Useful for: predicting `L` from observed `λ` and `W`, or sanity-checking your instrumentation. Comes back hard in Level 7.

## What you'll produce

The throughput-vs-p99 curve from the topic-03 measurements, plotted properly with annotations:
- Mark the knee
- Mark the SLO line if you have one
- Note the regime (memory-bound? compute-bound? queue-bound?)

This graph is **G1 of Project 1** and the first piece of `reports/week1.md`.
