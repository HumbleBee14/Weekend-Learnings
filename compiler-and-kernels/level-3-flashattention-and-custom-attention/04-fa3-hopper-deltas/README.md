# 04 — FA3 deltas: warp specialization, WGMMA, FP8

> Prereq: sub-module 03 (you have a working Triton FA2 forward). Hardware: H100 ideal. If you don't have H100, read the included annotated trace and proceed — you will not be blocked.

FA3 ([Shah, Bikshandi, Zhang, Thakkar, Ramani, Dao — arXiv 2407.08608, July 2024](https://arxiv.org/abs/2407.08608)) is FA2 with the Hopper-shaped pipelining bolted on. The algorithm is the same; the scheduling is different. Three deltas, in increasing order of how much they matter:

## Delta 1 — Producer/consumer warp specialization

FA2 used one warp group for everything: it issued loads, waited for them, did the GEMM, did the softmax, did the second GEMM. While the SFU computed exponentials, the tensor cores sat idle. While the TMA fetched the next K/V tile, the warp blocked.

FA3 splits the work:

- **Producer warps**: issue TMA loads of K and V into shared memory, signal a barrier when each tile lands.
- **Consumer warps** (two groups, "ping-pong"): one does QK^T into the score buffer, the other does the softmax and PV. They alternate per tile so the tensor cores stay busy.

This is *exactly* the pattern you built in Level 1 sub-module 05 for warp-specialized GEMM. The producer/consumer split. The TMA barrier. The two consumer groups alternating. Attention has one extra wrinkle — the softmax SFU work — but it slots into the same scheduling model.

What the timeline looks like for one Q tile, inner loop over K/V tiles n=0..3:

```
   time ──►
            │ tile 0       │ tile 1       │ tile 2       │ tile 3       │
   ─────────┼──────────────┼──────────────┼──────────────┼──────────────┤
   Producer │ TMA K0,V0    │ TMA K1,V1    │ TMA K2,V2    │ TMA K3,V3    │   (async copies)
   warps    │       ┊──────┊──────┊──────┊──────┊──────┊──────┊─────►   │   (run ahead by ~num_stages)
   ─────────┼──────────────┼──────────────┼──────────────┼──────────────┤
   Consumer │ ░░░░ wait    │ QKᵀ tile1    │ softmax+PV 2 │ QKᵀ tile3    │   ping (tensor cores: MMA)
   group A  │              │              │              │              │
   ─────────┼──────────────┼──────────────┼──────────────┼──────────────┤
   Consumer │ ░░░░ wait    │ softmax+PV 1 │ QKᵀ tile2    │ softmax+PV 3 │   pong (SFU + MMA)
   group B  │              │              │              │              │
   ─────────┴──────────────┴──────────────┴──────────────┴──────────────┘
              ↑ priming     ↑ steady state: tensor cores & SFU both busy every cycle
```

A and B alternate roles each tile so the tensor-core path is never idle waiting for softmax, and softmax is never idle waiting for MMA. Producer runs `num_stages` ahead so by the time a consumer needs tile n+1, the bytes have already landed. The FA2 kernel had A, B, and the producer all collapsed into one warp group — when one part worked the other two stalled. This picture is the entire "1.5–2× over FA2" story.

In Triton, you flip this on with one knob:

```python
for start_n in tl.range(0, N, BLOCK_N, warp_specialize=True, num_stages=3):
    ...
```

The compiler partitions the loop body into producer (loads) and consumer (compute) and schedules them across warp groups. The Triton implementation of this is described in the [Tawa paper (arXiv 2510.14719)](https://arxiv.org/abs/2510.14719); the upstreaming is [PR #6288](https://github.com/triton-lang/triton/pull/6288) and follow-ups.

## Delta 2 — WGMMA instead of HMMA

Ampere (A100) used `mma.sync` issued per warp (32 threads), max throughput ~312 TFLOPs/s FP16. Hopper introduced **WGMMA** (warp-group matrix multiply accumulate, `wgmma.mma_async`) issued per warp group (128 threads), much higher per-SM throughput, and *asynchronous* — the issuing warp keeps running while the tensor core does its thing.

In Triton you don't write WGMMA directly. `tl.dot` lowers to WGMMA on SM90, to HMMA on SM80, to tcgen05 MMA on SM100. The compiler picks. Your job is to keep the tile sizes friendly to the WGMMA instruction shapes (`64×N×16` for FP16, multiples thereof). Autotune handles it.

## Delta 3 — FP8 with incoherent processing

The naive way to do FP8 attention is per-tensor quantization: `Q, K, V` each get one scale. Outliers in any single head wreck the scale and you lose accuracy. FA3 does **block quantization** plus **incoherent processing**:

- *Block quantization*: scale per `(BLOCK_M, BLOCK_N)` tile, not per tensor.
- *Incoherent processing*: apply a random Hadamard transform to Q and K before quantizing, multiply by the same transform's transpose on the other side; the math is identity but the outliers get spread across dimensions instead of concentrating in a few channels.

FA3 reports RMSE 2.6× better than naive per-tensor FP8 attention. With these tricks, FP8 attention on H100 reaches ~1.2 PFLOPs/s — roughly 1.6× the FP16 number.

## Performance baselines

Tri Dao's numbers on H100:
- FA2 FP16: ~350 TFLOPs/s.
- FA3 FP16: ~740 TFLOPs/s (75% of peak).
- FA3 FP8: ~1.2 PFLOPs/s.

The FP16 1.5–2× over FA2 is mostly the warp specialization (Delta 1). WGMMA (Delta 2) is implicit in both — it's the only way to hit Hopper peak. FP8 (Delta 3) is the third multiplier on top.

## What you build

1. `fa3_warp_specialized_triton.py` — take your sub-module 03 kernel, change one knob (`warp_specialize=True` on the KV loop), add `num_consumer_groups=2`, and benchmark. On H100 expect 1.3–1.7× over the FA2 version.
2. `bench_fa3.py` — head-to-head: your FA2 Triton, your FA3 Triton, `F.scaled_dot_product_attention(FLASH_ATTENTION)` (FA3 on H100), and `F.scaled_dot_product_attention(CUDNN_ATTENTION)` for a cuDNN reference.
3. `trace_h100.md` — if you have H100, paste your `triton.proton` trace summary here. If you don't, read the included reference trace (we ship one from a known-good run) and answer the questions in the file.

## The trace, annotated

The shipped trace `trace_h100.md` is from a tuned H100 run on `(B=2, H=8, N=4096, D=64, bf16, causal=True)`. Key things to notice:

- **Producer warps**: spend ~85% of their time *between* TMA issue and the next barrier wait. They are not stalled; they are correctly running ahead.
- **Consumer warps**: 70%+ tensor-core utilization. The remaining time is softmax (SFU + FMA).
- **Async overlap**: the timeline shows MMA for tile T executing while TMA for tile T+1 is still in flight. This is what "the kernel does both at the same time" looks like in a profiler.
- **Comparison to non-warp-specialized**: the same kernel without `warp_specialize=True` has a sawtooth tensor-core utilization pattern (compute, idle, compute, idle); FA3 fills in the troughs.

## If you don't have H100

Don't pay to rent one for this sub-module. Read the [FA3 blog post](https://tridao.me/blog/2024/flash3/) and the [PyTorch FA3 announcement](https://pytorch.org/blog/flashattention-3/), study the shipped `trace_h100.md`, and write `notes.md` answering:

- Which line of your FA2 Triton kernel changes to enable warp specialization?
- Why does the speedup come from the inner loop, not the outer?
- What does the "ping-pong" schedule mean for two consumer groups?
- Why is FA3 specific to Hopper (would the same trick work on Ampere)?

Then move on. Sub-module 06 (FlexAttention) doesn't need the warp-specialized kernel — when you set `kernel_options={"BACKEND": "FLASH"}` on Blackwell it uses FA4; on Hopper PyTorch SDPA dispatches to FA3 for you; on Ampere it falls back to FA2. The warp-spec work is for *understanding*, not for the rest of the level.

## Definition of done

- [ ] Either: a benchmark showing your warp-specialized kernel is 1.3×+ faster than your sub-module 03 FA2 on H100.
- [ ] Or: a `notes.md` answering the four questions above, having read the trace and the FA3 paper.
- [ ] You can connect Level 1's warp-specialization pattern to attention specifically.

## References

- [FA3 paper — arXiv 2407.08608](https://arxiv.org/abs/2407.08608) Sections 3–4.
- [Tri Dao — FA3 blog post (Jul 2024)](https://tridao.me/blog/2024/flash3/).
- [PyTorch — FA3 announcement](https://pytorch.org/blog/flashattention-3/).
- [Colfax Research — FA3 deep dive](https://research.colfax-intl.com/flashattention-3-fast-and-accurate-attention-with-asynchrony-and-low-precision/).
- [Tawa paper — arXiv 2510.14719](https://arxiv.org/abs/2510.14719) for the Triton-side description.
- [PyTorch — Warp Specialization in Triton: Design and Roadmap (Jan 2026)](https://pytorch.org/blog/warp-specialization-in-triton-design-and-roadmap/).
