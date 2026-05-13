# If you don't have Hopper or Blackwell

> Read this in place of running [`02_warp_specialized_matmul.py`](02_warp_specialized_matmul.py) and [`03_warp_specialized_attention.py`](03_warp_specialized_attention.py) if your only GPU is T4 / A100 / RTX 4090. The scripts will still run and produce correct results on those chips — you just won't see the speedup. This document gives you what you would have seen.
>
> Numbers below are illustrative. They are consistent with H100 SOL data published in the [Tawa paper](https://arxiv.org/abs/2510.14719), the [PyTorch warp-spec design blog](https://pytorch.org/blog/warp-specialization-in-triton-design-and-roadmap/), and [Tri Dao's FA3 writeup](https://tridao.me/blog/2024/flash3/), but the exact numbers on your specific H100 would vary by ±10%. They are presented to teach the *shape* of the result, not to be quoted.

## What you'd see if you ran 01 vs 02 on an H100

`M = N = K = 4096`, fp16, H100 SXM5 (700 W, 989 TFLOPS fp16 peak):

| Kernel | Time (ms) | TFLOPS | % of cuBLAS | % of peak |
|---|---|---|---|---|
| `01_tma_matmul.py` (TMA, no warp spec) | 0.23 | ~600 | ~73% | ~61% |
| `02_warp_specialized_matmul.py` (warp spec, `num_consumer_groups=2`, `num_buffers_warp_spec=3`) | 0.18 | ~780 | ~95% | ~79% |
| `torch.matmul` (cuBLAS) | 0.17 | ~820 | 100% | ~83% |

The warp-spec version is ~28% faster than the TMA-only version and lands within 5% of cuBLAS. That gap — TMA-only at 73% of cuBLAS, warp-spec at 95% — is where the producer/consumer pipeline lives.

Now read the trace below to see *why*.

## An annotated proton trace

This is an illustrative snippet of what [`triton.proton`](https://triton-lang.org/main/profiling/proton.html) would show for the K-loop of `02_warp_specialized_matmul.py` on H100 at the winning config. The trace is collapsed to one K-loop iteration to make the pattern visible:

```
SM 0, iteration k:

  warp group 0 (PRODUCER, 4 warps, 128 lanes):
    t=0     : tma.load A[off_m, k]   -> smem.A[slot_k%3]    (~120ns issue)
    t=10    : tma.load B[k, off_n]   -> smem.B[slot_k%3]    (~120ns issue)
    t=30    : mbarrier.arrive smem.A[slot_k%3]
    t=30    : mbarrier.arrive smem.B[slot_k%3]
    t=40    : await mbarrier.consumed[slot_(k-3)%3]  (waits for ring slot to free)
    t=...   : begin iter k+1

  warp group 1 (CONSUMER 0, 4 warps, 128 lanes):
    t=0     : await mbarrier smem.A[slot_(k-1)%3]   (data filled by producer earlier)
    t=20    : await mbarrier smem.B[slot_(k-1)%3]
    t=40    : wgmma.fence
    t=50    : wgmma.async  acc += A[slot_(k-1)%3] @ B[slot_(k-1)%3]   (~600ns)
    t=650   : wgmma.commit_group
    t=660   : mbarrier.arrive consumed[slot_(k-1)%3]
    t=...   : begin iter k+1 (overlaps with consumer 1 doing wgmma)

  warp group 2 (CONSUMER 1, 4 warps, 128 lanes):
    t=300   : (ping-pong offset; running wgmma for iter k-2 while consumer 0
              runs wgmma for iter k-1)
    ...
```

Three things to read off the trace:

1. **The producer warps spend ~30ns issuing two TMA copies and then sit on `await`.** They are *not* the bottleneck. They are way ahead of the consumers. The pipeline depth of 3 (`num_buffers_warp_spec=3`) means by the time the consumer is on iteration `k`, the producer has already kicked off iterations `k+1` and `k+2`.
2. **The two consumer warp groups overlap.** Consumer 0 starts its `wgmma` at t=50. Consumer 1 starts a different iteration's `wgmma` at t=300. The tensor cores on this SM are never idle — between the two consumers, there's always a `wgmma` in flight. This is the FA3 ping-pong, in Triton.
3. **Every TMA load is decoupled from the warp that uses the data.** The producer issues the load and immediately waits on the "slot consumed" mbarrier — its job is just to keep the ring buffer full. The consumer waits on the "slot filled" mbarrier and consumes — its job is just to compute. Neither warp ever does both.

Compare against the trace you'd see from `01_tma_matmul.py` (same config, no warp spec):

```
SM 0, iteration k:

  warp group 0 (the only warp group, 4 warps):
    t=0     : tma.load A[off_m, k]   -> smem.A     (~120ns issue + wait)
    t=120   : tma.load B[k, off_n]   -> smem.B     (~120ns issue + wait)
    t=240   : wgmma.async  acc += A @ B            (~600ns)
    t=840   : ... begin iter k+1
```

One warp group is doing everything. The `wgmma` cannot start until both loads land. The next iteration's loads cannot start until this iteration's `wgmma` retires (in practice the compiler does pipeline this with `num_stages`, but only within the constraints of one warp group's instruction stream). Each iteration is ~840ns wall-clock; the warp-spec version is ~600ns because the load-issue is hidden behind the prior iteration's MMA. That ratio — 840 / 600 ≈ 1.4× — is roughly the speedup you'd measure.

## SM activity counters

Same shape, the proton metrics you'd care about:

| Metric | 01 (TMA only) | 02 (warp spec) | Why |
|---|---|---|---|
| `sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed` | ~62% | ~85% | Tensor cores busy more of the time |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | ~78% | ~92% | More warps doing useful work simultaneously |
| `dram__bytes_read.sum` | 4.0 GB/s/SM | 4.0 GB/s/SM | Same HBM traffic — fusion didn't change |
| `lts__t_sectors_aperture_device.sum / dram__bytes_read.sum` | 1.18 | 1.05 | Less L2 churn under warp spec (better prefetch) |
| `gpc__cycles_elapsed.avg` | ~840 | ~600 | Wall-clock per iteration |

The headline: **tensor-core active percentage rose from 62% to 85%, and that is where the speedup came from.** HBM traffic is identical — both kernels move the same bytes. We didn't fuse anything; we just stopped wasting tensor-core cycles waiting for the load-issuing warp to context-switch.

## What you'd see on Blackwell (B200)

Same shape, B200 SXM (1200W, ~2000 TFLOPS fp16 peak via tcgen05):

| Kernel | TFLOPS | % of peak | Speedup vs `01` |
|---|---|---|---|
| `01_tma_matmul.py` | ~1100 | ~55% | 1.0× |
| `02_warp_specialized_matmul.py` (`num_buffers_warp_spec=4`) | ~1650 | ~83% | 1.5× |
| cuBLAS | ~1750 | ~88% | 1.6× |

The warp-spec win is *larger* on Blackwell because the tensor cores are faster — the relative cost of a wasted tensor-core cycle is higher, so hiding the load-issue is worth more. `num_buffers_warp_spec` typically wants to go up by 1 (3 → 4 or 4 → 5) on Blackwell because TMEM gives you more SMEM to spend on the ring buffer and the deeper pipeline pays off.

The gap to cuBLAS (5%) is what FlashAttention-4 closes by going to CuTe-DSL: 2-SM cooperative MMA, TMEM-resident pipelined softmax, hand-laid-out FP4 epilogues. None of those are in Triton yet (May 2026). For 90% of kernels, the 5% is not worth ~70k lines of CuTe.

## What you'd see on T4 / A100 / RTX 4090 (the chips you might actually have)

`M = N = K = 4096`, fp16:

| GPU | `01` TFLOPS | `02` TFLOPS | Speedup | Why |
|---|---|---|---|---|
| T4 (Turing, fp16 peak ~65) | ~32 | ~32 | 1.0× | No TMA, no `wgmma`, `warp_specialize` no-ops |
| A100 (Ampere, fp16 peak ~312) | ~180 | ~185 | 1.03× | Has `cp.async`, no TMA, no `wgmma`; warp spec falls back |
| RTX 4090 (Ada, fp16 peak ~330) | ~210 | ~215 | 1.02× | Has TMA descriptor lowering to cp.async, no `wgmma`; warp spec marginal |

On all three: **the kernel runs correctly, the kernel runs fast (for the hardware), and `warp_specialize=True` does roughly nothing.** This is the right outcome for the wrong hardware. The structural lesson — write your kernel as a producer-friendly K-loop that the compiler can specialize — is still worth absorbing. The moment your code touches an H100 or MI300, the speedup is waiting.

## What this tells you about reading FA3, vLLM, SGLang

When you open vLLM's [`vllm/v1/attention/backends/triton_attn.py`](https://github.com/vllm-project/vllm/tree/main/vllm/v1/attention) or SGLang's [`fused_moe_triton`](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/layers/moe/fused_moe_triton), find the inner reduction loop and check three things:

1. Is `warp_specialize=True` set on the `tl.range`? (Should be, for the inner K-loop or KV-streaming loop.)
2. What `num_consumer_groups` does the autotune list show? (Often `[0, 2]` — FA3 ping-pong when 2.)
3. What `num_buffers_warp_spec` does it list? (Often `[2, 3, 4]` — the ring depth.)

If you see all three, you're looking at the producer/consumer pipeline from this sub-module, applied at scale. The 800 lines around it are paged-KV indexing, varlen masking, and the autotune key — wrapping the core loop you wrote in 02 and 03.

That is the bar this sub-module is training you toward.
