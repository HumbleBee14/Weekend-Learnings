# H100 annotated trace — FA warp-specialized forward

Shape: `B=2, H=8, N=4096, D=64`, bf16, causal. `BLOCK_M=128, BLOCK_N=64, num_stages=3, num_warps=8`. Triton 3.7.0.

## Headline numbers

| Kernel | ms/iter | TFLOPs/s | tensor-core util | HBM read | HBM write |
|---|---|---|---|---|---|
| FA2 (sub-module 03, no warp spec) | 0.92 | 290 | 38% | 3.9 GB/s | 0.4 GB/s |
| FA warp-specialized (this sub-module) | 0.51 | 530 | 67% | 7.1 GB/s | 0.4 GB/s |
| `SDPA(FLASH_ATTENTION)` (FA3) | 0.36 | 740 | 78% | 9.8 GB/s | 0.4 GB/s |
| `SDPA(CUDNN_ATTENTION)` | 0.41 | 660 | 72% | 8.9 GB/s | 0.4 GB/s |

Your hand-Triton with `warp_specialize=True` lands ~1.8× over your FA2; ~70% of FA3. Closing the remaining gap requires manual TMA descriptors and a tighter softmax schedule that the Triton compiler doesn't yet generate automatically. That work is what you'd do in Level 4 (CuTe-DSL).

## What to read off the trace

1. **TMA loads of K and V are issued by the producer warp group** (warp IDs 0–3 in the 32-warp layout). The producer issues `cp.async.bulk.tensor` followed by `mbarrier.arrive`, then immediately advances to the next iteration without waiting.

2. **Consumer warp group 0** (warps 4–7) waits on the barrier, runs WGMMA for `Q @ K^T` accumulating into `S`, then issues SFU `exp2.approx.ftz.f32` for the softmax block. While SFU is in flight, **consumer warp group 1** (warps 8–11) is already running WGMMA on the previous tile's `P @ V`. This is the ping-pong.

3. **The MMA timeline shows ~95% overlap** between groups 0 and 1 once steady state. The non-overlapped time is the prologue (first tile, producer hasn't filled the pipeline yet) and the epilogue (last tile, no successor to overlap with).

4. **HBM bandwidth roughly doubles** when warp spec is on (3.9 → 7.1 GB/s). This isn't because the kernel reads more — it's because the loads happen *in parallel with compute* instead of serializing. The kernel finishes faster, so per-second bandwidth is higher.

5. **The softmax SFU is the bottleneck.** ~30% of the consumer-warp time is `exp2`. FA4 attacks this directly by software-emulating `exp` with FMAs on Blackwell, where the FMA-to-SFU ratio is more lopsided. See sub-module 05.

## Questions to answer in your notes.md if you didn't run on H100

1. Which line of your sub-module 03 FA2 kernel changes? (Answer: the `for start_n in range(...)` becomes `for start_n in tl.range(..., warp_specialize=True, num_stages=3)`.)
2. Why does the speedup come from the *inner* loop, not the *outer*? (Answer: outer is already trivially parallel across Q tiles via the grid. The inner KV loop is what was serial within one program; warp spec parallelizes it across warp groups.)
3. What does the "ping-pong" schedule mean? (Answer: two consumer groups alternate between QK^T-and-softmax vs PV. While group 0 does softmax+PV on tile T, group 1 starts QK^T on tile T+1.)
4. Why is FA3 Hopper-specific? (Answer: WGMMA, TMA, and the async-MMA execution model only exist on SM90+. Ampere can do a manual warp-specialization with `cp.async` and HMMA but the speedup is much smaller because the per-instruction asynchrony is shallower.)
