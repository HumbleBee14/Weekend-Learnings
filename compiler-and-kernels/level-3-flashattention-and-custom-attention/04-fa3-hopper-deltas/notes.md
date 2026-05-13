# notes — fa3 deltas

## Connection back to Level 1

The warp-specialized GEMM you wrote in Level 1 sub-module 05 used the same producer/consumer pattern. The difference for attention:

- The "compute" half has *two* matmuls (QK^T and PV) plus the softmax, instead of one matmul.
- Two consumer warp groups alternating (ping-pong) means while group 0 is in the softmax SFU, group 1 is in PV WGMMA — total tensor-core occupancy goes up.

If you can recognize the warp_specialize=True knob in your GEMM and now in your attention kernel, you can recognize it in vLLM's Triton paged attention, in FlashInfer's prefill kernel, and in the Triton in-tree tutorial. Same pattern, applied many places.

## What the FA3 paper covers that we skipped

- Sections on FP8 incoherent processing — read for context, but you don't need to implement Hadamard-randomized FP8 to understand the speedup.
- Backward pass — covered briefly in sub-module 08.
- Variable-length forward — FlashInfer's territory; sub-module 07.

## What FA3 did *not* solve

- Softmax SFU bottleneck. FA3's SFU exp is still ~30% of consumer-warp time on H100. FA4 attacks this with software FMA-emulated exp on Blackwell (sub-module 05).
- Conditional rescale. FA3 rescales O on every tile boundary even when the max didn't change. FA4 skips it (sub-module 05).
- Single-CTA only. FA3 doesn't use Blackwell's 2-CTA MMA. FA4 does.
