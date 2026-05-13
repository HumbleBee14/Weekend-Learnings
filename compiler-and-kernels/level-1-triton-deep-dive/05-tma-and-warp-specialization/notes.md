# notes.md — your observations

Fill this in as you run the scripts. The point isn't bookkeeping; the point is to make yourself articulate what changed and why. The act of writing it down catches a lot of "I think I get it but I actually don't" before it metastasizes.

## Hardware

- GPU model:
- Compute capability:
- HBM peak bandwidth (TB/s):
- fp16 tensor-core peak (TFLOPS):
- Triton version (`python -c "import triton; print(triton.__version__)"`):

## 01 — TMA matmul, no warp spec

Shape: M=N=K=4096, fp16

- Triton TFLOPS:
- cuBLAS TFLOPS:
- % of cuBLAS:
- Winning autotune config (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages):

## 02 — warp-specialized matmul

- TFLOPS, warp_specialize=False:
- TFLOPS, warp_specialize=True:
- Speedup:
- Winning config (note especially `num_consumer_groups` and `num_buffers_warp_spec`):
- Did the winner have `num_consumer_groups=2`? (Ping-pong won?):

## 03 — warp-specialized attention

Shape: B=2, H=16, N=4096, D=64

- TFLOPS, warp_specialize=False:
- TFLOPS, warp_specialize=True:
- TFLOPS, `torch.nn.functional.scaled_dot_product_attention`:
- Your kernel as % of SDPA:

## In your own words

Three sentences. Don't peek at CONCEPTS.md.

1. What is a "producer warp" doing on every iteration of the K-loop?

2. What is a "consumer warp" doing on every iteration of the K-loop?

3. Why does `num_consumer_groups=2` (the FA3 ping-pong) help when one consumer group is already doing nothing-but-compute?

If you can't answer any of these without re-reading, you haven't earned the speedup. Go back and read [`CONCEPTS.md`](CONCEPTS.md) section "The async pipeline mental model" until you can.

## Surprises

Things that didn't match expectation:

-
-

## What you'd change next

If you had another two hours on this kernel, where would the time go?

-
