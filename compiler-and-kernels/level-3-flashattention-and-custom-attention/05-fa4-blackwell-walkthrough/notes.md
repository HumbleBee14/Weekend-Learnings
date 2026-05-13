# notes — fa4 walkthrough

## The five changes, for a teammate who finished sub-module 04

### 1. Five warp specializations

(your one-paragraph explanation here)

### 2. Software-emulated exponentials

(your one-paragraph explanation here)

### 3. Conditional softmax rescaling

(your one-paragraph explanation here)

### 4. 2-CTA cooperative MMA

(your one-paragraph explanation here)

### 5. CuTe-DSL, not C++

(your one-paragraph explanation here)

## Why CuTe-DSL not Triton

Triton's warp specialization is a two-way (producer/consumer) partitioning. FA4 needs 5-way custom partitioning with hand-tuned barriers. Triton's compiler doesn't expose that level of control. CuTe-DSL does, while still being Python.

## Why Blackwell-only

- 2-CTA MMA: only exists on SM100+.
- Asymmetric scaling motivation: FMA-to-SFU ratio doubled relative to Hopper, so software-emulated exp pays off more.
- tcgen05 MMA family: SM100 instruction set.

On Hopper, FA3 already balances the SFU/MMA budget well; the FA4 tricks would help less and the rewrite cost is not worth it. This is the lesson: kernels are written for specific hardware, and a different bottleneck shape requires a different kernel.

## What I want for Level 4

CuTe-DSL fluency so I could read FA4's source line by line and write my own GEMM kernel in the same dialect. That's exactly the level-4 deliverable.
