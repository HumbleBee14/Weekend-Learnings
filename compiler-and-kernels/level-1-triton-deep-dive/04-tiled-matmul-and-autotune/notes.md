# notes.md — tiled matmul and autotune

## Setup

GPU: 
Triton version: 

## Step 01 — basic tiled

TFLOPS achieved on 4096^3 fp16: 
% of torch.matmul (cuBLAS): 

Why this isn't great:


## Step 02 — TMA descriptors

TFLOPS achieved: 
% of cuBLAS: 

Did you see a meaningful jump vs step 01? (Yes on Hopper+; ~no on pre-Hopper.)
What does this tell you about your hardware?


## Step 03 — autotuned with pruning

Best config the autotuner picked:
  BLOCK_M = 
  BLOCK_N = 
  BLOCK_K = 
  num_warps = 
  num_stages = 

TFLOPS achieved: 
% of cuBLAS: 

### Three sentences on why this config won

(Walk through what each chosen value means on your hardware. SRAM footprint, register pressure, pipelining depth, tensor-core fragment alignment.)

1. 

2. 

3. 

## compare_with_inductor.py — table

```

```

Where did you beat Inductor?

Where did Inductor beat you?

Where did you both lose to cuBLAS?

## After reading the Inductor-emitted Triton

`TORCH_COMPILE_DEBUG=1 python compare_with_inductor.py` will dump Inductor's generated kernels to `torch_compile_debug/`. Find the matmul kernel for the 4096^3 case. Read it. What's similar to yours?

What's different?

What does Inductor do that you didn't?

## Generalizable take-away

In a sentence: what's the right way to autotune matmul-shaped kernels?
