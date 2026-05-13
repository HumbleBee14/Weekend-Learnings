# Notes — 06

## Hardware, dtype, shape
GPU model, dtype, (M, N).

## Latency table
| variant | ms/iter |
|---|---|
| eager (uncompiled) | |
| eager (compiled) | |
| Pattern A (triton_op) | |
| Pattern B (custom_op) | |

## Inductor output evidence
From `TORCH_COMPILE_DEBUG=1` runs:

- Pattern A output_code.py — describe the kernel(s): is the residual add inside the rmsnorm kernel?
- Pattern B output_code.py — describe the kernel(s): how many separate kernels?

## What you'd ship in production
Which pattern, why, and any edge cases (training, mutation, etc.) that would change your answer.
