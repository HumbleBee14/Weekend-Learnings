# Notes — 04

## Where the dump lived
[path to output_code.py]

## Kernels Inductor emitted
For each kernel: name, what op(s) it covers, approx line count.

## Fusion decisions
- Fused: [...]
- Not fused: [...]
- Reasons (your guess + verification by re-reading the kernel body):

## Matmul backend chosen
- Backend: [aten / triton template / cutlass]
- Tile sizes if visible:
- Would you have picked this? Why or why not?

## Buffer allocations (`empty_strided_cuda` calls in call(args))
Count them. If more than you expected, where are the extras coming from?

## Compared to Level 1 hand-Triton RMSNorm
- What Inductor did the same:
- What Inductor did differently:
- Bandwidth utilization gap (if you measured):

## What surprised me
