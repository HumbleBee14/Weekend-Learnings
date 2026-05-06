# 02 — First CUDA Kernels

## Files

- `CONCEPTS.md` — kernel structure, coalescing, vectorized loads, reductions, online softmax
- `vector_add.cu` — the hello world. Standalone; compile with nvcc.
- `relu_vec4.cu` — plain vs vec4 ReLU side by side. Shows what vectorized loads buy you.
- `softmax.cu` — numerically stable softmax with shared-memory + warp-shuffle reduction.
- `run_in_pytorch.py` — the modern path: compile CUDA inline from Python via `torch.utils.cpp_extension.load_inline`. No nvcc command, no Makefile.

## Quickstart — option A: standalone CUDA

```bash
# A100 = sm_80, T4 = sm_75, H100 = sm_90, B200 = sm_100
nvcc -O3 -arch=sm_80 vector_add.cu -o vector_add && ./vector_add
nvcc -O3 -arch=sm_80 relu_vec4.cu  -o relu_vec4  && ./relu_vec4
nvcc -O3 -arch=sm_80 softmax.cu    -o softmax    && ./softmax

# Catch bugs (out-of-bounds, races, init issues)
compute-sanitizer ./softmax
compute-sanitizer --tool racecheck ./softmax
```

## Quickstart — option B: PyTorch inline (recommended for learning)

```bash
pip install torch
python run_in_pytorch.py
```

First run takes 30–60s to compile the extension, then it's instant.

## Quickstart — option C: Colab

```python
# In a cell:
%%writefile vector_add.cu
# (paste vector_add.cu content)

# Next cell:
!nvcc -O3 -arch=sm_75 vector_add.cu -o vector_add && ./vector_add
```

(T4 on free Colab is sm_75.)

## What you should see

`vector_add`: a couple hundred GB/s on T4, ~1 TB/s on A100, ~3 TB/s on H100. The kernel is bandwidth-bound — there's no compute to speak of.

`relu_vec4`: vec4 should be 1.5–2× faster than plain. The bandwidth headroom is the same (you're still moving the same bytes), but vec4 generates fewer instructions and merges loads better.

`softmax`: bandwidth in the 200–500 GB/s range. Lower than vector_add because softmax does multiple passes over the data (max, sum, normalize). FlashAttention's whole job is to fuse those passes.

## Try

- **Drop -O3 and re-time.** Watch performance crater. The optimizer matters a lot for CUDA.
- **Run with `compute-sanitizer`.** All three should pass. Now intentionally remove the bounds check (`if (i < n)`) and run again — `memcheck` will catch the out-of-bounds.
- **Profile with Nsight Compute** (Level 3 will cover this in depth): `ncu --set full ./softmax`. Look at "Memory Throughput" and "Compute (SM) Throughput."
- **Modify softmax to skip the `__syncthreads()` after `smem[warp_id] = val`.** Run with `compute-sanitizer --tool racecheck`. It'll catch the race.

## What's still missing

These three kernels are simple. They don't use:
- Tensor cores (no matmul yet — Topic 3)
- Async memory copies (`cuda::memcpy_async`, TMA on Hopper+)
- Multiple CUDA streams
- Persistent kernels

Topic 3 (matmul) introduces tiling and tensor cores. Topic 4 (Triton) shows how a Python DSL handles all of this for you.
