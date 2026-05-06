# 02 — First CUDA Kernels

## Goal

Write three CUDA C++ kernels and run them. Each teaches one thing:

1. **Vector add** — launch config, thread indexing, bounds check
2. **ReLU + vectorized loads** — why `float4` matters, achieved bandwidth
3. **Softmax** — shared memory reduction, warp shuffles, online numerical stability

By the end you can write a basic kernel from memory and reason about its memory access pattern.

## The CUDA program structure

A CUDA program has three parts:

```
┌──────────────────────────────────────────────┐
│ HOST CODE (regular C++)                      │
│   - allocate host memory (malloc / new)      │
│   - allocate device memory (cudaMalloc)      │
│   - copy host → device (cudaMemcpy)          │
│   - launch kernel: kernel<<<grid, block>>>() │
│   - copy device → host (cudaMemcpy)          │
│   - free everything                          │
└──────────────────────────────────────────────┘
                  ↓ launches ↓
┌──────────────────────────────────────────────┐
│ DEVICE CODE (kernel: __global__ functions)   │
│   - runs on the GPU                          │
│   - one instance per thread                  │
│   - has access to threadIdx, blockIdx, etc   │
└──────────────────────────────────────────────┘
```

Function qualifiers you'll see:
- `__global__` — kernel, called from host, runs on device
- `__device__` — helper, called from device code only
- `__host__` — runs on CPU (default; you usually don't write this)
- `__host__ __device__` — compiles for both (useful for inline math helpers)

## Vector add — the hello world

```cuda
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // global thread index
    if (i < n) {                                    // bounds check (matters!)
        c[i] = a[i] + b[i];
    }
}
```

Launching:
```cpp
int N = 1 << 20;          // 1 million elements
int block_size = 256;
int grid_size = (N + block_size - 1) / block_size;  // ceil division
vector_add<<<grid_size, block_size>>>(a_d, b_d, c_d, N);
cudaDeviceSynchronize();  // wait for kernel to finish
```

Two things to notice:

1. **The bounds check** is non-optional. We launch `ceil(N/256)` blocks of 256 threads each. The last block has some threads with `i >= N` — without the check, they'd read past the end of the array.

2. **Launch is async.** `kernel<<<...>>>()` returns immediately; the kernel runs on the GPU in the background. `cudaDeviceSynchronize()` is the explicit wait. If you measure timing, you must sync first.

## Memory coalescing

The single most important thing about GPU memory access:

**Threads in a warp should read contiguous memory addresses.**

If thread 0 reads `a[0]`, thread 1 reads `a[1]`, ..., thread 31 reads `a[31]`, the hardware merges those into one 128-byte transaction. Coalesced. Fast.

If thread 0 reads `a[0]`, thread 1 reads `a[1024]`, ..., thread 31 reads `a[31744]`, that's 32 separate transactions. Uncoalesced. ~32× slower.

`vector_add` above is naturally coalesced because consecutive threads have consecutive `i`. Most "make this kernel fast" advice boils down to: keep your access pattern coalesced.

```
                    a[0] a[1] a[2] a[3] ... a[31]
Threads in warp:    T0   T1   T2   T3   ... T31
                    ↓    ↓    ↓    ↓        ↓
                    └──── one 128-byte fetch ────┘   ✓ coalesced

                    a[0]    a[64]   a[128]  ...
Threads in warp:    T0      T1      T2      ...
                    ↓       ↓       ↓
                    32 separate fetches            ✗ uncoalesced
```

## Vectorized loads (`float4`)

You can do better than one 4-byte load per thread. CUDA has built-in vector types:

```cuda
__global__ void relu_vec4(const float4* x, float4* y, int n_vec) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_vec) {
        float4 v = x[i];   // single 16-byte load
        v.x = fmaxf(v.x, 0.0f);
        v.y = fmaxf(v.y, 0.0f);
        v.z = fmaxf(v.z, 0.0f);
        v.w = fmaxf(v.w, 0.0f);
        y[i] = v;          // single 16-byte store
    }
}
```

Each thread now processes 4 floats. The hardware can issue 16-byte loads/stores efficiently. Fewer instructions, fewer requests, more bytes per cycle.

Caveat: alignment. `float4` requires 16-byte alignment. If your array isn't aligned, you get a runtime error or silent wrong answers. `cudaMalloc` returns aligned memory by default.

## Reductions and shared memory

Vector add and ReLU are *embarrassingly parallel* — each output depends only on its corresponding input. Softmax is harder because each output depends on the *whole row* (you need the max and the sum).

The pattern: **one block per row**, threads in the block cooperate via shared memory.

```cuda
__shared__ float smem[BLOCK_SIZE];

// Phase 1: each thread reduces its slice into one register
float thread_max = -INFINITY;
for (int j = threadIdx.x; j < N; j += BLOCK_SIZE) {
    thread_max = fmaxf(thread_max, x[row * N + j]);
}

// Phase 2: tree reduction across threads in the block via shared memory
smem[threadIdx.x] = thread_max;
__syncthreads();

for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
        smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + stride]);
    }
    __syncthreads();
}

float row_max = smem[0];   // every thread now has the max
```

Tree reduction visualization (8 threads, for clarity):

```
T0   T1   T2   T3   T4   T5   T6   T7      stride=4
 ↓    ↓    ↓    ↓                          T0 = max(T0, T4)
 max  max  max  max                        T1 = max(T1, T5)
                                           ...
T0   T1   T2   T3                          stride=2
 ↓    ↓                                    T0 = max(T0, T2)
 max  max                                  T1 = max(T1, T3)

T0   T1                                    stride=1
 ↓                                         T0 = max(T0, T1)
 max

T0       ← block-wide max in smem[0]
```

For a warp's worth of threads (32 or fewer), use **warp shuffles** instead — faster, no shared memory needed:

```cuda
// reduce within a warp using __shfl_down_sync
unsigned mask = 0xffffffff;  // all threads active
for (int offset = 16; offset > 0; offset /= 2) {
    val = fmaxf(val, __shfl_down_sync(mask, val, offset));
}
// thread 0 of the warp now has the max
```

## Online softmax (numerically stable)

Naive softmax: `exp(x_i) / sum(exp(x))`. Problem: if any `x_i > 88`, `exp(x_i)` overflows FP32.

Fix: subtract the max first.

```
softmax(x_i) = exp(x_i - m) / sum_j(exp(x_j - m))    where m = max(x)
```

This is mathematically identical (the `m` cancels) and numerically stable.

The *online* trick (we'll see this again in FlashAttention): you can compute max and sum in *one pass* by maintaining running `(m, ℓ)` and rescaling when a larger max appears:

```
init: m_old = -inf, ℓ_old = 0

for each new value x_new:
    m_new = max(m_old, x_new)
    ℓ_new = ℓ_old * exp(m_old - m_new) + exp(x_new - m_new)
    m_old, ℓ_old = m_new, ℓ_new

result: softmax(x_i) = exp(x_i - m_final) / ℓ_final
```

This is the same recursion FlashAttention uses to compute attention without materializing the full QK^T matrix. Recognize it — it'll come back hard in Topic 6.

## Setup and toolchain

Three options:

1. **Colab (free)** — easiest start. T4 GPU (SM75). Use `%%writefile kernel.cu` in a cell, then `!nvcc kernel.cu -o kernel && ./kernel`.
2. **PyTorch JIT extension** — `torch.utils.cpp_extension.load_inline()` compiles your `.cu` source on the fly and exposes it as a Python function. No setup.py. The modern way.
3. **Local + nvcc** — install CUDA Toolkit 13.x, run `nvcc -arch=sm_XX file.cu -o file`. Works if you have an NVIDIA GPU locally.

For this curriculum: pick (1) or (2). Don't waste a day on toolchain setup.

## Debugging — `compute-sanitizer`

`cuda-memcheck` was removed in CUDA 12. The replacement is `compute-sanitizer`:

```bash
nvcc -lineinfo kernel.cu -o kernel        # -lineinfo gives source attribution
compute-sanitizer ./kernel                # default tool: memcheck
compute-sanitizer --tool racecheck ./kernel
compute-sanitizer --tool synccheck ./kernel
compute-sanitizer --tool initcheck ./kernel
```

`memcheck` catches: out-of-bounds, misaligned, uninitialized memory.
`racecheck` catches: shared memory races (between threads in a block).
`synccheck` catches: divergent `__syncthreads()` calls (some threads hit it, others don't).

Always run with `compute-sanitizer` once before declaring a kernel "done." Wrong answers from CUDA are usually one of these four classes.

## Pitfalls

1. **Forgetting the bounds check.** First class of bug. Your last block has out-of-range threads. Always `if (i < n)`.
2. **Forgetting `cudaDeviceSynchronize()` before timing.** Kernel launch is async. Without sync, you measured the launch, not the work.
3. **Not checking errors.** Add a `CUDA_CHECK(cudaPeekAtLastError())` after every launch. Silent kernel launch failures are the worst kind of bug.
4. **Mismatched compute capability.** Compile with `-arch=sm_75` for T4, `sm_80` for A100, `sm_90` for H100. Wrong arch = silent fall-back to slow paths or runtime "no kernel image" errors.
5. **Ignoring the warmup.** First kernel launch includes JIT + context init. Always warm up before timing.

## References

- **Tinkerd — Writing CUDA Kernels for PyTorch** — https://tinkerd.net/blog/machine-learning/cuda-basics/  (the cleanest 2024-2026 walkthrough using `cpp_extension`)
- **PyTorch Wiki — CUDA basics** — https://github.com/pytorch/pytorch/wiki/CUDA-basics
- **NVIDIA Compute Sanitizer docs** — https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html
- **NVIDIA Programming Guide §3 (Programming Interface)** — https://docs.nvidia.com/cuda/cuda-c-programming-guide/#programming-interface
- **vLLM CUDA Core Dump debugging blog** — https://blog.vllm.ai/2025/08/11/cuda-debugging.html  (when sanitizer isn't enough)
