# 05 — Metal Shaders

## What this is

Metal Shading Language (MSL) is Apple's GPU compute language. C++14 dialect, with intrinsics for thread/threadgroup/SIMD-group execution. It is the CUDA C++ of the Apple world.

You will almost never write MSL yourself for an LLM. MLX's compiled kernels and `mx.fast` paths cover the standard ops, and Apple's matmul/attention paths target Neural Accelerators internally. Reach for MSL when:

- You need a custom attention variant (sliding window, MoE routing, RingAttention) not yet in MLX.
- You are writing a fused agentic-loop primitive — sample + tokenize + early-exit in one kernel.
- You want to eliminate Python overhead in a tight inner loop.

For everyone else, knowing the boundary exists is the deliverable.

## The execution model

```
device  >  command queue  >  command buffer  >  compute encoder  >  kernel
                                                                     │
                                                                     ▼
                                       grid of threadgroups, each a 3D block
                                       of threads, organized in SIMD-groups
                                       (32 threads each on Apple GPUs)
```

Compared to CUDA:

```
CUDA          Metal
──────        ──────
warp          SIMD-group       (32 threads on Apple, 32 on NVIDIA)
block         threadgroup
grid          grid
__shared__    threadgroup memory
__device__    device function
```

Most CUDA mental models port one-to-one. The differences are mostly in API surface (Objective-C++ / Swift host code, `[[buffer(0)]]` attribute syntax in MSL).

## A minimal kernel

```metal
#include <metal_stdlib>
using namespace metal;

kernel void add_one(
    device const float* in   [[buffer(0)]],
    device       float* out  [[buffer(1)]],
    uint                 tid [[thread_position_in_grid]])
{
    out[tid] = in[tid] + 1.0f;
}
```

Compile, dispatch from Swift or Python (via `mlx.core.metal`). Inputs are device buffers — on Apple Silicon that means shared DRAM, no copy.

## SIMD-group matrix — the NA path

For matmul-shaped kernels on M5 you want NAs. The mechanism is `simdgroup_matrix`:

```metal
#include <metal_stdlib>
#include <metal_simdgroup_matrix>

kernel void simdgroup_matmul(
    device const half*  A [[buffer(0)]],
    device const half*  B [[buffer(1)]],
    device       float* C [[buffer(2)]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint  sid [[simdgroup_index_in_threadgroup]])
{
    simdgroup_matrix<half, 8, 8>  Asg;
    simdgroup_matrix<half, 8, 8>  Bsg;
    simdgroup_matrix<float, 8, 8> Csg(0.0f);

    simdgroup_load(Asg, /*ptr=*/...);
    simdgroup_load(Bsg, /*ptr=*/...);
    simdgroup_multiply_accumulate(Csg, Asg, Bsg);
    simdgroup_store(Csg, /*ptr=*/...);
}
```

On M5 this instruction issues to the Neural Accelerator. On M3/M4 it falls back to the FMA path. Same code, different perf.

## Calling a custom kernel from MLX

MLX exposes a custom-kernel API so you can stay in Python and not touch Swift:

```python
import mlx.core as mx

source = """
kernel void add_one(
    device const float* in  [[buffer(0)]],
    device       float* out [[buffer(1)]],
    uint                 tid [[thread_position_in_grid]])
{
    out[tid] = in[tid] + 1.0f;
}
"""

kernel = mx.fast.metal_kernel(
    name="add_one",
    input_names=["in"],
    output_names=["out"],
    source=source,
)

x = mx.arange(16, dtype=mx.float32)
(y,) = kernel(inputs=[x], output_shapes=[(16,)], output_dtypes=[mx.float32], grid=(16, 1, 1), threadgroup=(16, 1, 1))
mx.eval(y)
print(y)  # [1, 2, ..., 16]
```

This is the lightest-touch way to write Metal for MLX models. The MLX repo has examples that go up to attention kernels.

## Threadgroup memory and tiling

Threadgroup memory (`threadgroup` qualifier) is the equivalent of CUDA shared memory. ~32 KB per threadgroup on most Apple GPUs. The matmul tiling story is the same as in Levels 2-4: load a tile of A and B into threadgroup memory, multiply-accumulate, write out a tile of C. With `simdgroup_matrix`, much of this tiling is hidden inside the intrinsic.

## When the cost is not worth it

A custom kernel competes against MLX's compiled kernels, which Apple's MLX team and a community of contributors have spent years tuning. Beating them takes real work and benchmarking discipline. For most hot paths, the better strategy is:

1. Express the math in `mx` ops, run it.
2. Profile with Xcode GPU frame capture or `mx.metal.start_capture` / `stop_capture`.
3. Fuse or rewrite only the kernel that dominates the trace.

Premature Metal kernel writing is the same trap as premature CUDA C++ — the framework has probably already done it better.

## References

- Metal Shading Language Specification (2026): https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
- MLX custom Metal kernels: https://ml-explore.github.io/mlx/build/html/dev/extensions.html
- MLX `mx.fast.metal_kernel`: https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.metal_kernel.html
- Metal Performance Shaders Graph: https://developer.apple.com/documentation/metalperformanceshadersgraph
- Xcode GPU frame capture: https://developer.apple.com/documentation/metal/debugging_tools/viewing_your_gpu_workload_with_the_metal_debugger
