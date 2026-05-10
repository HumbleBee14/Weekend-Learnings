# 05 — Metal Shaders

## Files

- `CONCEPTS.md` — execution model, the CUDA-to-Metal mental map, simdgroup_matrix as the NA path, when to write a kernel and when not to.
- `custom_kernel.py` — a working `mx.fast.metal_kernel` for `y = 2x + 1`, side-by-side with the MLX-ops version.

## Quickstart

```bash
pip install mlx
python custom_kernel.py
```

## Expected output

```
custom Metal kernel : 18.40 ms / 200 runs
mx ops (compiled)   : 17.10 ms / 200 runs
correctness         : max diff = 0.000e+00
```

The MLX compiled path beats your hand-written kernel on this trivial elementwise op. That is the point: MLX's compiler is already good. The right time to write a custom kernel is when profiling shows MLX is doing something suboptimal for your specific shape or fusion pattern.

## Try

- Replace the kernel with a fused (gelu(x) * y) — two ops MLX may not fuse automatically.
- Add an attention-shaped kernel using `simdgroup_matrix` types — read MLX's attention.metal in the source tree first.
- Use Xcode GPU frame capture: run a model from Python, capture, inspect kernel times, find your bottleneck.

## Where this goes

This is the deepest-specialization escape hatch. Most Level 8 readers will skim this topic and never touch MSL again. That is fine — you now know what is underneath MLX, and you can read other people's Metal kernels.
