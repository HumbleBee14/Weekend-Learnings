# 04 — Reading Inductor output

You compile a small forward block. Inductor dumps an `output_code.py`. You read it, line by line, until the question "what did `torch.compile` actually do?" has a concrete answer for this code.

This is the demystifier. Most learners' first reading of an Inductor dump is the moment torch.compile stops being magic.

## Hardware

T4 (or any CUDA GPU) is ideal — you'll see Triton kernels. CPU works too; Inductor will emit a C++ file and you read that instead. The CPU output is shorter and slightly easier to read for first-time learners. The GPU output is what matters in practice.

## What to run

```bash
TORCH_COMPILE_DEBUG=1 python compile_and_dump.py
```

`TORCH_COMPILE_DEBUG=1` is the magic env var: Inductor writes everything to `/tmp/torchinductor_<user>/` and prints the path. Find the `output_code.py` for the compiled function. It is a real Python file you can read.

The script in [`compile_and_dump.py`](compile_and_dump.py) is a `Linear → RMSNorm → GeLU → Linear` micro-block. The point is to be small enough that the resulting `output_code.py` is short, and rich enough that you can see fusion (RMSNorm+GeLU should fuse) and a matmul template choice.

## What you should look for

In `output_code.py` you will find:

1. **One or two `@triton.jit` kernels** (GPU) or `kernel_cpp_*` functions (CPU). For our example: typically one fused kernel for `RMSNorm + GeLU` (because they share the same elementwise iteration space and the reduction can be folded in), plus a separate matmul call.

2. **The `call(args)` function** at the bottom. This is the dispatcher that runs every step. Read it: it allocates output buffers, calls the kernels with the right grid/block sizes, and returns the outputs.

3. **Matmul backend choice**. You will see one of: `torch.ops.aten.mm.default(...)` (cuBLAS via ATen), a `triton_tem_*` template (Triton GEMM), or a `cutlass_*` call (CUTLASS template, if enabled). Inductor picked this based on its autotune cache.

4. **The grid expressions**. `grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), ...)`. These are the launch parameters.

5. **`empty_strided_cuda(...)` calls**. These are the output buffer allocations. One per intermediate tensor. If you see many: fusion didn't take and you're round-tripping HBM.

Compare to a hand-written equivalent (you wrote RMSNorm in Level 1 sub-module 03). Write three things in [`notes.md`](notes.md):
1. Which ops Inductor fused and which it didn't, with reasons.
2. The matmul backend it chose and whether you would have chosen the same.
3. One thing Inductor did well, one thing your hand-written version does better (or matches), at least one thing you didn't expect.

## Reference

- [TORCH_LOGS / debug recipe](https://docs.pytorch.org/tutorials/recipes/torch_logs.html).
- [Inductor codegen source: `torch/_inductor/codegen/triton.py`](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/codegen/triton.py).
- [TorchInductor design doc](https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747).

## Common pitfalls

- **You can't find `output_code.py`.** The path Inductor printed is the dir; the file is inside a subdirectory named with the hash of the FX graph. `find /tmp/torchinductor_$USER -name output_code.py` finds them all.
- **The Triton kernel is huge and you can't read it.** Inductor inlines a lot. Focus on the body of the `triton.jit`'d function and ignore the prologue/epilogue boilerplate at first read.
- **You read it once and called it done.** Read three or four different `output_code.py` files (for different shapes, dtypes, models) before you call this sub-module finished. Pattern recognition takes more than one example.
