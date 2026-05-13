# What's actually happening in these kernels

You can write the three kernels in this sub-module without understanding any of this. The kernels will work. They will be slow and you won't know why. This document is the gap between "the code compiles" and "I know what every line is doing."

## Tiles, programs, and the grid

A Triton kernel is a Python function decorated with `@triton.jit`. When you call it, Triton compiles it (the first call) and launches it on the GPU. The launch takes a `grid` argument that says how many *programs* (Triton's word for thread blocks) to run.

```python
add_kernel[(grid_size,)](x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE=1024)
```

Inside the kernel, `tl.program_id(0)` tells the current program which one it is. So if you launched `grid = (4,)`, four programs run with `pid` in `{0, 1, 2, 3}`. Each program handles one tile — a slice of the data of size `BLOCK_SIZE`.

For a vector of 4096 elements with `BLOCK_SIZE = 1024`, you launch 4 programs. Program 0 handles `data[0:1024]`, program 1 handles `data[1024:2048]`, and so on.

**The kernel as a whole is the work for one tile.** You write what one program does. The grid says how many programs run. The compiler maps each program to one or more warps inside one SM. You never see warps directly unless you ask for them with `num_warps`.

## `tl.arange` and the index vector

Inside the kernel, `tl.arange(0, BLOCK_SIZE)` produces a *vector* of size `BLOCK_SIZE` containing `[0, 1, 2, ..., BLOCK_SIZE-1]`. This vector lives in registers across the warps of the program. You use it to compute the global indices you want to load:

```python
pid = tl.program_id(0)
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # indices for THIS tile
```

`offsets` is now a vector of size `BLOCK_SIZE` containing the global indices for this program's tile. Program 0 sees `[0, 1, ..., 1023]`, program 1 sees `[1024, 1025, ..., 2047]`, etc.

## `tl.load` and `mask=`

`tl.load(ptr + offsets)` reads `BLOCK_SIZE` elements from memory at the given offsets. If `offsets` is contiguous (the common case), the hardware coalesces the reads into a small number of HBM transactions. This is fast.

The issue: what if your data has 4097 elements and `BLOCK_SIZE = 1024`? You'd launch 5 programs (`ceil(4097/1024)`). Program 4 has `offsets = [4096, 4097, ..., 5119]` but only index 4096 is valid. Reading the others would access garbage past your tensor's end.

The fix is `mask=`. You compute a boolean vector saying which indices are valid, and pass it to `tl.load`:

```python
mask = offsets < n_elements
x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
```

Lanes where `mask` is `False` skip the load entirely (no garbage read), and `other=0.0` says what value to use in those lanes instead. The same idea applies to `tl.store` — out-of-range writes are suppressed.

Forgetting `mask=` is the #1 silent-correctness bug in Triton code. Your kernel "works" on power-of-2 shapes and silently corrupts memory on others.

## Why `BLOCK_SIZE` is a `tl.constexpr`

You'll see `BLOCK_SIZE: tl.constexpr` in kernel signatures. This tells Triton: "this value is known at compile time, generate specialized code for it." The compiler unrolls loops, picks register layouts, and emits tensor-core fragments based on this value. If you changed it, Triton would recompile.

Practical consequence: you can't pass `BLOCK_SIZE` from a tensor or compute it at runtime. It has to be a literal you pass at launch time. You can autotune over different `BLOCK_SIZE` values (see sub-module 04) — Triton compiles a specialized kernel per value and picks the best one.

## Online softmax — what it is and why you need it

Softmax over a row is `out[i] = exp(x[i]) / sum_j(exp(x[j]))`. Three problems with the naive version:

1. **Numerical overflow.** If any `x[j]` is large (~89 in FP32, ~17 in FP16), `exp(x[j])` is `+inf`. You divide one inf by another inf and get NaN. Fix: subtract the row max before exponentiating. `out[i] = exp(x[i] - m) / sum_j(exp(x[j] - m))` where `m = max(x)`. Same result, no overflow.
2. **Two passes over the data.** The fix above requires you to first find `m` (one pass), then compute the sum of exps (second pass), then divide (third pass). For a kernel that's memory-bound (loading the row from HBM is expensive), three passes is wasteful.
3. **The "online" insight.** You can compute `m` and `sum` in a *single pass* if you maintain them as running stats and rescale when you see a new larger element. Here's the algorithm:

```
running_max = -inf
running_sum = 0
for x in row:
    new_max = max(running_max, x)
    running_sum = running_sum * exp(running_max - new_max) + exp(x - new_max)
    running_max = new_max
# At the end: running_max is the row max, running_sum is sum(exp(x_i - running_max))
# Then second pass: out[i] = exp(x[i] - running_max) / running_sum
```

The rescaling step (`running_sum *= exp(running_max - new_max)`) corrects the running sum when a new larger max is found — old contributions get scaled down.

For our row-softmax kernel, the row is small enough to fit in one tile, so we can hold all of `x[i]` in registers. The "online" formulation isn't strictly needed there — we just `m = tl.max(x); s = tl.sum(tl.exp(x - m)); out = tl.exp(x - m) / s`. But the *idea* — running stats updated as you scan — is the foundation of every FlashAttention kernel. We meet it formally here.

For a derivation with worked numbers, see the file [`03_softmax_row.py`](03_softmax_row.py) — the docstring runs through it on a small example.

## Reductions: `tl.sum` and `tl.max`

Inside a Triton program, `tl.sum(x)` and `tl.max(x)` reduce a vector to a scalar. Under the hood, this is implemented as a tree reduction across the warps of the program, using shared memory for intermediate results. You don't write the tree — the compiler emits it.

Reductions are the reason kernels like RMSNorm, LayerNorm, softmax, and attention require a tile *per row* of the input. If you tried to put multiple rows in one tile and reduce only across the row dimension, the compiler can do it but the indexing gets gnarly. The clean pattern is: one program per row, the program's tile is the full row (or a portion of it if the row is huge), reduce across the tile.

## What happens after you write the kernel

When you call the kernel for the first time, Triton compiles it. This takes ~1 second of wall-clock and produces a cached binary in `~/.triton/cache/`. Subsequent launches with the same constexpr arguments are instant. If you change `BLOCK_SIZE` from 1024 to 2048, a new compile happens.

This matters for two reasons:

- **Benchmarking.** Your first call includes compile time. Always warm up before timing. `triton.testing.do_bench` does 25 warmup iterations by default.
- **Cache invalidation.** If you edit the kernel source, Triton hashes the source and recompiles. If you do something weird like `monkey-patch a constexpr from a closure variable`, Triton might miss the change. When in doubt: `rm -rf ~/.triton/cache`.

That's enough theory. Open the files and run them.
