# CuTe layout algebra — three concrete examples

CuTe's whole abstraction collapses into one idea: a `Layout` is `(Shape, Stride)`, and `(coord) -> offset` is computed by inner product. Every memory pattern in a GPU kernel — global tensor, swizzled shared memory, register fragment, thread-data partition — is a Layout.

Once that lands, the rest of CUTLASS reads as composition of these layouts.

## Example 1 — a contiguous row-major tile

```
8 rows × 16 cols, row-major.

  Shape  = (8, 16)
  Stride = (16, 1)

  offset(r, c) = r*16 + c*1
  offset(3, 5) = 53
```

If you reshaped this into 4 row-blocks of 2 rows each:

```
  Shape  = ((2, 4), 16)        # nested shape
  Stride = ((16, 32), 1)       # nested stride: inner stride 16, outer stride 32

  Coord  ((row_in_block, block_idx), col)
  offset((1, 2), 5) = 1*16 + 2*32 + 5*1 = 85
```

Same memory, viewed as four blocks. No data moved. Just a different layout.

## Example 2 — swizzled shared-memory layout (avoiding bank conflicts)

Shared memory on NVIDIA GPUs is organized into 32 banks of 4 bytes. If 32 threads in a warp all hit the same bank with different addresses, the loads serialize. Swizzling permutes addresses so that consecutive threads hit consecutive banks.

CuTe expresses swizzles as a third parameter: `Swizzle(B, M, S)` where:
- `B` = number of bits to swap
- `M` = mask offset
- `S` = shift amount

```
Layout = composition( Swizzle(3, 3, 3), Layout((8, 64), (64, 1)) )
```

The `composition` operation says: "to find the offset for coord (r, c), first compute r*64 + c via the inner Layout, then xor it with bits selected by Swizzle(3,3,3)." The result is the bank-conflict-free address.

You don't have to derive the Swizzle parameters by hand. CUTLASS gives you `Layout::Swizzle<...>` types matched to common tile shapes. The point is you *can* express any access pattern this way.

## Example 3 — a warp fragment matching `wgmma`

Hopper's `wgmma.m64n128k16` instruction expects the A matrix in registers laid out across a warpgroup (128 threads = 4 warps) in a specific pattern. Each thread holds a fragment of (say) 8 elements. The mapping from `(warp, lane, element)` to logical `(row, col)` of the A tile is fixed by the hardware.

In CuTe:

```
  ThrLayout = Layout((4, 32), ...)         # warp x lane in a warpgroup
  ValLayout = Layout(8, ...)               # 8 elements per thread
  TiledMMA  = composition(ThrLayout, ValLayout, mma_atom = SM90_64x128x16_F16F16F32_SS)
```

The `mma_atom` knows the hardware's data-layout requirement. The TiledMMA layout describes how each thread's 8 elements correspond to logical `(row, col)` of the 64x16 A tile. When you call `cute::gemm(tCsA, tCsB, tCrC)`, CUTLASS uses these layouts to issue the right `wgmma` with the right operands without you computing any indices.

## Why this matters

Before CuTe: every CUTLASS kernel hand-coded these mappings as nested template parameters, with comments explaining the math. Errors were template instantiation failures.

After CuTe: layouts compose. You describe what you want, CUTLASS figures out the indices. Template errors are still possible, but they're errors *about layouts*, not errors about thread offsets.

The same algebra is what the CuTe DSL (Python) exposes — a Python embedding of the layout algebra, with `@cute.kernel` emitting MLIR. The C++ and Python paths share the IR underneath; the choice is purely about iteration speed and error messages.

## Source pointers

In the CUTLASS repo:
- `include/cute/layout.hpp` — the C++ layout type. The math is in `composition` and `complement`.
- `include/cute/swizzle.hpp` — the swizzle algebra.
- `include/cute/atom/mma_atom.hpp` — hardware-instruction descriptors.
- `media/docs/cute/01_layout.md` — the human-facing introduction.

In the wider ecosystem:
- FlashAttention 3 source — `csrc/flash_attn/src/flash_fwd_kernel.h` (https://github.com/Dao-AILab/flash-attention) — production CUTLASS+CuTe.
- DeepGEMM — https://github.com/deepseek-ai/DeepGEMM — minimal, readable CuTe kernels.
