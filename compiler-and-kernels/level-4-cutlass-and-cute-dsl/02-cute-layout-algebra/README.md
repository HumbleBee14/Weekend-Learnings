# 02 — CuTe layout algebra, derived

> Outer: [`../README.md`](../README.md) · Hardware: none. Pure Python.

If you remember one thing from this level, remember this:

> **A CuTe Layout is a function from coordinates to integers.** Given `Layout = (shape, stride)`, applying it to a coordinate `c` produces the memory offset `inner_product(c, stride)`.

Every operation in CuTe — composition, coalesce, divide, complement, product, swizzle — is defined to preserve or transform this function in a specific way. Nothing is magic. We derive it.

The work in this submodule:

1. Read this README. Do the worked examples with pencil and paper. Verify each step.
2. Open [`cute_algebra.py`](cute_algebra.py). It's a ~250-line pure-Python implementation of Layout, composition, coalesce, divide, and swizzle. The worked examples are also `pytest` cases.
3. Run `pytest cute_algebra.py -v`. All tests pass on a clean checkout.
4. Modify inputs in the worked examples; predict outputs; verify. If you ever predict wrong, stop and re-derive.

This is identical pedagogy to Level 1's "bandwidth journey" — same operation, written multiple ways, with the prediction-then-verify loop. The algebra has to land in your fingers, not just your eyes.

## The function

A Layout `L = (S, D)` where `S` is the shape and `D` is the stride (both tuples of the same nested structure). For a coordinate `c = (c₀, c₁, ..., cₙ)`:

```
L(c) = c₀·D₀ + c₁·D₁ + ... + cₙ·Dₙ
```

The shape tells you the valid range of each coordinate (`0 ≤ cᵢ < Sᵢ`). The stride tells you how the offset moves when you step that coordinate.

Two ways to read a Layout:
- **As a memory access pattern.** "How do I find element `[i,j]` in this tensor?"
- **As a function on integers.** Coordinates can also be flat: `L(k)` for a 1D index `k` traverses the shape in column-major order by default. This duality is what makes composition work.

### Worked example 1 — basics

```
Layout A = (4, 8) : (8, 1)        # 4×8 row-major
A(2, 3) = 2·8 + 3·1 = 19          # element [2,3] is at offset 19
size(A) = 4·8 = 32                # 32 elements total
cosize(A) = max offset + 1 = 32   # uses contiguous 32 ints
```

```
Layout B = (4, 8) : (1, 4)        # 4×8 column-major
B(2, 3) = 2·1 + 3·4 = 14
```

Same shape, different layout. The bytes on disk are the same; the function is different.

Picture what these two layouts actually do on a 2×3 example (`shape=(2,3)`, contrasting strides `(3,1)` row-major vs `(1,2)` column-major):

```
Logical coords (i,j):           Flat memory (12 cells shown, * = used):

   j=0  j=1  j=2
 ┌─────┬─────┬─────┐
i=0 (0,0)(0,1)(0,2)
 ├─────┼─────┼─────┤
i=1 (1,0)(1,1)(1,2)
 └─────┴─────┴─────┘

Layout (2,3):(3,1)  — row-major, stride=(3,1)
   (i,j) → i·3 + j·1
   (0,0)→0  (0,1)→1  (0,2)→2
   (1,0)→3  (1,1)→4  (1,2)→5

   addr:  0   1   2   3   4   5
         ┌───┬───┬───┬───┬───┬───┐
         │0,0│0,1│0,2│1,0│1,1│1,2│   row 0 first, then row 1
         └───┴───┴───┴───┴───┴───┘

Layout (2,3):(1,2)  — column-major, stride=(1,2)
   (i,j) → i·1 + j·2
   (0,0)→0  (0,1)→2  (0,2)→4
   (1,0)→1  (1,1)→3  (1,2)→5

   addr:  0   1   2   3   4   5
         ┌───┬───┬───┬───┬───┬───┐
         │0,0│1,0│0,1│1,1│0,2│1,2│   col 0 first, then col 1, then col 2
         └───┴───┴───┴───┴───┴───┘
```

*Same shape, same six elements, two different functions from coordinate to address.*

### Worked example 2 — nested shapes for tiling

A 64×64 matrix in row-major BF16 storage, viewed as a 4×4 grid of 16×16 tiles:

```
Layout T = ((16, 4), (16, 4)) : ((1, 256), (64, 16384))
```

Read the modes left-to-right, inside-out:
- Outer-M index `M_outer ∈ [0, 4)`: step `256` elements (16 rows × 16 tile-width = ... wait, 64 cols × 16 rows = 1024, not 256). The strides in CuTe are *not* re-derived from shape — they are *given* and the algebra preserves them. The layout I wrote is wrong for a 4×4 tiling of a row-major 64×64. Let me redo this.

The row-major 64×64 storage has layout `(64, 64) : (64, 1)`. Tiling into 16×16 blocks, the *natural* nested layout that matches the same memory:

```
T = ((16, 4), (16, 4)) : ((64, 1024), (1, 16))
```

Decoding: inner M (`m_in ∈ [0, 16)`) strides by 64 (one row of the matrix). Outer M (`m_out ∈ [0, 4)`) strides by `16·64 = 1024` (16 rows = one block-row). Inner N (`n_in ∈ [0, 16)`) strides by 1 (one column). Outer N (`n_out ∈ [0, 4)`) strides by 16 (one block-column).

The lesson: **stride is given by what memory the layout points at, not by the shape.** The shape tells you the *structure of indexing*; the stride tells you what each step costs.

`cute_algebra.py` has this exact example as `test_64x64_tile_nesting()`. Run it; print the offsets for a few coordinates; convince yourself the formula is right.

### Composition — the rule that does the work

```
A ∘ B = R, defined by R(c) = A(B(c))
```

You apply B first to get an intermediate offset, then feed that into A. The trick is that A's coordinates are "integers in A's domain" — so B has to produce values in `[0, size(A))`. CuTe's composition is well-defined when B's image fits in A's domain.

**Integral case.** When A and B are both single-mode (one shape element each):

```
A = a:b, B = s:d
R = A ∘ B = s : (b·d)
```

Proof in one line: `R(k) = A(B(k)) = A(k·d) = (k·d)·b = k·(b·d)`. The shape comes from B (R has `s` elements). The stride is the product of strides.

**Multi-mode case.** Distribute over modes of B:

```
A ∘ (B₀, B₁) = (A ∘ B₀, A ∘ B₁)
```

You break B apart, compose each piece with A, then concatenate.

### Worked example 3 — composition

```
A = (6, 2) : (8, 2)
B = (4, 3) : (3, 1)

# Step 1: decompose B
B = (4:3, 3:1)

# Step 2: distribute
A ∘ B = (A ∘ 4:3, A ∘ 3:1)

# Step 3: compose each piece
A ∘ 4:3:
  We need to apply A to coordinates 0, 3, 6, 9 (the integral coords).
  But A's flat domain has size 12. Flat coord 3 in A (which has shape (6,2) traversed column-major as default) corresponds to (3, 0). A(3,0) = 24. So R₀(1) = 24.
  Simpler: A ∘ k:1 walks the flat domain of A from 0 in steps of k. Stride of A in its flat-traversal order is what comes out.
  Concretely, A = (6,2):(8,2). Its flat-traversal stride is (8, 2) — flat coord 0..5 stride by 8, then flat coord 6..11 starts at the next column with stride 2. Applying B = 4:3 means take coords 0, 3, 6, 9. These cross the column boundary.
  Result: A ∘ 4:3 = (2, 2) : (24, 2).
  Read this as: "two groups of two; first group strides by 24, second by 2." This expresses crossing the column boundary at the right place.

A ∘ 3:1:
  Take flat coords 0, 1, 2 of A. All in the first column. Stride 8.
  Result: A ∘ 3:1 = 3 : 8.

# Step 4: concatenate
A ∘ B = ((2, 2), 3) : ((24, 2), 8)
```

This is the canonical multi-mode composition example from the CuTe docs. The `cute_algebra.py` test `test_composition_canonical()` reproduces it. Run with `-v -s` and watch the intermediate values print.

### Coalesce — simplification without changing the function

A layout is **coalesced** when no further simplification preserves the function. The rules:

1. **Drop size-1 modes:** `s:d ++ 1:d' → s:d` (the `1:d'` mode contributes nothing because coord is always 0).
2. **Merge contiguous modes:** if `s₁·d₁ == d₂`, then `s₁:d₁ ++ s₂:d₂ → (s₁·s₂):d₁`.

```
(2, 1, 6) : (1, 6, 2)
  drop the 1-mode → (2, 6) : (1, 2)
  is 2·1 == 2? yes → merge → 12 : 1
```

After coalesce: `12:1`. The function is still the identity on 0..11.

### Divide — the operation that tiles

`logical_divide(A, B)` splits A into "what B picks out" and "the rest."

```
A ÷ B = A ∘ (B, complement(B, size(A)))
```

You use this every time you tile a matrix into blocks. For a 1D example:

```
A = 24 : 1    (24 contiguous elements)
B = 4 : 1     (a tile of 4 contiguous elements)

complement(B, 24) = 6 : 4
  (6 tiles, stride 4 — each tile starts every 4th element)

A ÷ B = ((4:1), (6:4))
  mode 0 is the tile (4 elements within one tile)
  mode 1 indexes the 6 tiles
```

For the 2D matrix case, you `zipped_divide(A, tiler)` where `tiler` is a (BLOCK_M, BLOCK_N) shape. The result is `((BLOCK_M, BLOCK_N), (num_tiles_M, num_tiles_N))` — first mode is intra-tile, second is inter-tile. This is how every CuTe GEMM partitions its work.

### Swizzle — bank-conflict elimination

Shared memory on NVIDIA GPUs is 32 banks of 4 bytes. Two threads in the same warp accessing the same bank serialize. A naive 16-bf16-wide tile (32 bytes per row) hits bank conflicts on a transposed access.

The fix: XOR-based swizzling. `Swizzle<BBits, MBase, SShift>(addr)` returns `addr XOR ((addr >> SShift) & ((1 << BBits) - 1)) << MBase`. The XOR shifts each row's bank assignment so threads land on different banks.

The canonical patterns:

| Pattern | BBits, MBase, SShift | Element type | Use |
|---|---|---|---|
| Swizzle<0,4,3> | none | any | trivial, no swizzle |
| Swizzle<1,4,3> | 32B | FP32 | 32-byte tile rows |
| Swizzle<2,4,3> | 64B | FP16/BF16 (32 wide) | 64-byte tile rows |
| Swizzle<3,4,3> | 128B | BF16 (64 wide) | 128-byte tile rows — most BF16 GEMMs |

For a 64×64 BF16 SMEM tile (128 bytes per row), use `Swizzle<3,4,3>`. The MMA atom you pair with must expect 128B-swizzled SMEM; that pairing is documented in the CUTLASS examples and codified in `TiledMMA::SmemLayoutAtom`.

The `Swizzle` class in `cute_algebra.py` implements the XOR rule and the test `test_swizzle_no_bank_conflicts()` checks that a 32-thread warp accessing a swizzled tile lands on 32 distinct banks.

## Worked example 4 — a complete tile description

Putting it together: describe the SMEM tile for the A operand of a BF16 GEMM on Hopper, BLOCK_M=128, BLOCK_K=64.

```
# Global memory: row-major M×K
gA_layout = (M, K) : (K, 1)

# We want to TMA-load a (128, 64) box into SMEM with 128B swizzle
box = (128, 64)
sA_layout = composition(swizzle<3,4,3>, (128, 64) : (64, 1))
  # 128 rows, 64 cols, row-major within, swizzled XOR pattern applied
```

This layout description is exactly what you hand to `cute.make_tensor_descriptor` for the TMA, and to the MMA atom for the WGMMA load. The composition rules let you derive the swizzled SMEM layout from the unswizzled tile shape — you don't write the swizzle by hand.

## Run the algebra module

```bash
cd 02-cute-layout-algebra
pip install pytest      # nothing else; the module is pure-Python
pytest cute_algebra.py -v
```

Tests cover:

- `test_basic_indexing` — Layout `(4,8):(8,1)` maps `(2,3) → 19`
- `test_64x64_tile_nesting` — nested layout offsets match flat row-major
- `test_composition_integral` — `a:b ∘ s:d = s:(b·d)`
- `test_composition_canonical` — the `(6,2):(8,2) ∘ (4,3):(3,1)` example
- `test_coalesce_size_one` — `(2,1,6):(1,6,2)` coalesces to `12:1`
- `test_coalesce_merge_contiguous` — adjacent contiguous modes merge
- `test_divide_1d` — `24:1 ÷ 4:1 = ((4:1),(6:4))`
- `test_swizzle_no_bank_conflicts` — 32-thread warp hits 32 banks

If you change a stride in `test_composition_canonical` and the result doesn't match what the formula predicts, the test should fail. Use that as a sanity check.

## What you should be able to do next

- Given any `(shape, stride)`, compute `Layout(c)` for any coordinate.
- Given two layouts `A` and `B`, compose them by hand for small sizes (≤ 16 elements).
- Coalesce any layout to canonical form.
- Pick the right swizzle for a tile width in bytes.
- Read CUTLASS example files (`hopper/dense_gemm_persistent.py`) and identify the SMEM layout as a composition of a base layout and a swizzle.

Submodule 03 puts the algebra to work in a real CuTe-DSL kernel.

## References

- [CuTe Layout Algebra docs](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/02_layout_algebra.html) — the canonical source.
- [Cris Cecka — CuTe Layout Representation and Algebra (arXiv 2603.02298)](https://arxiv.org/abs/2603.02298) — the paper. Reads well after you've done the examples.
- [Lei Mao — CuTe Layout Algebra](https://leimao.github.io/article/CuTe-Layout-Algebra/) — clearer than the docs in places, with diagrams.
- [Jay Shah — A note on the algebra of CuTe Layouts (PDF)](https://leimao.github.io/downloads/article/2024-10-20-CuTe-Layout-Algebra/layout_algebra.pdf).
- [Lei Mao — CuTe Swizzle](https://leimao.github.io/blog/CuTe-Swizzle/) — the XOR construction.
- [Simon Veitner — Understanding CuTe Swizzling](https://veitner.bearblog.dev/understanding-cute-swizzling-the-math-behind-32b-64b-and-128b-patterns/) — the canonical patterns derived.
- [Colfax — Categorical Foundations for CuTe Layouts](https://research.colfax-intl.com/categorical-foundations-for-cute-layouts/) — optional math depth.
