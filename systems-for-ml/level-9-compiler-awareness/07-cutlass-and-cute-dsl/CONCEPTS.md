# 07 — CUTLASS and CuTe DSL

CUTLASS is NVIDIA's open-source library of CUDA primitives for linear algebra — specifically GEMMs and convolutions, written as templated C++ that lets you assemble custom kernels with the same performance as cuBLAS. Repo: https://github.com/NVIDIA/cutlass.

CuTe DSL is the 2025–2026 evolution: a Python-fronted DSL (technically a Python embedding of MLIR) for tensor *layouts* and *thread/data partitioning*, replacing the most fearsome layers of CUTLASS's C++ template wizardry while compiling down to the same fast PTX.

If you remember one thing: **CUTLASS/CuTe is the kernel author's tool**. It's what FlashAttention's authors use, what FlashInfer's authors use, what people writing custom MoE GEMMs use. Triton sits one level higher (Python DSL, less control); cuBLAS sits one level higher still (just call the function, no control). Choose CUTLASS when Triton's autotuner can't find the layout you need.

## What problem CUTLASS exists to solve

A modern GEMM kernel on Hopper or Blackwell is not "matmul + maybe a bias add." It's a careful orchestration of:

```
  HBM
   │   cp.async.bulk (TMA, Tensor Memory Accelerator)
   ▼
  Shared memory (SMEM)  ← organized in swizzled layouts to avoid bank conflicts
   │   ldmatrix into registers, in the layout the wgmma instruction wants
   ▼
  Tensor core MMA (wgmma on Hopper, 5th-gen tensor cores on Blackwell)
   │   register-resident accumulator
   ▼
  Epilogue: bias add, activation, scale, downcast to FP8/FP4
   │   tma.store back to HBM
   ▼
  HBM
```

Every box is a tunable. Tile sizes, swizzle pattern, pipeline stages, warp specialization, epilogue fusion. Get them right and you hit 80–90% of peak. Get them wrong and you sit at 20%.

CUTLASS's contribution is *composing* these decisions. Instead of writing the whole kernel, you parameterize a `Gemm` template by tile shape, threadblock shape, warp shape, epilogue, and a layout descriptor. The library does the rest. The cost: deeply nested C++ templates, error messages that span pages.

## Why CuTe replaced the old way

CuTe is a layout algebra. A `Layout` is a mapping `(coord) -> offset` defined by two integer tuples — a `Shape` and a `Stride`. It can describe a contiguous tensor, a transposed view, a swizzled shared-memory layout, or how 32 threads in a warp map onto an 8x16 fragment, all in the same vocabulary.

```
Example: an 8x16 row-major tile.

  Shape  = (8, 16)
  Stride = (16, 1)

Coord (3, 5) -> 3*16 + 5*1 = 53.

Now make it column-major: Stride = (1, 8). Same shape. Coord (3, 5) -> 43.

Now make it a swizzled SMEM layout for avoiding bank conflicts —
that's just a different stride/shape composition, expressed in
the same algebra.
```

Two operations matter:

- **`composition`** — given two layouts, produce a third that is "Layout A indexed by Layout B." This is how you express "a warp-level fragment as a view into a threadblock-level shared-memory tile."
- **`tiled_divide` / `local_partition`** — slice a big tensor into a per-thread tile so each thread knows exactly which elements it owns.

These two operations subsume what used to be hundreds of lines of index arithmetic in CUDA C. They're also what makes CuTe expressible as a Python DSL — the abstraction is small enough that a Python frontend doesn't lose any expressiveness.

References:
- CuTe layout algebra docs — https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/00_quickstart.md
- The "01_layout" notebook in the CUTLASS repo (`media/docs/cute/01_layout.md`) is the cleanest introduction.

## CuTe DSL (Python) — the 2026 path

CuTe DSL is part of CUTLASS 4.x. It exposes the same layout algebra as Python, with `@cute.kernel` decorators that emit MLIR (Triton dialect equivalent, lowered to PTX).

```python
import cutlass.cute as cute

@cute.kernel
def gemm_kernel(A, B, C, alpha, beta):
    # A is (M, K), B is (K, N), C is (M, N). Layouts come from the caller.
    tA = cute.local_tile(A, tile_shape=(128, 64))   # threadblock tile of A
    tB = cute.local_tile(B, tile_shape=(64, 128))
    tC = cute.local_tile(C, tile_shape=(128, 128))

    # Stage tA, tB through shared memory with a swizzled layout.
    sA = cute.make_shared(tA, layout=cute.Swizzle(3, 3, 3))
    sB = cute.make_shared(tB, layout=cute.Swizzle(3, 3, 3))

    # Allocate a register fragment shaped for the wgmma instruction.
    rC = cute.make_fragment(tC, mma_atom=cute.MMA_F16_F16_F32_M64N128K16)

    # Pipeline: cp.async into smem, wgmma from smem, write back.
    for k_tile in cute.range(K_TILES):
        cute.copy(tA[k_tile], sA, async_=True)
        cute.copy(tB[k_tile], sB, async_=True)
        cute.cp_async_wait_all()
        cute.gemm(sA, sB, rC)

    cute.copy(rC, tC)
```

(That's a sketch — actual CuTe DSL syntax is close to this but evolves; see https://github.com/NVIDIA/cutlass/tree/main/python.)

The Python frontend is what makes CuTe DSL usable. The IR underneath is doing exactly what the C++ CUTLASS kernel did; the difference is your error messages now point at Python lines and your iteration loop is seconds instead of minutes.

## Where CUTLASS / CuTe sits in the toolchain

```
  Triton kernel (Python DSL)              CUTLASS C++ template kernel
        │                                          │
        │  Triton compiler (MLIR)                  │  nvcc + CUTLASS templates
        ▼                                          ▼
  PTX                                          PTX
        │                                          │
        ▼                                          ▼
                       SASS / GPU
```

```
  CuTe DSL kernel (Python)
        │  CuTe compiler (MLIR-based)
        ▼
  PTX  →  SASS
```

CuTe DSL is a third path that targets the same PTX, with the CUTLASS-grade tile/layout control of the C++ path and the iteration speed of the Python path.

## Who actually uses this

- **FlashAttention 3 and 4** — the production attention kernel for Hopper/Blackwell — was written using CUTLASS / CuTe abstractions. The reason FA3 outpaced naive Triton attention by 2x on H100 is largely warp-specialization patterns expressed cleanly in CUTLASS. See https://research.colfax-intl.com/cutlass-tutorial-persistent-kernels-and-stream-k/.
- **FlashInfer** — vLLM's attention backend — uses CUTLASS-derived kernels for prefill and CUTLASS templates for some decode shapes. https://github.com/flashinfer-ai/flashinfer.
- **TensorRT-LLM** — NVIDIA's own LLM inference engine — builds on CUTLASS for its custom GEMMs. https://github.com/NVIDIA/TensorRT-LLM.
- **DeepSeek's open-source kernels** (DeepGEMM, FlashMLA, released early 2025) — all CUTLASS / CuTe. https://github.com/deepseek-ai/DeepGEMM.

If you read the source of any production-fast LLM inference engine in 2026, you will see CUTLASS includes.

## When to reach for CUTLASS / CuTe vs Triton

Use Triton when:
- You want a kernel that's good enough quickly.
- The shape is well-supported by Triton's autotuner.
- Compile time and iteration speed matter more than the last 20% of perf.

Use CUTLASS / CuTe when:
- You need warp specialization patterns Triton can't express cleanly.
- You need exact control over the tensor-core instruction schedule (e.g., overlapping wgmma with cp.async.bulk in a specific way).
- You're targeting a new NVIDIA generation before Triton catches up. The Hopper TMA gap was where CUTLASS won most clearly; the Blackwell tensor-memory gap is the current frontier.
- You're publishing a kernel for the community to extend (FA3, DeepGEMM).

In 2026 the practical answer for most teams is: write in Triton, profile, and only drop to CUTLASS for the 1–2 hot kernels that need it.

## What's actually changing in 2026

- **CUTLASS 4.x** (released through 2024–2025) made CuTe the canonical way to write CUTLASS kernels. The older "thread-block / warp / mma" template towers are deprecated for new code.
- **CuTe DSL (Python)** is in active development; the Python frontend is becoming the recommended entry point for new contributors. Removes the C++ template-error pain.
- **Blackwell support** (B200/B300, 5th-gen tensor cores, FP4) lands progressively in CUTLASS through 2025–2026. The new tensor memory hierarchy on Blackwell needs new layout patterns CuTe expresses well.
- **DeepSeek's DeepGEMM** showed the open-source community can ship CUTLASS-quality kernels independent of NVIDIA — a real shift in who writes the fast inference kernels.
- **FlashAttention 4** (announced 2025) uses CuTe DSL extensively; the source is a useful read for anyone learning the abstractions in real production code.

## Reading list (in order)

1. CUTLASS README — https://github.com/NVIDIA/cutlass — 15 minutes.
2. CuTe quickstart — `media/docs/cute/00_quickstart.md` in the repo — 30 minutes.
3. CuTe layout algebra — `media/docs/cute/01_layout.md` — 1 hour.
4. One example kernel — `examples/cute/tutorial/sgemm_*.cu` — read, don't necessarily run.
5. Colfax persistent-kernels tutorial — https://research.colfax-intl.com/cutlass-tutorial-persistent-kernels-and-stream-k/ — the 2024 essay that explains why CUTLASS won on Hopper.

## Inspecting CUTLASS without writing C++

The included `inspect_cutlass.py` script walks the CUTLASS repo (if cloned) and prints a curated list of files to read first. Use it as a guided tour. The actual kernel-writing exercise is out of scope for awareness week — that's months of investment, the right project for someone going deep on this track.
