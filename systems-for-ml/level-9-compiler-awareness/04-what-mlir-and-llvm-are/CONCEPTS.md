# 04 — What MLIR and LLVM Are

The two terms get blurred together. They are not the same project, not the same generation, not the same problem.

## LLVM in one paragraph

LLVM is a compiler infrastructure project from the early 2000s. Its core deliverable is **LLVM IR**: a single, low-level, SSA-form intermediate representation that looks roughly like a typed RISC assembly with infinite virtual registers. Every modern systems compiler (Clang for C/C++, rustc, swiftc, the GCC successor projects) eventually lowers its frontend into LLVM IR, runs the LLVM optimization passes, and emits machine code via the LLVM backend for the target architecture. LLVM IR is **flat** — one set of instructions, one type system, one optimization pipeline.

```
C/C++/Rust/Swift  →  Clang/rustc/swiftc  →  LLVM IR  →  LLVM passes  →  x86 / arm64 / RISC-V / ...
```

References:
- LLVM project — https://llvm.org/
- LLVM IR reference — https://llvm.org/docs/LangRef.html
- Kaleidoscope tutorial (build a tiny language frontend on LLVM) — https://llvm.org/docs/tutorial/

## The problem ML hit

LLVM IR is too low-level for ML. You can express a matmul in LLVM IR — it's just a loop nest with some FMAs — but by the time it's down there, the compiler has lost the information it needs to choose a tile size, fuse with a downstream activation, or pick a tensor-core instruction. High-level optimizations need high-level information, and "this is a matmul of these shapes that should be tiled differently on H100 vs A100" is not the kind of thing LLVM IR can carry.

Two responses to this:

1. **Domain-specific compilers.** XLA had its own IR (HLO). TVM had Relay/Halide IR. Glow had its own IR. PyTorch's first generation (TorchScript) had its own IR. Each was a separate codebase with its own pass infrastructure.
2. **A unified framework.** What if there were one IR system that supported many *coexisting* dialects, with progressive lowering from high-level (matmul, conv) all the way down to LLVM-level scalars? That's MLIR.

## MLIR in one paragraph

MLIR (Multi-Level IR) is a compiler infrastructure project that ships with LLVM and was designed by the same people. Its core idea: a single SSA IR substrate that supports many **dialects** — each dialect is a self-contained set of operations and types, and a single MLIR module can mix multiple dialects. Compilers built on MLIR define their own dialects (or reuse standard ones) and write **passes** that transform the program — typically lowering operations from a high-level dialect to a lower-level one until the bottom of the chain emits LLVM IR or machine code.

```
high-level dialects (stablehlo, tosa, linalg)
        │  passes
        ▼
mid-level dialects (affine, scf, vector, gpu)
        │  passes
        ▼
low-level dialects (memref, llvm, nvgpu, spirv)
        │  passes
        ▼
LLVM IR / PTX / SPIR-V / machine code
```

The win: every project building an ML compiler can share the pass infrastructure, the IR machinery, the parser/printer, the testing tooling, and many of the standard dialects. A new accelerator company defines a small handful of dialects specific to their hardware and reuses everything above that.

References:
- MLIR project — https://mlir.llvm.org/
- "Why MLIR" — https://mlir.llvm.org/docs/Rationale/Rationale/
- Toy tutorial (build a tiny ML compiler with MLIR) — https://mlir.llvm.org/docs/Tutorials/Toy/
- The MLIR paper (Lattner et al., 2020) — https://arxiv.org/abs/2002.11054

## Dialects relevant to ML

These are the ones you'll see referenced in PyTorch, JAX, IREE, Triton, and tt-mlir issue trackers:

| Dialect | What it is | Where it shows up |
|---|---|---|
| `stablehlo` | Vendor-neutral high-level ML ops (matmul, conv, reduce, etc.) | JAX/XLA, PyTorch via torch_xla, IREE input |
| `tosa` | Tensor Operator Set Architecture; another high-level neutral set | TFLite, IREE, hardware vendors |
| `linalg` | Generic structured ops on tensors (a typed loop-nest abstraction) | Almost every MLIR-based compiler — the "go-to" mid-level |
| `tensor` | First-class tensor type with structured ops (extract, insert, pad) | Sits with linalg as the working pair for tensor-level transforms |
| `affine` | Affine loop nests; supports polyhedral analysis | Older Halide/TVM-style optimizations |
| `scf` | Structured control flow (`for`, `if`, `while`) | After loops are materialized from linalg |
| `vector` | Vector types and ops, lowering to SIMD/SIMT | Mid-level, just above hardware |
| `gpu` | Generic GPU primitives (launch, thread/block IDs) | Vendor-neutral GPU lowering |
| `nvgpu` | NVIDIA-specific (TMA, mma, async copies) | NVIDIA-specific GPU lowering |
| `spirv` | SPIR-V dialect (Vulkan / OpenCL bytecode) | IREE Vulkan/Metal targets |
| `llvm` | LLVM IR as a dialect | The bottom; emit via LLVM toolchain |
| `transform` | A meta-dialect that *describes* IR rewrites as IR | Schedule-as-IR; used by recent PyTorch IR rewriters |
| `triton` / `ttgpu` | Triton's IRs | Triton compiler internals |

The conceptual move that makes MLIR work: **a pass is just an IR rewriter from one set of dialects to another.** Lowering is just a sequence of passes. Optimization is just a sequence of passes. Hardware-specific specialization is just a sequence of passes. Everything reuses the same infrastructure.

References:
- linalg dialect — https://mlir.llvm.org/docs/Dialects/Linalg/
- gpu dialect — https://mlir.llvm.org/docs/Dialects/GPU/
- transform dialect — https://mlir.llvm.org/docs/Dialects/Transform/

## The "progressive lowering" mental model

A canonical lowering chain for an ML kernel:

```
stablehlo.dot_general              ← "this is a matmul"
        │  stablehlo -> linalg
        ▼
linalg.matmul                       ← "this is a structured matmul op"
        │  linalg -> linalg with tiling
        ▼
linalg.matmul (tiled)               ← "tiled with these block sizes"
        │  linalg -> scf + vector
        ▼
scf.for + vector.contract           ← "these are the loops, this is the inner contraction"
        │  vector -> nvgpu / gpu / llvm
        ▼
nvgpu.mma + gpu.launch              ← "use tensor cores, launch this many threads"
        │  -> llvm
        ▼
llvm.func + llvm.call (intrinsics)  ← LLVM IR with NVPTX intrinsics
        │  LLVM backend
        ▼
PTX                                  ← NVIDIA assembly
```

Every arrow is one or more passes. Every box is a real, inspectable IR snapshot. You can run `mlir-opt` (or whatever the project's `*-opt` tool is) with `--mlir-print-ir-after-all` and watch the program transform step by step.

## Why this matters to a non-compiler-engineer

You almost never write MLIR by hand. You will, however, encounter it:

- **Reading an issue** in vLLM / IREE / tt-mlir / Triton. Bug reports include MLIR snippets. Knowing what dialect you're looking at and where it sits in the chain is enough to follow the conversation.
- **Diagnosing a missed fusion or a wrong layout.** When `torch.compile` or IREE produces slow code, the dump-after-all-passes IR is the first place to look. Spotting where a high-level op got prematurely lowered to scalars tells you which pass to file a bug against.
- **Targeting a new accelerator.** When a vendor says "we have a PJRT plugin and a tt-mlir backend," they mean: their hardware is reachable from the StableHLO frontend through their MLIR pipeline. You need to know that's a real path, not vapor.

## Common confusions, cleared up

- **MLIR is not a competitor to LLVM.** It sits *above* LLVM and emits LLVM IR (or PTX/SPIR-V) at the bottom. Both ship from the same monorepo.
- **A "dialect" is not a language.** It's a set of operations and types within MLIR. Dialects compose freely within a single module.
- **MLIR is not just for ML.** It's used in CIRCT (hardware design), Flang (Fortran), Mojo (general-purpose), and elsewhere. ML happens to be its biggest user.
- **Triton is not an alternative to MLIR.** Triton's compiler is *built on* MLIR. The TTIR and TTGIR IRs are MLIR dialects.
- **StableHLO is not "Google's IR" anymore.** It's an OpenXLA project with multi-vendor governance. Its spec is the contract.

## What to walk away with

- LLVM IR is one IR; MLIR is many cooperating dialects.
- The "multi-level" in MLIR is the load-bearing word — you don't lose high-level structure prematurely.
- Almost every modern ML compiler is built on MLIR by 2026: Triton, IREE, OpenXLA's ingest path, tt-mlir, Mojo, MLC-LLM's lowering.
- You don't write MLIR; you read it when something goes wrong, and you trust that the lowering chain underneath your tools is doing what its passes claim to do.
