# Level 6 — MLIR in Practice

> Outer reference: [`compiler-and-kernels/README.md`](../README.md) · Project: out-of-tree MLIR pass — tile a `linalg.matmul`; run through IREE CPU + Metal

## Week goal

`systems-for-ml` Level 9 was a one-week tour of MLIR. This week you actually write code. By Friday you should be able to:

- Build LLVM/MLIR from source and run `mlir-opt` on real IR
- Write an out-of-tree MLIR optimization pass using `PatternRewriter`
- Understand the `linalg` → `loops` → `llvm` lowering chain for a matmul
- Follow Tenstorrent's TT-Forge lowering stack: PyTorch → StableHLO → TTIR → TTNN → Metalium
- Deploy a model through IREE to CPU and Metal (your M5 Mac runs Metal natively)

## Where this fits

- **Comes after:** Levels 1–5. You now understand what well-optimized kernels look like and what the hardware expects. MLIR is how a compiler automates the decisions you've been making by hand.
- **Comes before:** Level 7 (StableHLO — the exchange format that sits above MLIR's hardware dialects).

## 2026 reality check

- **MLIR is the compiler substrate for everything that matters.** Triton lowers to LLVM through MLIR. IREE compiles ML ops to CPU/Metal/CUDA through MLIR. Tenstorrent's TT-Forge, AMD's ROCm/HIP compiler, Apple's Core ML compiler — all MLIR internally.
- **The practical entry point is narrower than it looks.** You don't need to learn all of MLIR. The three dialects that matter for ML: `linalg` (structured ops), `tensor` (value semantics), and `stablehlo` (portable ML op set). Everything else follows from those three.
- **IREE is the most accessible production MLIR compiler.** It's open-source, well-documented, targets CPU/Metal/CUDA, and has a Python API. It is the fastest path from "I know MLIR dialects" to "I compiled and ran a model on my Mac."
- **The Transform dialect** (stable as of 2025) lets you write optimization schedules as MLIR IR itself — tiling, vectorization, loop unrolling — without writing C++ passes. This is the hands-on path for learning without a full LLVM build.
- **`mlir-in-action` (O'Reilly, 2025)** is the first practical book. It's worth having.

## Topic-by-topic deep dive

| # | Topic | What you build |
|---|-------|---------------|
| 01 | mlir-architecture | Dialects, operations, regions, blocks, values — the core model |
| 02 | linalg-and-tensor-dialects | ML's "sweet spot" dialects; matmul as linalg.matmul |
| 03 | writing-a-pass | Out-of-tree C++ pass; PatternRewriter; mlir-opt pipeline |
| 04 | transform-dialect | Tiling + vectorization without C++ |
| 05 | iree-cpu-and-metal | Compile a model through IREE; run on your Mac |
| 06 | tt-forge-lowering-stack | PyTorch → TTIR → TTNN → Metalium |
| 07 | mlir-in-context | How Triton, Inductor, XLA, AMD all sit on MLIR |

### 01 — `mlir-architecture`

**The core model.** MLIR is a reusable compiler infrastructure. Everything in MLIR is an *operation* (`mlir::Operation`). Operations live in *regions* (collections of *basic blocks*). Operations have *operands* (SSA values) and *results* (more SSA values). The type system is extensible — any dialect can define new types.

**Dialects.** A dialect is a namespace for operations, types, and attributes. Examples: `func.func`, `arith.addf`, `linalg.matmul`, `tensor.extract_slice`, `gpu.launch`. The key insight: you don't lower from one big IR to another — you lower incrementally, *dialect by dialect*. First you lower `linalg.matmul` to `linalg.generic`, then to loop nests, then to `llvm.call`, then to LLVM IR, then to PTX.

**The multi-level part.** The "multi-level" in MLIR is this dialect hierarchy. High-level dialects (`stablehlo`, `linalg`) describe *what* to compute. Low-level dialects (`llvm`, `nvvm`, `amdgpu`) describe *how* to compute it on specific hardware. Passes lower from high to low, one dialect at a time.

**Hands-on.** Install `mlir-native-tools` (`pip install mlir-native-tools` on most platforms). Write a small `.mlir` file with a `func.func` and a `linalg.matmul`. Run `mlir-opt --convert-linalg-to-loops` on it. Read the output — the matmul is now loop nests. Run `mlir-opt --convert-linalg-to-loops --lower-affine --convert-scf-to-cf --convert-cf-to-llvm` — it's now LLVM IR.

### 02 — `linalg-and-tensor-dialects`

**`linalg` dialect.** The ML "sweet spot." Provides named ops (`linalg.matmul`, `linalg.conv_2d`, `linalg.batch_matmul`) and the generic op (`linalg.generic`) that expresses arbitrary structured computations with explicit iteration domains. The critical property: `linalg` ops carry enough information (iteration domain, access patterns) for the compiler to tile, vectorize, and parallelize them automatically.

**`linalg.generic`.** The universal linalg op. A matmul is:
```mlir
linalg.generic {
  indexing_maps = [
    affine_map<(m,n,k) -> (m,k)>,    # A[m,k]
    affine_map<(m,n,k) -> (k,n)>,    # B[k,n]
    affine_map<(m,n,k) -> (m,n)>     # C[m,n]
  ],
  iterator_types = ["parallel", "parallel", "reduction"]
} ins(%A, %B) outs(%C) {
  ^bb0(%a: f32, %b: f32, %c: f32):
    %mul = arith.mulf %a, %b : f32
    %add = arith.addf %c, %mul : f32
    linalg.yield %add : f32
}
```
The `iterator_types` tell the compiler which loops can be parallelized (`"parallel"`) and which must be accumulated (`"reduction"`). This is what enables automatic tiling and vectorization.

**`tensor` dialect.** Value-semantic tensors (immutable, functional). `tensor.extract_slice` extracts a sub-tensor without mutation. This is what enables fusion — the compiler can see the slice operations and fuse the producers/consumers of slices.

### 03 — `writing-a-pass`

**The structure of an MLIR pass.** A pass is a C++ class that:
1. Inherits from `mlir::OperationPass<FuncOp>` (or another op type)
2. Implements `runOnOperation()`
3. Uses `mlir::RewritePatternSet` + `mlir::PatternRewriter` to match-and-replace ops

**Example: a tiling pass.** Match every `linalg.matmul`, replace it with a tiled version using `linalg::tileLinalgOp`:
```cpp
struct TileMatmulPass : public OperationPass<func::FuncOp> {
  void runOnOperation() override {
    auto fn = getOperation();
    RewritePatternSet patterns(fn.getContext());
    patterns.add<TileMatmulPattern>(fn.getContext(), {8, 8, 8}); // tile sizes
    if (failed(applyPatternsAndFoldGreedily(fn, std::move(patterns))))
      signalPassFailure();
  }
};
```

**Build steps.**
1. Clone `llvm-project`. Build with `-DLLVM_ENABLE_PROJECTS=mlir -DCMAKE_BUILD_TYPE=Release`. (Takes 30–60 min. Worth doing once.)
2. Create an out-of-tree dialect project using the `standalone` example as a template.
3. Write a pass that takes a `linalg.matmul` and tiles it to size 8×8×8.
4. Run through `mlir-opt --tile-matmul`. Look at the output — is the tiling correct?
5. Add a second pass that vectorizes the tiled loops using `mlir::VectorizeLinalgOpPass`.
6. Run the pipeline end-to-end: tile → vectorize → lower-to-llvm → translate-to-llvmir. Compile the output with `llc`. Run it.

**Alternative: Jeremy Kun's blog.** If the full LLVM build is too slow on your machine, follow [Jeremy Kun's MLIR pass tutorial](https://www.jeremykun.com/2023/08/10/mlir-writing-our-first-pass/) which works with a smaller prebuilt setup.

### 04 — `transform-dialect`

**What it is.** The Transform dialect lets you express the *transformation schedule* — "tile this matmul with sizes 8×8×8, then vectorize, then unroll" — as MLIR IR, not C++. This means transformation schedules are data, not code — you can parameterize them, specialize them per hardware, and evolve them without recompiling the compiler.

```mlir
// Transform IR: tile a matmul
transform.sequence failures(propagate) {
  ^bb0(%root: !transform.any_op):
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %root
    %tiled, %loops = transform.structured.tile_using_for %matmul [8, 8, 8]
    transform.structured.vectorize %tiled
}
```

**Why it matters for learning.** You can experiment with different tiling strategies without rebuilding the compiler. Change tile sizes, add loop unrolling, try different vectorization strategies — all by editing a text file.

**Build steps.** Install `iree-compiler`. Write a Transform dialect schedule that tiles + vectorizes a `linalg.matmul`. Run it through IREE's CPU backend. Measure the impact of different tile sizes on performance.

**Resources.**
- [MLIR Transform dialect — arxiv 2409.03864](https://arxiv.org/html/2409.03864v2)
- [IREE Transform dialect tutorial](https://iree.dev/community/blog/2024-01-29-iree-mlir-linalg-tutorial/)

### 05 — `iree-cpu-and-metal`

**IREE.** The most accessible production MLIR compiler. It compiles ML programs to CPU (via LLVM), Metal (via SPIRV → MSL), CUDA, Vulkan. For the ML curriculum, it's the practical tool that connects MLIR theory to running code.

**IREE Python API:**
```python
import iree.compiler as ireec
import iree.runtime as iree_rt

# Compile a PyTorch model to Metal (M5 Mac!)
mlir_module = torch.export.export(model, example_inputs)
target_backends = ["metal"]  # runs on your M5 Mac
flatbuffer = ireec.compile_str(mlir_module.mlir, target_backends=target_backends)

# Run
config = iree_rt.Config("local-task")
ctx = iree_rt.SystemContext(config=config)
vm_module = iree_rt.VmModule.from_flatbuffer(ctx.instance, flatbuffer)
result = vm_module.main(*inputs)
```

**Why this is interesting on your M5 Mac.** Metal is Apple's GPU compute API. IREE compiles MLIR → SPIRV → MSL (Metal Shading Language) → Metal pipeline. Your M5 Mac's GPU runs the compiled code. This is a real hardware target, not simulation.

**Build steps.**
1. `pip install iree-compiler iree-runtime`.
2. Write a small model (matmul + ReLU). Compile to CPU and Metal via IREE.
3. Compare to PyTorch MPS (Metal Performance Shaders) — both use Metal, but IREE goes through the MLIR lowering chain while MPS uses Apple's hand-optimized library.
4. Look at the generated MSL (Metal Shading Language) — `iree-compile --mlir-print-ir-after-all`. This is the Metal shader your M5 runs.

### 06 — `tt-forge-lowering-stack`

**Why Tenstorrent's stack is the best real-world MLIR case study.** TT-Forge is fully open-source (Apache 2.0), production-quality, and its lowering chain is clear and well-documented. It's the canonical example of a company building a full ML compiler on MLIR for custom hardware.

**The chain:**
```
PyTorch / JAX
    ↓ (torch.export or PJRT)
StableHLO
    ↓ (tt-forge-fe: StableHLO → TTIR)
TTIR (Tenstorrent IR — high-level)
    ↓ (compiler: TTIR → TTNN)
TTNN (Tenstorrent Neural Network — ops library calls)
    ↓ (runtime dispatch)
Metalium kernels (Tenstorrent's Triton-like kernel library)
    ↓
Tensix cores (the actual Tenstorrent RISC-V + matrix engine)
```

**What to study.** Clone `tt-mlir`. Read `include/ttmlir/Dialect/TTIR/`. Understand what ops are in TTIR that aren't in standard MLIR (they model Tenstorrent's unique data movement model — multi-cast, unicast, scatter). Read the `lib/Conversion/TTIRToTTNN/` conversion passes — this is the lowering from abstract ops to concrete TTNN API calls.

**Connection to your work.** The passes in TT-Forge are the same structural kind as the pass you wrote in Topic 03 — `PatternRewriter`, match op, replace with lower-level equivalent. The difference is the ops being matched are hardware-specific.

### 07 — `mlir-in-context`

**How everything sits on MLIR:**

| System | MLIR role |
|---|---|
| Triton | Lowers `triton` dialect → `triton_gpu` → `llvm` → PTX via MLIR |
| Inductor | Triton codegen uses MLIR internally; CUTLASS backend generates MLIR |
| JAX/XLA | Lowers through `mhlo`/`stablehlo` → `linalg` → `llvm` via OpenXLA |
| TT-Forge | `stablehlo` → `ttir` → `ttnn` (custom dialects) |
| IREE | `stablehlo`/`linalg` → `spirv`/`llvm`/`nvvm`/`amdgpu` |
| ROCm (AMD) | `rocdl` dialect in LLVM/MLIR; `amdgpu` dialect for GPU ops |
| Core ML (Apple) | Internal MLIR for model compilation; IREE for Metal |
| Mojo (Modular) | Built on MLIR; `KGEN` dialect |

**The insight.** Every system you've learned in `systems-for-ml` and this track has MLIR underneath. When you debug a Triton kernel that generates wrong code, the bug might be in the `triton_gpu` → `llvm` lowering pass. When you see a vLLM kernel fusion that's suboptimal, it's the Inductor-MLIR pipeline that made that decision. MLIR is the common substrate.

## Project this week

```
compiler-and-kernels/
└── mlir/
    ├── standalone_pass/           # out-of-tree MLIR pass (CMake project)
    │   ├── CMakeLists.txt
    │   ├── TileMatmulPass.cpp
    │   └── tile_matmul.mlir       # test input
    ├── transform_schedule.mlir    # Transform dialect tiling schedule
    ├── iree_demo.py               # PyTorch → IREE → CPU + Metal
    └── reports/
        └── level6-mlir.md        # lowering diagram + pass walkthrough
```

## Definition of done

- [ ] You built `mlir-opt` and ran a lowering pipeline on a `linalg.matmul`.
- [ ] You wrote a C++ MLIR pass that tiles a matmul. It runs and produces correct output.
- [ ] You deployed a model through IREE to CPU and Metal. Both produce correct outputs.
- [ ] You can trace the TT-Forge lowering chain from PyTorch to Metalium on a whiteboard.
- [ ] `reports/level6-mlir.md` includes a lowering diagram and the annotated pass code.

## Resources

- **MLIR docs** — [mlir.llvm.org](https://mlir.llvm.org/). "Why MLIR?" and "Getting Started."
- **Jeremy Kun — Writing Our First Pass** — [jeremykun.com/2023/08/10/mlir-writing-our-first-pass](https://www.jeremykun.com/2023/08/10/mlir-writing-our-first-pass/).
- **MLIR in Action (O'Reilly 2025)** — the practical book. Worth buying.
- **IREE getting started** — [iree.dev/guides/ml-frameworks/pytorch](https://iree.dev/guides/ml-frameworks/pytorch/).
- **IREE/MLIR/Linalg tutorial** — [iree.dev/community/blog](https://iree.dev/community/blog/2024-01-29-iree-mlir-linalg-tutorial/).
- **MLIR Transform dialect** — [arxiv.org/abs/2409.03864](https://arxiv.org/html/2409.03864v2).
- **TT-Forge GitHub** — [github.com/tenstorrent/tt-forge](https://github.com/tenstorrent/tt-forge).
- **tt-mlir GitHub** — [github.com/tenstorrent/tt-mlir](https://github.com/tenstorrent/tt-mlir).
- **Standalone MLIR dialect example** — `llvm-project/mlir/examples/standalone/`.

## What you'll be able to do after this week

> Write an MLIR optimization pass in C++ using PatternRewriter. Follow a model's lowering chain through linalg → loops → LLVM IR. Deploy a PyTorch model through IREE to both CPU and Metal on your Mac. Read and navigate the TT-Forge codebase well enough to understand what each dialect is doing and why.
