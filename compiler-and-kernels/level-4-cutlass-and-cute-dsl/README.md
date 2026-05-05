# Level 4 — CuTe DSL and CUTLASS 4.x

> Outer reference: [`compiler-and-kernels/README.md`](../README.md) · Project: BF16 persistent GEMM in CuTe-DSL on SM90; NVFP4 variant on SM100

## Week goal

FA4 is written in CuTe-DSL. The NVFP4 GEMMs in TRT-LLM are written in CuTe-DSL. TorchInductor's fourth autotuning backend is CuTe-DSL. To understand the kernel layer that production LLM inference actually runs on Blackwell, you need this. By Friday you should be able to:

- Understand CuTe's layout algebra — how tensors are described as `(shape, stride)` compositions
- Write a dense BF16 GEMM in CuTe-DSL for Hopper (SM90) using persistent grid + TMA + WGMMA
- Understand the NVFP4 GEMM variant — how block scaling integrates into the GEMM pipeline
- Know when to reach for CuTe-DSL vs Triton vs `torch.compile`

## Where this fits

- **Comes after:** Level 3 (FA4 is the motivating example — it's the first major attention kernel written in CuTe-DSL; you should understand *why* the FA team chose it before learning the tool).
- **Comes before:** Level 5 (kernel fusion — once you can write GEMMs in CuTe-DSL, you can fuse epilogues into them).

## 2026 reality check

- **CuTe-DSL (Python) is stable as of CUTLASS 4.x (2025–2026).** It generates PTX/SASS on-the-fly via a Python JIT. No C++ toolchain required. FA4 is the canonical production proof that CuTe-DSL is production-grade.
- **TorchInductor added CuTe-DSL as its fourth backend.** For NVFP4 GEMMs on Blackwell, it's ~5% behind CUTLASS C++ — acceptable for most workloads and much faster to iterate on.
- **vLLM uses CUTLASS (C++ templates) for FP8/FP4 GEMM today.** The migration toward CuTe-DSL is in progress. Understanding both the C++ template heritage and the Python-DSL present is useful context.
- **CuTe is not Triton.** Triton operates at the tile level (you specify tile sizes, the compiler handles threads). CuTe operates at the hardware level — you explicitly specify TMA descriptors, WGMMA register layouts, and tensor memory instructions. It's lower-level, more control, more hardware knowledge required.

## Topic-by-topic deep dive

| # | Topic | What you build |
|---|-------|---------------|
| 01 | cute-layout-algebra | `(shape, stride)` compositions; tensor views; layout swizzles |
| 02 | tma-and-wgmma | Async data movement; warp group matrix multiply |
| 03 | persistent-gemm-sm90 | BF16 dense GEMM on Hopper with TMA + WGMMA |
| 04 | nvfp4-gemm-sm100 | Block-scaled 4-bit GEMM on Blackwell |
| 05 | gemm-epilogues | Fusing bias, activation, scaling into the GEMM output |
| 06 | inductor-cutedsl-backend | How Inductor uses CuTe-DSL; when it fires |
| 07 | cutlass-cpp-heritage | C++ templates for CUTLASS; reading vLLM's GEMM code |

### 01 — `cute-layout-algebra`

**The core abstraction.** In CuTe, every tensor is described by a `Layout`: a pair `(shape, stride)`. `shape` is the size in each dimension; `stride` is how many elements to skip per step in that dimension.

```python
# A 4×8 row-major matrix:
# shape = (4, 8), stride = (8, 1)
# element [i,j] is at offset i*8 + j*1

# A 4×8 column-major matrix:
# shape = (4, 8), stride = (1, 4)
# element [i,j] is at offset i*1 + j*4
```

**Layout composition.** Layouts can be nested and composed. A tiled layout `(m/BLOCK_M, BLOCK_M):(BLOCK_M, 1)` represents a block-row-major arrangement. CuTe's `make_layout`, `composition`, and `coalesce` operations let you build complex memory access patterns from simple building blocks. This is the algebra that makes TMA descriptors, swizzled shared memory layouts, and register file layouts all describable in one unified language.

**Why it matters.** Every performance decision in a GEMM kernel is a layout decision: how tiles are laid out in shared memory to avoid bank conflicts, how registers are arranged for WGMMA inputs, how the output accumulator maps back to global memory. CuTe makes these decisions explicit and composable instead of buried in CUDA C++ templates.

**Swizzle.** Shared memory bank conflicts occur when multiple threads in a warp access addresses that map to the same bank. Swizzle patterns rearrange shared memory layout to avoid this. CuTe's `make_swizzle_layout` generates the conflict-free arrangement for a given tile size.

**Build steps.** Work through the CuTe layout algebra documentation. Write a Python script that constructs layouts for: (1) a row-major 64×64 BF16 matrix, (2) the same matrix tiled into 16×16 blocks, (3) a swizzled shared memory layout for a 64×64 tile. Visualize the offset patterns — `layout(i, j)` → offset.

### 02 — `tma-and-wgmma`

**TMA (Tensor Memory Accelerator).** TMA is a Hopper hardware unit that copies multidimensional tensor regions from global memory to shared memory (or vice versa) asynchronously, without warp participation. You create a `TMA descriptor` (a 128-byte structure that encodes the tensor's dimensions, strides, element type, and swizzle) and then issue async copy instructions.

In CuTe-DSL:
```python
tma_desc = make_tensor_descriptor(
    gmem_ptr, shape=(M, K), strides=(K, 1), 
    box_shape=(BLOCK_M, BLOCK_K), swizzle=SWIZZLE_128B
)
# Async copy: load tile into smem
copy_async(tma_desc, smem_tile, coords=(m_tile, k_tile))
```

**WGMMA (Warp Group Matrix Multiply-Accumulate).** WGMMA is the Hopper tensor core instruction that operates on a full warp group (128 threads). Input matrices reside in shared memory or registers; the accumulator lives in registers. WGMMA has higher throughput than the older HMMA instruction (FA2 used HMMA; FA3/FA4 use WGMMA).

In CuTe-DSL:
```python
mma = TiledMMA(SM90_64x128x16_F32BF16BF16_SS)  # WGMMA variant
# Execute: C += A * B
gemm(mma, tiled_C, tiled_A, tiled_B)
```

**The connection.** TMA loads K/V tiles into shared memory asynchronously (producer warps). WGMMA computes the matmul from shared memory (consumer warps). These two together — TMA + WGMMA with warp specialization — are what Level 1 was building toward. You've now seen them from the Triton side and the CuTe-DSL side.

### 03 — `persistent-gemm-sm90`

**The canonical CuTe-DSL exercise.** Write a BF16 GEMM on Hopper that:
1. Uses persistent grid (one program per SM, each SM processes multiple output tiles)
2. TMA for async A and B tile loads
3. WGMMA for compute
4. Double-buffered shared memory (ping-pong between two SMEM buffers while computing)
5. Warp specialization (producer warps drive TMA; consumer warps run WGMMA)

**Colfax Research tutorials** are the canonical learning resource for this. [Their Blackwell GEMM tutorial](https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/) is the clearest step-by-step walkthrough of a real CuTe-DSL GEMM. Work through it, but implement for SM90 (Hopper) first — it's more accessible than SM100.

**Benchmark target.** On A100/H100:
- cuBLAS BF16: ~280–300 TFLOPS/s on large square GEMMs
- Your CuTe-DSL GEMM: aim for 80–90% of cuBLAS

**Build steps.**
1. Clone CUTLASS. Navigate to `examples/python/CuTeDSL/`.
2. Run the dense GEMM example to confirm your environment.
3. Implement your own: start from the example, modify the tile sizes, add profiling (`triton.proton` or Nsight).
4. Plot achieved TFLOPS vs matrix size (64×64 → 8192×8192). Understand where and why you fall below cuBLAS.

### 04 — `nvfp4-gemm-sm100`

**Why NVFP4 matters.** Blackwell B200 delivers ~18,000 sparse FP4 TFLOPS — 2× over FP8. For a weight-only FP4 model (weights in FP4, activations in BF16), the GEMM has asymmetric operands: one FP4 tensor (weights) and one BF16 tensor (activations). This is called `NVFP4A16` (FP4 weights, FP16/BF16 activations).

**The block scaling challenge.** NVFP4 uses block scaling: every 16 elements share one E8M0 scale factor. Before the GEMM can run, the FP4 weights need to be dequantized (or the scale needs to be folded into the GEMM epilogue). The dequantization path adds overhead; the epilogue path requires a custom GEMM template.

**The CuTe-DSL solution.** CUTLASS provides a `SM100_F32F4BF16BF16_SS` WGMMA instruction that handles FP4×BF16 natively with hardware dequantization. CuTe-DSL exposes this via `TiledMMA(SM100_F32F4BF16BF16_SS)`. The block scales are applied as part of the WGMMA instruction with no extra overhead.

**Build steps (requires Blackwell or SM100 emulation).**
1. Read the [Colfax sub-byte GEMM tutorial](https://research.colfax-intl.com/cutlass-tutorial-sub-byte-gemm-on-nvidia-blackwell-gpus/).
2. Run the NVFP4 example from `examples/python/CuTeDSL/`.
3. If no Blackwell access: run in simulation mode (`CUTLASS_EMULATION=1`) and read the generated code, even if you can't measure real performance.
4. Understand: where are the block scales stored? How do they align to the weight tiles? What changes in the loop structure vs the BF16 GEMM?

### 05 — `gemm-epilogues`

**What epilogues do.** After the GEMM accumulator is computed (`C += A * B`), the epilogue applies: bias addition, activation function (GeLU, SiGLU), scaling (FP8 requantization output scale), ReLU. Without epilogue fusion, each of these would be a separate kernel — more HBM round-trips.

**In CuTe-DSL.** Epilogue visitors are composable lambdas that transform the accumulator tile before writing to global memory. The bias tensor is loaded from a separate TMA descriptor and added element-wise. The activation is applied as a pointwise op in the epilogue body.

```python
# Fused linear + bias + GELU epilogue
epilogue = EpilogueVisitor([
    AddBias(bias_tma_desc),
    ApplyGELU(),
    ScaleOutput(output_scale)
])
```

**When this matters.** For LLM serving, every linear layer is `xW + b` followed by an activation. Without epilogue fusion, `W@x`, `+b`, and `activation` are three kernels. With epilogue fusion, it's one GEMM with an epilogue — 3× fewer HBM writes.

### 06 — `inductor-cutedsl-backend`

**How Inductor uses CuTe-DSL.** PyTorch 2.6 added CuTe-DSL as Inductor's fourth GEMM backend (alongside Triton, CUTLASS C++, cuBLAS). For NVFP4 GEMMs on Blackwell, Inductor's autotuner tries all four backends and picks the fastest. The CuTe-DSL backend is currently ~5% behind CUTLASS C++ on NVFP4 — acceptable for most use cases, and much faster to iterate on.

**How to see it fire.** Run `TORCH_COMPILE_DEBUG=1` on a model with NVFP4 weights on a Blackwell GPU. Look for kernels named `cutlass_gemm_*` or `cute_dsl_*` in the output code.

**When Triton wins vs CuTe-DSL.** Triton wins for: reduction-heavy ops (softmax, RMSNorm), custom elementwise ops, ops where the tile shape doesn't match WGMMA's fixed tile. CuTe-DSL wins for: GEMM-shaped ops where you need WGMMA, FP4 with block scaling, custom memory layouts that Triton's layout model can't express.

### 07 — `cutlass-cpp-heritage`

**Why you still need to read CUTLASS C++.** vLLM's FP8 GEMM kernels are CUTLASS C++ templates (as of 2026). TRT-LLM's MHA kernel is CUTLASS C++. FlashInfer uses CUTLASS C++ for its paged attention. Even as CuTe-DSL grows, the C++ layer is where most production code still lives.

**Reading CUTLASS C++.** The template hierarchy is deep but regular. A CUTLASS GEMM is parameterized by: element type, layout, tile shape (thread block tile → warp tile → instruction tile), epilogue visitor, and swizzle. The outermost template (`cutlass::gemm::device::Gemm<...>`) has 15+ template parameters — but once you know CuTe layouts, you can read what each one does.

**Build steps.** Open [`vllm/csrc/cutlass_extensions/`](https://github.com/vllm-project/vllm/tree/main/csrc/cutlass_extensions). Find the FP8 GEMM kernel. Identify: the WGMMA instruction being used, the tile shapes, the epilogue. You should be able to connect every template parameter to a concept from this week.

## Project this week

```
compiler-and-kernels/
└── gemm/
    ├── cute_bf16_gemm.py         # BF16 persistent GEMM on SM90
    ├── cute_nvfp4_gemm.py        # NVFP4 GEMM on SM100 (or simulation)
    ├── epilogue_fusion.py        # Fused linear + bias + GELU
    └── reports/
        └── level4-cute-dsl.md   # benchmark table + layout diagrams
```

**Benchmark table:**

| Kernel | Shape | Precision | TFLOPS/s | % of cuBLAS |
|---|---|---|---|---|
| cuBLAS | 4096×4096 | BF16 | | 100% |
| Your CuTe-DSL | 4096×4096 | BF16 | | |
| cuBLAS | 4096×4096 | FP8 | | 100% |
| NVFP4 (SM100 / simulation) | 4096×4096 | FP4 | | |

## Definition of done

- [ ] You can draw the CuTe layout algebra: `(shape, stride)` composition, swizzle, TMA descriptor structure.
- [ ] You have a working BF16 persistent GEMM in CuTe-DSL with benchmark numbers vs cuBLAS.
- [ ] You understand what NVFP4 block scaling adds to the GEMM pipeline.
- [ ] You've read vLLM's CUTLASS C++ GEMM code and can annotate its template parameters.

## Resources

- **CUTLASS GitHub** — [github.com/NVIDIA/cutlass](https://github.com/NVIDIA/cutlass). Start with `examples/python/CuTeDSL/`.
- **CUTLASS docs** — [docs.nvidia.com/cutlass](https://docs.nvidia.com/cutlass/latest/overview.html).
- **Colfax: Blackwell GEMM tutorial** — [research.colfax-intl.com](https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/).
- **Colfax: sub-byte GEMM (NVFP4)** — [research.colfax-intl.com](https://research.colfax-intl.com/cutlass-tutorial-sub-byte-gemm-on-nvidia-blackwell-gpus/).
- **Ian Barber: CuTe-DSL deep-dive** — [ianbarber.blog/2025/07/04/cute-dsl](https://ianbarber.blog/2025/07/04/cute-dsl/).
- **TorchInductor CuTe-DSL backend blog** — [pytorch.org/blog/gemms-torchinductor-cutedsl-backend](https://pytorch.org/blog/gemms-torchinductor-cutedsl-backend/).

## What you'll be able to do after this week

> Read and write CuTe-DSL GEMM kernels using TMA, WGMMA, persistent grids, and fused epilogues. Understand the NVFP4 block-scaling GEMM for Blackwell. Read vLLM's and TRT-LLM's CUTLASS C++ kernels. Know precisely when to use CuTe-DSL vs Triton vs torch.compile for GEMM-shaped workloads.
