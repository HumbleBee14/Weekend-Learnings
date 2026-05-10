# 14 — CPU SIMD and SME2

## The 2026 framing

Pure SIMD-only LLM inference is a 2023 conversation. In 2026 the story is matrix extensions: dedicated CPU instructions that do small dense matmuls in one shot, the same shape transformer linear layers want.

The four families that matter:

| Family | Where | Op shape | Notes |
|--------|-------|----------|-------|
| Intel AMX | Sapphire Rapids, Granite Rapids, Xeon 6 | tile×tile bf16/int8 GEMM | The big x86 win. ~2x on Llama 3.2 3B int8 vs AVX-512. |
| Arm SME / SME2 | Apple M4+, recent Android SoCs | scalable streaming matmul | llama.cpp added SME2 kernels late 2025. |
| AVX-512 + VNNI + BF16 | AMD Zen4/5, SPR+ | wide vector dot-products | Baseline x86. Good but matrix ext beats it. |
| NEON | Every ARM CPU | 128-bit SIMD | Universal fallback in llama.cpp. |

Apple's older "AMX" coprocessor (M1–M3, undocumented, exposed via Accelerate) is effectively superseded. The 2026 Apple CPU matmul story is **SME** (M4+), and the GPU matmul story is **M5 Neural Accelerators** (Topic 04).

## What SME2 actually is

SVE2's matrix sibling. SME ("Scalable Matrix Extension") adds:

- A **streaming SVE mode** (SSVE) — wider effective vectors (typically 256–2048 bits implementation-defined) but a restricted ISA.
- **ZA storage** — a 2D architectural array (square tile, side = SVL bytes). Holds intermediate accumulators across instructions.
- **Outer-product instructions** — `FMOPA`, `BFMOPA`, `SMOPA`, `UMOPA`. One instruction does an outer product of two vectors into ZA. That is exactly what a small GEMM tile is.

SME2 (the 2024 revision) adds multi-vector instructions and lookup-table ops. ARM's pitch is "one instruction = a (k×n) outer product accumulated into ZA," which collapses an inner GEMM loop to a few instructions.

```
        vec_a  (k lanes)             vec_b  (n lanes)
          |                              |
          +---- FMOPA  (outer product) --+
                       |
                   ZA tile  (k×n accumulator, persistent across calls)
                       |
                   read out into vector regs, store to memory
```

Reference: [ARM SME programmer's guide](https://developer.arm.com/documentation/109246/latest/) and the [ARM SME2 intrinsics reference](https://developer.arm.com/architectures/instruction-sets/intrinsics/#f:@navigationhierarchiessimdisa=[sme]).

## Why this matters for LLMs

A 4-bit quantized matmul against a streaming activation is bandwidth-bound on the weight side, but the per-token compute on the activations is not free either. SME2's outer-product form keeps the accumulator hot in ZA across an entire row of output activations, eliminating a lot of memory traffic for partial sums.

llama.cpp's SME kernels (in `ggml/src/ggml-cpu/arch/arm/`) target the dequant-and-matmul fused path for `Q4_0`, `Q4_K`, `Q8_0`. The numbers reported by the llama.cpp authors and corroborated by the Cortensor write-up: roughly 1.3–1.7x on M4 vs the NEON path for prompt-processing-dominated workloads. Decode is more bandwidth-bound and gains less.

References:
- [llama.cpp PR adding SME](https://github.com/ggml-org/llama.cpp/pull/10752)
- [Cortensor — CPU instruction sets for LLM inference](https://docs.cortensor.network/technical-architecture/ai-inference/cpu-instruction-sets-for-llm-inference-avx-amx-sme-vs-gpus)

## When CPU beats GPU on a Mac

Counterintuitive in 2026 but real:

1. **Single request, decode-only, very small model (≤3B at 4-bit).** GPU launch overhead per token can exceed actual compute for tiny models. CPU wins on TTFT for batch=1.
2. **Sustained low-power.** GPU pulls 30–60W on M-series under load. P-cores at moderate clocks pull a fraction of that. For an always-on local agent doing 1 tok every few seconds, CPU keeps the fan off.
3. **Cold-start sensitivity.** First-token latency on GPU includes Metal pipeline state warming and shader cache hits. CPU has no warmup curve.
4. **Co-tenant workloads.** When the GPU is busy (a Mac doing screen rendering, a video call, a separate model), CPU inference avoids contention.

When GPU dominates: any concurrency, prompt processing of long contexts (compute-bound), models above ~7B, anywhere throughput matters.

## The Accelerate framework path

macOS ships [Accelerate.framework](https://developer.apple.com/documentation/accelerate), which includes BLAS, LAPACK, vDSP, BNNS, and (since macOS 15) the [BNNSGraph](https://developer.apple.com/documentation/accelerate/bnnsgraph) compiled-graph API. Accelerate auto-dispatches to:

- AMX coprocessor on M1–M3 (still works, undocumented).
- SME on M4+.
- NEON otherwise.

You almost never call SME intrinsics directly from app code. You either:

1. Link Accelerate (`-framework Accelerate`) and call `cblas_sgemm` / `cblas_hgemm` / BNNS ops, and the runtime picks the right backend.
2. Use llama.cpp built with `-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=Apple` for prompt processing, and let its `ggml-cpu` SME kernels run for decode.
3. Use MLX, which dispatches CPU ops through its own kernels (and Accelerate where it helps) when stream is `mx.cpu`.

## llama.cpp CPU backend in 2026

```
ggml/src/ggml-cpu/
├── arch/
│   ├── arm/      <-- NEON, SVE, SME, SME2 kernels
│   ├── x86/      <-- AVX2, AVX-512, AMX kernels
│   └── ...
├── ggml-cpu.cpp  <-- runtime dispatch
```

Build flags that matter on Apple Silicon:

- `-DGGML_NATIVE=ON` — autodetect host capabilities. Default-on.
- `-DGGML_METAL=OFF` — exclude GPU. Forces CPU-only build. Useful for measuring CPU-only.
- `-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=Apple` — prompt processing through Accelerate.
- Threads: `-t <N>`. **N = number of P-cores.** Adding E-cores hurts throughput on Apple Silicon — different perf curves, scheduler thrashes between them.

```
M3 Max, 7B Q4_K_M, decode (single request), batch=1:
  -ngl 99       (full GPU)        : ~95 tok/s
  -ngl 0 -t 12  (CPU only, P-cores): ~22 tok/s
  -ngl 0 -t 16  (CPU + E-cores)    : ~18 tok/s   <-- regression
```

Numbers ballpark, your hardware will differ. The shape of the result is robust.

## NEON, in case you want to read the kernels

NEON is 128-bit SIMD: 16x int8, 8x int16/fp16, 4x fp32, 2x fp64 per register. The llama.cpp 4-bit dequant-and-dot kernels use:

- `vld1q_u8` to load 16 packed nibbles.
- `vshrq_n_u8` / `vandq_u8` to split high/low nibbles.
- `vsubq_s8` to subtract the zero-point (8 for symmetric Q4_0).
- `vdotq_s32` (NEON dot-product, ARMv8.2-A+) for the int8 inner product.
- `vfmaq_f32` to scale by the block scale and accumulate fp32.

The SME path replaces the `vdotq` chain with `SMOPA` outer products into ZA, which is wider and accumulates in-register across the whole tile instead of streaming through fp32 vectors.

## CPU vs GPU on Apple Silicon — the bandwidth view

Both the P-core cluster and the GPU pull from the same unified DRAM. The peak bandwidth quoted for an M3 Max (~400 GB/s) is shared. The P-cores cannot saturate it alone — they top out around 200–250 GB/s in practice. The GPU can. So:

- For a memory-bound kernel (LLM decode), the GPU has more bandwidth to work with.
- For a compute-bound kernel that fits in cache (short prompts, small models), the CPU's matrix extensions plus L2 locality can be competitive.

```
   CPU cluster                          GPU
   (P-cores, ~200-250 GB/s practical)   (~400 GB/s practical)
       \                                 /
        \                               /
         \---- shared LPDDR5x DRAM ----/
                  (~400 GB/s peak)
```

## Common pitfalls

1. **Adding E-cores to the thread count.** They run at lower clocks and the scheduler migrates threads, hurting cache locality. P-cores only.
2. **Comparing CPU and GPU on full models.** GPU wins by a wide margin on anything ≥7B. The interesting CPU regimes are small-model, low-power, batch-1.
3. **Assuming SME is automatic on M4+ everywhere.** llama.cpp picks it up if built recently. Older binaries do not. `llama-bench --help | grep -i sme` and check the build log.
4. **Targeting Apple's old AMX coprocessor.** Effectively superseded; do not write new code against it. SME on M4+ is the path.
5. **Ignoring Accelerate for prompt processing.** A linked Accelerate BLAS will dominate hand-written kernels for the matmul-heavy prefill phase.

## What to walk away with

- The four CPU families (AMX, SME/SME2, AVX-512, NEON) and which one runs where.
- A built llama.cpp that can flip between CPU and GPU (`-ngl 0` vs `-ngl 99`) and a reproducible tok/s comparison.
- An honest answer to "should I use the CPU?" — usually no, but the cases where yes are real and worth knowing.

## References

- ARM SME programmer's guide: https://developer.arm.com/documentation/109246/latest/
- ARM SME2 intrinsics: https://developer.arm.com/architectures/instruction-sets/intrinsics/#f:@navigationhierarchiessimdisa=[sme]
- llama.cpp SME PR: https://github.com/ggml-org/llama.cpp/pull/10752
- Apple Accelerate framework: https://developer.apple.com/documentation/accelerate
- BNNSGraph API: https://developer.apple.com/documentation/accelerate/bnnsgraph
- Cortensor — CPU instruction sets for LLM inference: https://docs.cortensor.network/technical-architecture/ai-inference/cpu-instruction-sets-for-llm-inference-avx-amx-sme-vs-gpus
- Intel AMX overview: https://www.intel.com/content/www/us/en/products/docs/accelerator-engines/advanced-matrix-extensions/overview.html
