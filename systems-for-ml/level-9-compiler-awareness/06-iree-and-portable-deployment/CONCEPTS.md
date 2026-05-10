# 06 — IREE and Portable Deployment

IREE (Intermediate Representation Execution Environment) is the answer to one specific question: *can a single compiled artifact run on Vulkan, Metal, CUDA, ROCm, and CPU without re-tuning per backend?* The answer in 2026 is "yes, for a meaningful range of models" — and that's enough to make IREE the de facto open-source portable-deployment stack.

Project home: https://iree.dev/. Code: https://github.com/iree-org/iree.

## The shape of the problem

Deployment fragmentation, viewed from one team:

```
Same model, four targets, four toolchains:

  PyTorch -> torch.compile -> CUDA       (NVIDIA datacentre)
  PyTorch -> Core ML        -> Metal     (Apple devices)
  PyTorch -> NNAPI / TFLite -> Vulkan    (Android)
  PyTorch -> ONNX -> ONNXRuntime -> CPU  (servers without GPUs)
```

Four conversion paths, four sets of bugs, four sets of perf cliffs. IREE collapses this:

```
  PyTorch / JAX / TF / ONNX
        │   import to StableHLO (or Linalg)
        ▼
  IREE compiler  (MLIR pipeline, Linalg + Flow + HAL dialects)
        │
        ▼
  IREE artifact (.vmfb)  ───┐
                            │  IREE runtime
                            ▼
              ┌──────────┬──────────┬────────┬──────┐
              ▼          ▼          ▼        ▼      ▼
            CUDA       Vulkan      Metal   ROCm   CPU
                                                  (LLVM)
```

One compile, multiple runtime backends. The artifact (`.vmfb`) bundles the compiled kernels for the target(s) you asked for at compile time.

## The dialect chain

This is where MLIR's "multi-level" nature shows up clearly. A model passes through several dialects on its way down:

```
  StableHLO       (frontend-portable IR, vendor-neutral ML ops)
       │  (lower)
       ▼
  Linalg          (generic tensor ops, fusable, optimisable)
       │  (form dispatch regions)
       ▼
  Flow            (IREE's own dialect: dispatch regions + dataflow)
       │  (assign to streams)
       ▼
  Stream          (asynchronous execution, command buffers)
       │  (target-specific lowering)
       ▼
  HAL             (Hardware Abstraction Layer: device-agnostic API)
       │
       ▼
  Backend codegen
   ├── LLVM        → CPU
   ├── SPIR-V      → Vulkan / Metal
   ├── NVVM/PTX    → CUDA
   └── ROCDL       → ROCm
```

The `Flow` dialect is IREE's distinctive contribution. It expresses "this region of the program is a unit of work that gets dispatched to a device." That's a different concept from "this is a fused kernel" — Flow is about *scheduling* across devices and queues, not about kernel fusion. Flow regions get further lowered to Stream (async) and HAL (device-portable command submission).

References:
- IREE architecture overview — https://iree.dev/developers/general/developer-overview/
- IREE dialect docs — https://iree.dev/reference/mlir-dialects/
- MLIR Linalg dialect — https://mlir.llvm.org/docs/Dialects/Linalg/

## What the runtime actually does

The IREE runtime is small (C, embeddable). It loads a `.vmfb`, sets up a HAL device for the requested backend, and dispatches command buffers. Compared to ONNXRuntime or TFLite, it's deliberately *not* a kernel registry — there is no big switch statement matching ops to handwritten implementations. Every kernel that runs came out of the compiler.

This means:
- New op support is a compiler change, not a runtime change.
- The runtime binary stays small; useful for embedded targets.
- Perf is bounded by what the compiler can codegen, not by what kernels someone has hand-written.

Trade-off: for ops where a hand-tuned kernel already exists (FlashAttention on CUDA, for example), IREE's codegen may be slower. IREE supports `ukernel` overrides for exactly this case — drop in a hand-tuned implementation for a specific shape/target.

## Importing PyTorch

The current path is `iree-turbine`: https://github.com/iree-org/iree-turbine. It exposes `aot.export` that takes a PyTorch `nn.Module`, traces through the same `torch.export` mechanism vLLM and ExecuTorch use, and emits StableHLO. The compiler then takes over.

```python
import torch
import iree.turbine.aot as aot

class M(torch.nn.Module):
    def forward(self, x):
        return torch.nn.functional.silu(x) @ x.transpose(-1, -2)

m = M()
example = torch.randn(4, 8)
exported = aot.export(m, example)
exported.save_mlir("model.mlir")          # the StableHLO
exported.compile(save_to="model.vmfb")    # ready-to-run IREE artifact
```

The `model.mlir` is human-readable. Open it. You'll see `stablehlo.dot_general`, `stablehlo.transpose`, `stablehlo.logistic` — the StableHLO equivalent of the PyTorch ops, vendor-neutral.

## Where IREE is and isn't a fit (2026)

Fits well:
- Edge / on-device deployment with mixed silicon (Android Vulkan, iOS Metal, desktop CPU/GPU). This was always IREE's strongest story.
- Embedded inference where binary size matters and you can accept "the compiler picks the kernel."
- Anyone who wants to *avoid* the per-vendor toolchain matrix (PyTorch + Core ML + ONNX + ...).
- Production research on novel hardware — vendors building new accelerators reuse IREE's MLIR pipeline rather than reinventing it.

Less of a fit:
- Datacentre LLM inference. vLLM / SGLang / TRT-LLM still win on NVIDIA because they're integrated with FlashAttention/FlashInfer, paged KV cache, continuous batching. IREE has none of that orchestration.
- Workloads dominated by attention kernels where IREE's codegen lags hand-tuned CUTLASS/Triton.

## How this connects to the rest of the level

```
  Topic 03 (XLA vs Inductor):  StableHLO is the input to IREE.
  Topic 04 (MLIR/LLVM):        IREE is an MLIR-native compiler;
                                Linalg, Flow, Stream, HAL are MLIR dialects.
  Topic 05 (accelerator landscape): IREE is the open-source counterpart
                                     to Modular MAX in the "one IR, many
                                     backends" story.
  Topic 07 (CUTLASS/CuTe):     IREE can call CUTLASS-derived kernels via
                                ukernel overrides on CUDA targets.
```

## What's actually changing in 2026

- **`iree-turbine` is the supported PyTorch path** — replaced the older `SHARK-Turbine`. Stability and op coverage improved through 2025.
- **AMD has invested in IREE** as part of its open ROCm story; ROCm + IREE on MI300X is a real deployment path for teams that want vendor neutrality.
- **WebGPU backend matured** — IREE-compiled artifacts running in-browser is now usable for small models (sub-1B), which is interesting for client-side inference.
- **Cross-compilation for Apple Silicon** via Metal works; LLM inference at small scale on iOS uses IREE in some shipping apps.
- **StableHLO has stabilized as the import contract** — the IREE team has stopped accepting new frontend dialects directly; frontends are expected to emit StableHLO. This is the same convergence story XLA pushed.

## Reading list (in order)

1. https://iree.dev/ — landing page, 10 minutes.
2. https://iree.dev/developers/general/developer-overview/ — architecture, 30 minutes.
3. https://github.com/iree-org/iree-turbine — PyTorch import, README only.
4. https://openxla.org/stablehlo/spec — StableHLO spec, skim. This is what gets imported.
5. https://iree.dev/guides/ml-frameworks/pytorch/ — PyTorch end-to-end walkthrough.

That's the awareness pass. Going deeper means reading the dispatch-region formation passes, which is real compiler engineering work.
