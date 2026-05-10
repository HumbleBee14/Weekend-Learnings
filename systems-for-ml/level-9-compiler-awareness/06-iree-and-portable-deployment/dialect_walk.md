# Dialect walk — one matmul, top to bottom in IREE

A `4x16 @ 16x32` matmul as it descends the IREE pipeline. Every level here is real MLIR you can dump from the `iree-compile` CLI with `--mlir-print-ir-after-all`.

## 1. StableHLO (input)

```mlir
func.func @main(%arg0: tensor<4x16xf32>, %arg1: tensor<16x32xf32>) -> tensor<4x32xf32> {
  %0 = stablehlo.dot_general %arg0, %arg1,
       contracting_dims = [1] x [0] : (tensor<4x16xf32>, tensor<16x32xf32>) -> tensor<4x32xf32>
  return %0 : tensor<4x32xf32>
}
```

Vendor-neutral. The `dot_general` carries enough information (which dims contract, batch dims, etc.) to be lowered for any backend. Spec: https://openxla.org/stablehlo/spec.

## 2. Linalg (after `--iree-input-type=stablehlo`)

```mlir
%init = tensor.empty() : tensor<4x32xf32>
%zero = arith.constant 0.0 : f32
%filled = linalg.fill ins(%zero : f32) outs(%init : tensor<4x32xf32>) -> tensor<4x32xf32>
%mm = linalg.matmul
        ins(%arg0, %arg1 : tensor<4x16xf32>, tensor<16x32xf32>)
        outs(%filled : tensor<4x32xf32>) -> tensor<4x32xf32>
```

`linalg.matmul` is structured: defined by index maps and iterator types, fusable with surrounding elementwise ops via `linalg.generic`. This is where most actual codegen optimization happens. Dialect docs: https://mlir.llvm.org/docs/Dialects/Linalg/.

Source pointers (IREE repo, https://github.com/iree-org/iree, `compiler/src/iree/compiler/InputConversion/StableHLO/`):
- `Preprocessing/StableHLOToLinalg.cpp` — the lowering itself.

## 3. Flow (dispatch region formation)

```mlir
flow.executable @main_dispatch_0 {
  flow.executable.export public @main_dispatch_0_matmul_4x32x16
  builtin.module {
    func.func @main_dispatch_0_matmul_4x32x16(%a: tensor<4x16xf32>, %b: tensor<16x32xf32>) -> tensor<4x32xf32> {
      %0 = linalg.matmul ins(...) outs(...) -> tensor<4x32xf32>
      return %0
    }
  }
}

func.func @main(...) -> tensor<4x32xf32> {
  %r = flow.dispatch @main_dispatch_0::@main_dispatch_0_matmul_4x32x16(%arg0, %arg1)
       : (tensor<4x16xf32>, tensor<16x32xf32>) -> tensor<4x32xf32>
  return %r
}
```

Now the matmul is *a unit of dispatchable work*. The compiler decided this kernel is its own region; surrounding ops would have been folded in if they were fusable. Pass: `iree-flow-form-dispatch-regions`. Source: `compiler/src/iree/compiler/Dialect/Flow/Transforms/FormDispatchRegions.cpp`.

## 4. Stream (async execution)

```mlir
%timepoint = stream.timepoint.immediate => !stream.timepoint
%result, %t1 = stream.async.dispatch @main_dispatch_0::...(%arg0, %arg1) : ... => !stream.timepoint
%ready = stream.timepoint.await %t1 => %result
```

Dispatches now carry an explicit async timeline. The runtime can overlap multiple dispatches and signal completion via timepoints. This is what lets IREE saturate a GPU command queue without explicit user threading.

## 5. HAL (device-portable)

```mlir
%cmd = hal.command_buffer.create device(%dev) ...
hal.command_buffer.dispatch<%cmd> target(@main_dispatch_0::@main_dispatch_0_matmul_4x32x16)
                                 workgroups([%x, %y, %z])
hal.command_buffer.execution_barrier<%cmd>
hal.command_buffer.finalize<%cmd>
```

The HAL dialect is device-agnostic: same IR, different runtime backend. The CUDA backend lowers `hal.command_buffer.dispatch` to `cuLaunchKernel`; the Vulkan backend to `vkCmdDispatch`. Source: `runtime/src/iree/hal/`.

## 6. Backend codegen

For CUDA: the dispatch region from step 3 has been independently lowered through GPU-specific dialects (`gpu`, `nvvm`) and emerges as PTX, which gets bundled into the `.vmfb`.
For Vulkan: same path, but lowering through `spirv` dialect and bundling as SPIR-V binaries.
For CPU: lowering through `vector` and `llvm` dialects; the kernel is a JIT-compiled function pointer.

## What to take from the walk

- A single source program produces multiple compiled kernels (one per target) but only one set of host orchestration code.
- The Flow→Stream→HAL chain is what makes IREE's runtime tiny: by the time you reach HAL, the program is already cut into device-portable command-buffer pieces.
- Linalg is the hot spot for codegen quality. Most of IREE's perf work in 2024–2025 was Linalg-level transformation passes (tiling, fusion, vectorization).
