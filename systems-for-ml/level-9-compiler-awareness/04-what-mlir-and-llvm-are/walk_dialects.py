"""A reading-only walkthrough: print canonical MLIR snippets for the
dialects you'll see in ML compiler stacks, in lowering order.

This script does not invoke mlir-opt — it just emits well-formed text so
you can read the shape of each dialect in one place. To run real MLIR
passes, install one of:

  - the LLVM/MLIR toolchain (mlir-opt, mlir-translate)
  - the IREE compiler (`iree-compile`) — Topic 06
  - JAX (lowered.as_text("stablehlo")) — Topic 03
"""

SNIPPETS = {
    "stablehlo (high-level, ML-shaped)": """\
func.func @block(%x: tensor<4x64xf32>, %w1: tensor<64x128xf32>) -> tensor<4x128xf32> {
  %0 = stablehlo.dot_general %x, %w1,
         contracting_dims = [1] x [0] : (tensor<4x64xf32>, tensor<64x128xf32>) -> tensor<4x128xf32>
  %1 = stablehlo.logistic %0 : tensor<4x128xf32>
  %2 = stablehlo.multiply %0, %1 : tensor<4x128xf32>
  return %2 : tensor<4x128xf32>
}
""",
    "linalg (structured loop-nest abstraction)": """\
func.func @matmul(%A: tensor<4x64xf32>, %B: tensor<64x128xf32>) -> tensor<4x128xf32> {
  %c0 = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<4x128xf32>
  %acc = linalg.fill ins(%c0 : f32) outs(%init : tensor<4x128xf32>) -> tensor<4x128xf32>
  %r = linalg.matmul ins(%A, %B : tensor<4x64xf32>, tensor<64x128xf32>)
                     outs(%acc : tensor<4x128xf32>) -> tensor<4x128xf32>
  return %r : tensor<4x128xf32>
}
""",
    "scf + vector (loops + SIMD/SIMT)": """\
func.func @tiled(%A: memref<4x64xf32>, %B: memref<64x128xf32>, %C: memref<4x128xf32>) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c128 = arith.constant 128 : index
  %c8 = arith.constant 8 : index
  scf.for %i = %c0 to %c4 step %c8 {
    scf.for %j = %c0 to %c128 step %c8 {
      // tile body would compute an 8x8 block via vector.contract on tensor cores
      // (omitted: this is illustrative, not runnable)
    }
  }
  return
}
""",
    "gpu / nvgpu (hardware-aware launch)": """\
func.func @launch(%A: memref<4x64xf32>, %B: memref<64x128xf32>, %C: memref<4x128xf32>) {
  gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1)
             threads(%tx, %ty, %tz) in (%bsx = %c128, %bsy = %c1, %bsz = %c1) {
    // nvgpu.mma.sync would issue a tensor-core MMA here
    gpu.terminator
  }
  return
}
""",
    "llvm (the bottom; emit via LLVM toolchain)": """\
llvm.func @kernel(%a: !llvm.ptr<f32>, %b: !llvm.ptr<f32>, %c: !llvm.ptr<f32>) {
  // scalar/vector LLVM IR: loads, FMAs, stores, plus NVPTX intrinsics for tensor cores
  llvm.return
}
""",
}


def main() -> None:
    for title, body in SNIPPETS.items():
        print(f"=== {title} ===")
        print(body)


if __name__ == "__main__":
    main()
