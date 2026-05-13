"""
quantize_to_nvfp4.py — fuse NVFP4 quantization into the GEMM epilogue.

NVFP4 layout (per Blackwell, with hardware support):
  - 4-bit FP values (E2M1, 1 sign + 2 exp + 1 mantissa) packed 2-per-byte
  - Per-16-element E8M0 block scales (one scale per group of 16 along N)
  - Per-tensor FP32 scale (often computed offline)

The epilogue per output tile:
  1. Block-max-reduce over each 16-element N stride.
  2. Derive E8M0 scale = ceil(log2(block_max / FP4_max)).
  3. Quantize: round(acc / 2^scale), clamp to FP4 range.
  4. Pack two FP4 values per byte.
  5. Store quantized data + scales.

EVT structure (two store roots, shared BlockMax):

      Store(D_q)     Store(S)
         |              |
       PackFP4       EmitScale
         |              |
       Quantize  ─→ BlockMax
                       |
                    Accumulator

Hardware: B200. The cvt.rn.satfinite.e2m1 instruction is SM100+.
Run:
    python quantize_to_nvfp4.py

The file is structured as a read-along; running requires a Blackwell-capable
CuTe-DSL build (cutlass.cute.dtype.NVFP4 etc.) and a B200.
"""

import torch
import cutlass
import cutlass.cute as cute
# These imports name the API as it's converging in CUTLASS 4.x; if your
# install differs, the matching upstream example is
# examples/python/CuTeDSL/blackwell/dense_blockscaled_gemm_persistent.py.
from cutlass.cute.epilogue import (
    AccumulatorSource, Reduce, Compute, Store, EpilogueVisitorTree,
)


BLOCK_SCALE_N = 16            # NVFP4 block-scale granularity along N


def build_quantize_evt(out_data_desc, out_scale_desc):
    acc = AccumulatorSource()                                # FP32 accumulator

    # Block-max reduction along N in groups of 16.
    block_max = Reduce(op="amax", input=acc.abs(), axis="N", group=BLOCK_SCALE_N)

    # E8M0 scale: log2 of (block_max / FP4_max), rounded up, clamped.
    # FP4 E2M1 max = 6.0 (the largest finite representable).
    scale = Compute(
        op="cvt_to_e8m0",
        input=Compute(op="ceil_log2", input=block_max / 6.0),
    )

    # Quantize: round(acc / 2^scale) to FP4.
    quant_factor = Compute(op="ldexp", value=1.0, exponent=-scale)
    quant_fp4 = Compute(op="cvt_to_nvfp4", input=acc * quant_factor)

    # Two store roots: quantized data and per-block scales.
    store_data = Store(quant_fp4, out_data_desc)
    store_scale = Store(scale, out_scale_desc)

    return EpilogueVisitorTree([store_data, store_scale])


def main():
    M, K, N = 4096, 4096, 4096
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10:
        print("NVFP4 epilogue requires SM100 (B200). Read the file structure;")
        print("running it needs Blackwell hardware.")
        return

    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    W = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)

    # Allocate outputs:
    #   - Quantized data: M x N FP4, packed → M x (N // 2) bytes
    #   - Block scales:   M x (N // 16) E8M0
    out_data = torch.empty(M, N // 2, device="cuda", dtype=torch.uint8)
    out_scales = torch.empty(M, N // BLOCK_SCALE_N, device="cuda", dtype=torch.uint8)

    # Build EVT and run GEMM with this epilogue. Plug into the SM100 persistent
    # kernel from submodule 05's walkthrough_03. The wiring is left for the
    # reader — see dense_blockscaled_gemm_persistent.py upstream.

    print("EVT topology built. Plug into the SM100 persistent GEMM mainloop.")
    print(f"out_data shape={tuple(out_data.shape)}, dtype=FP4 packed")
    print(f"out_scales shape={tuple(out_scales.shape)}, dtype=E8M0")


if __name__ == "__main__":
    main()
