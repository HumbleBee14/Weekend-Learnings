# NVFP4 walkthrough — annotating the SM100 block-scaled GEMM

> Pairs with [`cutlass/examples/python/CuTeDSL/blackwell/dense_blockscaled_gemm_persistent.py`](https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/blackwell/dense_blockscaled_gemm_persistent.py)
> The optional capstone extension. Requires B200 to run; readable on any machine.

The NVFP4 path is structurally the same as your BF16 persistent GEMM (`gemm.py`). What changes:

## Inputs

- A: `(M, K)` FP4, packed two-per-byte. Physical shape `(M, K/2)` `uint8`.
- B: `(K, N)` FP4, packed two-per-byte. Physical shape `(K/2, N)` `uint8`.
- A_scales: `(M, K/16)` E8M0 (one byte per scale).
- B_scales: `(K/16, N)` E8M0.
- Output: `(M, N)` BF16 or FP8 typically — the post-GEMM dtype is your choice.

## What changes in the kernel

### MMA atom

```python
# BF16:
from cutlass.cute.nvgpu.warpgroup import SM90_64x128x16_F32BF16BF16_SS
# NVFP4:
from cutlass.cute.nvgpu.tcgen05 import SM100_MMA_F32_NVFP4_BS_2SM_SS
```

The atom name encodes: F32 accumulator, NVFP4 inputs, Block-Scaled, 2-SM cluster, SMEM-to-SMEM operand layout. The MMA reads the block scales natively — no manual dequantize step.

### TMA descriptors

Two new descriptors for the scales. They're loaded into SMEM and fed to the MMA alongside the data tiles.

```python
tma_a_data    = cute.create_tma_atom(SM90_TMA_LOAD, mA, box_shape=(BLOCK_M, BLOCK_K // 2))
tma_a_scales  = cute.create_tma_atom(SM90_TMA_LOAD, mA_scales, box_shape=(BLOCK_M, BLOCK_K // 16))
# similarly for B
```

Note `BLOCK_K // 2` (FP4 is half the bytes) and `BLOCK_K // 16` (one scale per 16 elements).

### Tile sizes

NVFP4 throughput is 2× FP8 on Blackwell, so you can use larger tiles to keep the tensor cores fed:

- `BLOCK_M = 256` (2-SM cooperative MMA: 128 per CTA)
- `BLOCK_N = 256`
- `BLOCK_K = 128` (so each scale group of 16 packs into a clean partition)

### Accumulator

Same FP32 TMEM accumulator as BF16. The MMA does the dequantize-multiply-accumulate internally; the result coming out of TMEM is FP32 just like BF16.

### Epilogue

Same as BF16. The accumulator → BF16 (or whatever output dtype) → TMA store. If you want NVFP4 output (for cascading FP4 → FP4 layers), use the EVT from submodule 06 (`quantize_to_nvfp4.py`).

## What stays exactly the same

- Persistent grid, internal tile loop.
- Producer/consumer warp specialization.
- Multi-stage SMEM pipeline.
- The CuTe layout algebra for SMEM tiles.
- The mbarrier protocol.
- TMEM allocate/dealloc.

## Numbers to expect

On B200 at M=N=K=4096:

| Kernel | dtype | TFLOPS (dense) | % cuBLAS |
|---|---|---|---|
| cuBLAS BF16 | BF16 | ~2200 | 100% (BF16 reference) |
| Your CuTe-DSL BF16 | BF16 | ~1900 | ~86% |
| cuBLAS NVFP4 | NVFP4 | ~4400 | 100% (NVFP4 reference) |
| `dense_blockscaled_gemm_persistent.py` | NVFP4 | ~3900 | ~89% |

The NVFP4 path delivers ~2× the BF16 throughput on the same hardware, with model-accuracy degradation typically <1% on language modeling tasks (the NVIDIA blog post claims this; verify on your model).

## What this means for FlashAttention-4 and friends

FA4 on B200 uses the same NVFP4 path for its FP4-resident attention scores in some configurations. The block-scaling lets the QK^T product retain enough dynamic range to feed into softmax without per-element FP32 promotion. This is what's behind the ~20% FA4-vs-cudnn speedup on Blackwell.

## References

- [NVIDIA: Introducing NVFP4](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)
- [Colfax: Hardware-supported Block-scaling on Blackwell](https://research.colfax-intl.com/cutlass-tutorial-hardware-supported-block-scaling-with-nvidia-blackwell-gpus/)
- [Colfax: Sub-byte GEMM on Blackwell](https://research.colfax-intl.com/cutlass-tutorial-sub-byte-gemm-on-nvidia-blackwell-gpus/)
- [PyTorch: Faster Diffusion on Blackwell — MXFP8 and NVFP4](https://pytorch.org/blog/faster-diffusion-on-blackwell-mxfp8-and-nvfp4-with-diffusers-and-torchao/)
