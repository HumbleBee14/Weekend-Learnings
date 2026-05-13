# Notes — 01

Your own answers to the eight diagnostic questions go here. Don't peek at the reference answers below until you've written yours.

## My answers

1.
2.
3.
4.
5.
6.
7.
8.

---

## Reference answers (peek after writing your own)

1. **cuBLAS is closed, tuned for HPC shapes, and slow to adopt new precisions.** LLM inference has decode shapes (M=1..8) where cuBLAS misses on tile sizing, and needs NVFP4/MXFP8 with hardware block scaling — features that arrive in CUTLASS first.
2. **GEMM-shaped = the inner loop is `C += A @ B` for some tile.** Linear layers, attention QK and PV products, MoE expert matmuls, embedding lookups (degenerate GEMM). For LLaMA-70B BF16, GEMM accounts for >95% of FLOPs.
3. The vLLM FP8 GEMM is `cutlass::gemm::kernel::Sm90GemmTmaWarpSpecializedFP8...` with template params for tile shape (typically 128×128×128), cluster shape (often (1,1,1) or (2,1,1)), mainloop schedule (warp-specialized cooperative or ping-pong), and an EVT epilogue (scale + bias).
4. **CUTLASS is the library of kernels; CuTe is the algebra that those kernels are written in.** You can use CUTLASS without ever touching CuTe (pick a Gemm template and parametrize it). You cannot write a new kernel without CuTe.
5. ThunderKittens is a higher-level abstraction with its own tile primitives, owned externally. CuTe-DSL shares the algebra with the C++ CUTLASS library (so a kernel idea ports between them), is owned by NVIDIA (so it tracks hardware day-zero), and reuses CUTLASS's MMA atoms and TMA descriptors directly.
6. **TMEM** — 256 KB on-SM register-file-adjacent storage that now holds the MMA accumulator. **`tcgen05.mma`** — issued by one thread per CTA (vs WGMMA's full warpgroup). **2-SM cooperative MMA** — pairs of CTAs share one MMA tile to saturate the bigger tensor cores. **NVFP4 / MXFP8** — 4- and 8-bit precisions with hardware block scaling that the MMA reads natively.
7. FA4 uses (i) five distinct warp roles including dedicated correction and softmax warps; Triton's warp specialization is producer/consumer only. (ii) a TMEM-resident accumulator with explicit `tcgen05.ld` moves; Triton can't address TMEM directly. (iii) a cubic-polynomial softmax to avoid the SFU bottleneck; Triton doesn't expose the inline-PTX path easily. (iv) 2-SM cooperative MMA on Blackwell.
8. **Triton wins for:** bandwidth-bound ops (LayerNorm, RMSNorm, softmax, residual streams), custom elementwise kernels, kernels where tile shape and warp count are flexible, and prototyping. **`torch.compile` wins for:** anything Inductor already covers well (most pointwise+reduction fusion), since it's free. **CuTe-DSL / CUTLASS wins for:** GEMM at peak, new precisions with hardware block scaling, kernels that need TMEM and `tcgen05`, kernels where the tile-to-warp mapping is the limit.
