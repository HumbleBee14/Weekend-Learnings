# Capstone — BF16 persistent GEMM in CuTe-DSL, benchmarked

> Outer: [`../README.md`](../README.md) · Hardware: H100 ideal. A100 acceptable with caveats. B200 for NVFP4 extension.

The capstone takes the stage-5 persistent GEMM from submodule 04 and turns it into a tuned, benchmarked, lift-and-use BF16 GEMM kernel. The bar: **stage-5 within 15% of cuBLAS on 4096³ BF16 on your hardware**, with a written report.

The NVFP4 extension (optional, B200 only) annotates the upstream SM100 block-scaled GEMM and runs it for a comparison row.

## What's in this folder

```
_capstone-bf16-persistent-gemm/
├── README.md                       this file
├── gemm.py                         the tuned kernel
├── benchmark.py                    runs the benchmark grid and emits the table
├── nvfp4_walkthrough.md            annotated NVFP4 SM100 kernel notes
└── reports/
    └── REPORT_TEMPLATE.md          fill this in
```

## Step-by-step

### 1. Lift stage-5 and clean it up

Copy `stage5_persistent.py` from submodule 04 into `gemm.py`. Then:

- Add a module-level docstring describing what it does, what hardware it targets, what shapes it's tuned for.
- Add a one-line docstring to each `@cute.kernel` and `@cute.jit` function.
- Comment every CuTe layout expression. If a stride is non-obvious, explain why it is what it is.
- Add a `pytest`-style test that runs the kernel on M=N=K=512 BF16 and verifies against `torch.matmul` to 1e-2 atol.

The point is that someone reading your kernel cold can follow it. Production code in vLLM and TRT-LLM passes this bar.

### 2. Tune

Pick H100 or A100 explicitly. The tuning space:

| Knob | Values | Notes |
|---|---|---|
| `BLOCK_M` | 64, 128, 256 | 128 default; 256 only with cluster |
| `BLOCK_N` | 64, 128, 256 | 128 default; 256 only with cluster |
| `BLOCK_K` | 32, 64, 128 | 64 default; 32 for small-K shapes |
| `NUM_STAGES` | 2, 3, 4 | 3 default; 4 needs careful SMEM accounting |
| `cluster` | (1,1), (2,1), (2,2) | (1,1) for single-SM; (2,1) for 2-CTA cooperative on H100/B200 |

Many combinations are illegal:
- `BLOCK_M * BLOCK_K * 2 * NUM_STAGES + BLOCK_N * BLOCK_K * 2 * NUM_STAGES > 228 KB` (H100 SMEM per CTA) — fails.
- `BLOCK_M * BLOCK_N * 4` (FP32 accumulator) won't fit in registers — fails.
- WGMMA atoms only support certain `(BLOCK_M, BLOCK_N, BLOCK_K)` combinations — others fail.

Write a `is_valid_config(...)` function that prunes these before launching. The autotune sweep should take minutes, not hours.

### 3. Benchmark

Run `python benchmark.py`. The script:

1. Loads cuBLAS BF16 reference.
2. Loads your Triton matmul from `compiler-and-kernels/level-1-triton-deep-dive/04-tiled-matmul-and-autotune/` (you wrote this in Level 1).
3. Loads each stage from submodule 04 (stages 1–5).
4. Runs each on a grid of shapes:
   - Square: M=N=K ∈ {512, 1024, 2048, 4096, 8192}
   - LLaMA FFN-1: M=8192, K=4096, N=11008
   - LLaMA QKV: M=8192, K=4096, N=12288 (Q+K+V combined)
   - Decode: M=8, K=4096, N=4096
5. Emits the markdown benchmark table.

The script handles warmup (25 iters), measurement (100 iters), correctness check (vs `torch.matmul`), and the percent-of-cuBLAS computation. It writes the table to `reports/benchmark_<hostname>_<date>.md`.

### 4. Write the report

Use `reports/REPORT_TEMPLATE.md`. The five sections:

1. **Hardware and software.** Exact GPU, driver, CUDA version, CuTe-DSL version, Torch version, the cuBLAS that `torch.matmul` resolves to (check via `CUBLAS_LOGINFO_DBG=1`).
2. **Numbers.** The full table. No edits.
3. **Where you matched cuBLAS, where you fell short.** Per shape, one sentence. The likely suspects: small-K shapes need smaller `BLOCK_K`; non-square shapes need cluster shape changes; decode shapes need a different mainloop entirely.
4. **What each tuning knob bought.** One sentence each for the knobs that moved the needle.
5. **Two surprises.** Things you predicted wrong. These are the most valuable parts of the report. If you had no surprises, you weren't paying attention.

### 5. (Optional, B200) NVFP4 extension

Open [`cutlass/examples/python/CuTeDSL/blackwell/dense_blockscaled_gemm_persistent.py`](https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/blackwell/dense_blockscaled_gemm_persistent.py). Annotate every line that differs from your SM90 BF16 GEMM. The deltas:

- MMA atom: `SM100_F4_2SM_NVFP4_BS` or similar (FP4-input WGMMA with block scaling)
- Per-block scale loads (separate TMA descriptor for the scale tensor)
- Smaller per-element bandwidth (FP4 is 4× denser than BF16)
- 2-SM cooperative cluster default
- TMEM accumulator (still FP32; only inputs are FP4)

Save the annotated file as `nvfp4_walkthrough.md` with inline commentary. If you have B200, run it and add an "NVFP4 / cuBLAS NVFP4" row to your benchmark table.

## Definition of done

- [ ] `gemm.py` is documented, tested, and passes correctness on 4 different shapes.
- [ ] Tuning sweep with `is_valid_config` pruning produces a winning config in < 5 minutes.
- [ ] Benchmark on H100 (or A100): stage-5 within 15% of cuBLAS on 4096³ BF16.
- [ ] Report filled in, including two named surprises.
- [ ] (Optional) NVFP4 walkthrough annotated; if B200, benchmark row added.

## Reference numbers (what to expect)

On H100 SXM (132 SMs), CUDA 12.5, CuTe-DSL 4.5:

| Shape | cuBLAS BF16 | Your Triton (Level 1) | Your CuTe-DSL stage 5 |
|---|---|---|---|
| 4096³ | ~620 TFLOPS (100%) | ~480 TFLOPS (~77%) | ~550 TFLOPS (~89%) |
| LLaMA FFN-1 | ~600 TFLOPS | ~470 TFLOPS | ~530 TFLOPS |
| Decode M=8 | ~30 TFLOPS | ~20 TFLOPS | ~35 TFLOPS* |

\* The decode shape is where persistent grids win — non-persistent kernels lose to launch overhead. The "%" of cuBLAS at M=8 is misleading because cuBLAS isn't tuned for this shape; the absolute TFLOPS comparison vs Triton is what matters.

If your H100 stage-5 lands above 90% of cuBLAS on 4096³, double-check the timing (warmup, correct shape, launch actually happening) before celebrating. CuTe-DSL can beat cuBLAS on specific shapes but it's not the common case.

## Hand-off

A successful capstone is one a colleague could pick up, drop into their own benchmark, and start tuning further. The kernel in `gemm.py` should be clean enough for that.

The next level of this track (Level 5 — Kernel Fusion) extends this work by *replacing* the GEMM mainloop in places — sometimes you fuse the GEMM into an upstream op, sometimes you fuse a downstream op into the epilogue. Both directions become tractable once you have a working CuTe-DSL GEMM.
