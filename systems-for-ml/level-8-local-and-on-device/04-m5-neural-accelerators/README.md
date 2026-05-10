# 04 — M5 Neural Accelerators

## Files

- `CONCEPTS.md` — what NAs are (per-GPU-core matmul units), how MLX targets them, dtype eligibility, why decode at batch 1 still doesn't win much, the prefill/training story.
- `probe_na.py` — fp32 vs fp16 vs bf16 matmul TFLOPS on your Mac. NA-equipped chips show 3–4× fp16/bf16 over fp32.

## Quickstart

```bash
pip install mlx
python probe_na.py
```

## Expected output

M5 Max:

```
   dtype     TFLOPS
    fp32       6.50
    fp16      26.40
    bf16      26.10
```

M3 Max (no NAs, FMA path everywhere):

```
   dtype     TFLOPS
    fp32       7.20
    fp16      15.90
    bf16      15.50
```

The M3 fp16 is ~2x fp32 (wider FMA). The M5 fp16 is ~4x fp32 — that extra factor is the Neural Accelerator.

## Try

- Test fp8 if your MLX version supports it (`mx.float8_e4m3`). On M5 Max with macOS 26.2+ and MLX 0.26+, this should land ~50 TFLOPS.
- Compare prefill speeds in `mlx_lm.generate` between `--max-tokens 1` (TTFT only, prefill-bound) and `--max-tokens 1024` (decode-bound). Prefill ratio scales with NA gain; decode ratio is closer to memory-bandwidth ratio.
- Skim the Apple MLX-on-M5 paper, look at the MoE numbers — sparse models surface NA gains during decode because each token's compute is still substantial.

## Where this goes

This is the substrate behind Topic 03's MLX advantage on M5 hardware. Topic 05 (Metal shaders) shows how to hand-write code that targets `simdgroup_matrix` to use NAs from custom kernels.
