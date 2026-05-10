# 04 — Local Quant Formats

## Files

- `CONCEPTS.md` — GGUF K-quants vs i-quants, when each wins, Unsloth Dynamic v2.0, FP4 in llama.cpp (new in 2026), EXL2's decline

## What you do this topic

No new code — read CONCEPTS.md, then run a few quick experiments using `llama.cpp` directly.

## Quickstart

```bash
# Build llama.cpp with Metal (Mac) or CUDA (Linux/Windows)
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp && cmake -B build && cmake --build build --config Release -j

# Pull a few quants of the same model
mkdir gguf-tests && cd gguf-tests
huggingface-cli download bartowski/Qwen2.5-1.5B-Instruct-GGUF \
    --include "*Q4_K_M.gguf" --local-dir .
huggingface-cli download bartowski/Qwen2.5-1.5B-Instruct-GGUF \
    --include "*IQ4_XS.gguf" --local-dir .
huggingface-cli download bartowski/Qwen2.5-1.5B-Instruct-GGUF \
    --include "*IQ3_M.gguf" --local-dir .

# Same for Unsloth Dynamic v2 (when available)
huggingface-cli download unsloth/Qwen2.5-1.5B-Instruct-GGUF \
    --include "*UD-Q4_K_XL.gguf" --local-dir .

# Run them and compare tok/s
../build/bin/llama-bench -m Qwen2.5-1.5B-Instruct-Q4_K_M.gguf
../build/bin/llama-bench -m Qwen2.5-1.5B-Instruct-IQ4_XS.gguf
../build/bin/llama-bench -m Qwen2.5-1.5B-Instruct-IQ3_M.gguf
```

## What you should observe

```
Format         File size       tok/s (M2 Mac CPU)    tok/s (M2 Mac GPU/Metal)
Q4_K_M         900 MB          ~25                    ~75
IQ4_XS         820 MB          ~18                    ~70           (smaller, slower CPU)
IQ3_M          720 MB          ~14                    ~65           (much smaller, slower)
UD-Q4_K_XL     ~870 MB         ~24                    ~75           (Unsloth Dynamic v2)
```

Numbers approximate. The pattern: i-quants are smaller and lower quality drop, but slower decode on CPU. K-quants faster on CPU. Unsloth Dynamic competitive on size *and* speed.

## Try

- **Same model, different sources.** Download `Q4_K_M` from `bartowski/...`, `unsloth/...`, and `lmstudio-community/...`. Compare quality. Quality varies because of imatrix differences.
- **Compute KL-divergence vs the BF16 reference.** This is the right quality metric (covered in Topic 06). Each tier matters less in absolute size and more in KL gap.
- **Try FP4 in llama.cpp.** New in 2026. NVFP4 or MXFP4 on supported hardware.
- **Compare GGUF on Mac with MLX-converted equivalents** (Level 8 covers MLX). MLX is often 2× faster on M-series.

## Where this goes

- Topic 05 — extreme quantization (BitNet, IQ2_M)
- Level 8 — full local-first week (Apple Silicon, MLX, agentic local stacks)
- For datacenter quantization, Topics 02-03 are the path
