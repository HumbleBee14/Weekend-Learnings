# 05 — llama.cpp Deep Dive

## What it is

A C/C++ tensor library (GGML) + a model file format (GGUF) + a family of backends (CPU, CUDA, Metal, Vulkan, ROCm, SYCL, OpenCL). Started 2023 as "run LLaMA on a MacBook"; now the dominant local-inference runtime and a credible CUDA serving option for batch=1 latency-sensitive workloads.

It's also the project that, in May 2026, finally landed FP4 (NVFP4 / MXFP4) support — meaning you can run an FP4-quantized Llama on a CUDA GPU through llama.cpp, not just through TRT-LLM.

## Why it's in this week

Don't dismiss llama.cpp as "the consumer one." Production ML teams reach for it when:

- **Batch=1 latency is the SLA** on a small/mid model. Cold-start is faster, ITL is competitive, no Python overhead.
- **CPU-only deployment at low QPS.** A 16-core EPYC node can serve a 7B at batch=1 cheaper than a small GPU once you account for $/Mtok.
- **Apple Silicon production.** Metal backend is mature; M-series unified memory makes it the obvious choice on Mac.
- **Edge / embedded.** Jetson, Orin, ARM SoCs. llama.cpp's no-Python build is a feature.
- **Local-first stack.** Ollama, LM Studio, GPT4All, Jan, Cortex — all wrap llama.cpp.

Project 2's break-it list explicitly includes a llama.cpp-on-CPU vs small-GPU cost-per-token comparison.

## GGML / GGUF in one diagram

```
                ┌──────────────────────────────────────────┐
                │  GGUF file                                │
                │  ┌────────────────────────────────────┐  │
                │  │ Header (magic, version, n_tensors) │  │
                │  ├────────────────────────────────────┤  │
                │  │ Metadata (architecture, vocab,     │  │
                │  │   chat template, quant scheme,     │  │
                │  │   tokenizer model — all in-file)   │  │
                │  ├────────────────────────────────────┤  │
                │  │ Tensor index                       │  │
                │  ├────────────────────────────────────┤  │
                │  │ Tensor data (mmap'd at load time)  │  │
                │  │   - per-tensor quant type          │  │
                │  │   - block layout (32 weights/block │  │
                │  │     for K-quants, 256 for i-quants)│  │
                │  └────────────────────────────────────┘  │
                └──────────────────────────────────────────┘

                              │ mmap, no copy
                              ▼
                ┌──────────────────────────────────────────┐
                │  GGML runtime                             │
                │  - per-op dispatch (matmul, attention,    │
                │    rmsnorm, rope, softmax)                │
                │  - per-quant kernels per backend          │
                │  - SIMD on CPU (AVX-512, AVX2, NEON)      │
                │  - CUDA / Metal / Vulkan kernels          │
                └──────────────────────────────────────────┘

                              │
                              ▼
                ┌──────────────────────────────────────────┐
                │  llama-server (HTTP, OpenAI-compat)       │
                │  llama-cli   (interactive)                │
                │  llama-bench (microbench)                 │
                └──────────────────────────────────────────┘
```

GGUF is **self-describing** — vocab, chat template, tokenizer are all inside the file. That's why a single `.gguf` works across runtimes (llama.cpp, Ollama, LM Studio, candle) without tokenizer drift.

## The quant family

Level 4 Topic 04 covers quantization in depth. The summary that matters here:

```
K-quants     Q4_K_M, Q5_K_M, Q6_K
             32-element blocks, 6-bit superblock scales
             The 2023-era classic. Still the safe default.

i-quants     IQ4_XS, IQ3_XS, IQ2_XS, IQ1_M
             256-element blocks, importance-aware codebooks
             2024+. Smaller files, similar quality at the same bpw.

Imatrix      Calibration-set-driven importance for i-quants
             Most modern HF GGUF uploads use imatrix by default.

Unsloth Dynamic v2.0
             2026 frontier. Beats both K- and i-quants at small sizes.
             Look for "UD" or "Dynamic" in filenames.

NVFP4 / MXFP4
             Just landed in llama.cpp (May 2026). Local users get FP4.
```

For the bake-off, pick one quant per engine that's apples-to-apples — usually FP8 for vLLM/SGLang/TRT-LLM, Q4_K_M (or UD-Q4) for llama.cpp.

## Where it wins, where it loses

```
Workload                                  llama.cpp result
──────────────────────────────            ────────────────
Batch=1 chat on a 7B, A10/L4              competitive with vLLM
Batch=1 chat on a 7B, M3 Max              wins (Metal mature)
Batch=64 chat on a 7B, A10                loses badly (continuous batching less mature)
CPU-only batch=1 on EPYC                  competitive on $/Mtok at low QPS
Long context (32K+) at high QPS           loses (paged KV less optimized)
Embedding models                          wins on CPU; close on GPU
Multi-LoRA                                limited; not the primary use case
Structured output                         present (grammar.cpp), less polished than xgrammar
Multimodal (Qwen2.5-VL, etc.)             supported via llava family; ahead of TRT-LLM, behind vLLM
```

## llama-server is OpenAI-compatible

```bash
llama-server -m model.gguf --host 0.0.0.0 --port 8003 \
    --ctx-size 8192 -ngl 999 -t 8
```

Flags:
- `-ngl` — number of layers to offload to GPU (999 = all)
- `-t` — CPU threads (matters for hybrid CPU/GPU and CPU-only)
- `--ctx-size` — context budget per request
- `--parallel` / `-np` — number of slots (continuous-ish batching; less mature than vLLM)
- `--cache-type-k`, `--cache-type-v` — KV quantization (q4_0, q8_0, f16)

It serves `/v1/chat/completions` and `/v1/embeddings`. Same OpenAI client code as Topics 01, 03, 04.

## CPU SIMD — when CPU actually beats GPU

Level 8 Topic 09 goes deeper. The quick framing:

- llama.cpp's CPU backend uses AVX-512 / AVX2 / NEON / AMX kernels per architecture.
- For a 7B at Q4 on a 16-core EPYC, you can sustain ~30-50 tok/s at batch=1.
- A small cloud GPU (L4 / A10) at $0.40-0.80/hr does ~60-150 tok/s.
- $/Mtok crossover is workload-conditional: if your QPS is low and your nights are quiet, CPU wins.
- AMX on Sapphire Rapids / Granite Rapids and SME2 on Apple M-series further close the gap.

## Pitfalls

1. **Treating it as "obviously slower."** Measure. On batch=1 7B on Mac, llama.cpp with Metal is faster than vLLM-on-CUDA-rented-GPU by an order of magnitude in cost per token.
2. **Forgetting `-ngl`.** Default may not offload all layers. Check `nvidia-smi` / activity monitor; if the GPU isn't pegged, you're partially on CPU.
3. **Comparing K-quants to FP8.** Different quality tradeoffs. Use `lm-eval-harness` (Level 4 Topic 06) before declaring a winner on quality.
4. **Continuous batching isn't llama.cpp's strong suit.** `--parallel` exists; it's not vLLM-class. For high-QPS serving, vLLM/SGLang/TRT-LLM win.
5. **Forgetting the Metal+Mac sweet spot.** On M-series, llama.cpp is *the* serving stack. Don't run vLLM-CPU on Mac and call it a comparison.

## What to do this topic

1. Install: `brew install llama.cpp` (Mac) or build from source for CUDA / Vulkan.
2. Download a GGUF (`huggingface-cli download bartowski/Qwen2.5-7B-Instruct-GGUF Qwen2.5-7B-Instruct-Q4_K_M.gguf`).
3. Run `llama-server` on port 8003.
4. Hit it with the same OpenAI-client harness from Topic 01 (`serve_and_hit.py`).
5. Run `bench_local.py` (this folder) for the CPU-only and GPU paths.

## References

- llama.cpp source — https://github.com/ggml-org/llama.cpp
- GGUF spec — https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
- llama-server docs — https://github.com/ggml-org/llama.cpp/tree/master/tools/server
- llama-bench — https://github.com/ggml-org/llama.cpp/tree/master/tools/llama-bench
- Quantization guide (Hugging Face) — https://huggingface.co/docs/hub/gguf
- FP4 in llama.cpp PR thread — https://github.com/ggml-org/llama.cpp/pulls?q=fp4
- Unsloth Dynamic v2.0 GGUFs — https://docs.unsloth.ai/basics/dynamic-v2.0-ggufs
