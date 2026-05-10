# 04 — TensorRT-LLM

## What it is

NVIDIA's high-performance LLM inference library. The throughput leader on Hopper (H100/H200) and Blackwell (B200/GB300), especially with FP8 and NVFP4. Built on top of **TensorRT** (the runtime) and increasingly contributing kernels to **FlashInfer** (the cross-engine kernel layer).

Two flows in 2026:

- **PyTorch flow** — Python-first, looks like vLLM. The default since TRT-LLM 1.0.
- **C++ engine-build flow** — the original 2023 model. Build a hardware-specific engine plan via `trtllm-build`, run it via `trtllm-serve`. Still supported, still the throughput champion when fully tuned, increasingly viewed as legacy for new development.

The PyTorch flow has largely replaced the engine-build flow for new projects. NVIDIA's own examples lead with it.

## NIM — the customer-facing wrapper

NVIDIA Inference Microservices. A containerized package: TRT-LLM under the hood + Triton Inference Server in front + a curated catalog of pre-tuned engines per (model, GPU, precision) combination. NIM is what enterprises actually buy from NVIDIA. TRT-LLM is the engine inside.

For Project 2: benchmark TRT-LLM directly. NIM adds container-management overhead that's irrelevant to the engine comparison.

## What TRT-LLM does that vLLM doesn't (as cleanly)

- **NVFP4 with two-level scaling on Blackwell** — Hopper's FP8 was already best-in-class; NVFP4 doubles the throughput at minor quality cost. TRT-LLM ships the most polished NVFP4 path (Level 4 Topic 02 covers the math).
- **Kernel auto-tuning per (shape, precision, GPU)** — the build step picks the best kernel implementation per layer. vLLM's FlashInfer also tunes, but TRT-LLM's tuning library is older and broader.
- **Multi-block attention** — splits long-context attention across SMs differently. Helps decode at very long context.
- **In-flight batching with chunked context** — same idea as vLLM's continuous batching + chunked prefill, but the C++ implementation has lower per-step CPU overhead.

## What vLLM closed in 2025-2026

The TRT-LLM throughput lead has narrowed substantially:

- vLLM gained FlashInfer kernels (many of them contributed by NVIDIA).
- vLLM's V1 scheduler eliminated most of the Python overhead that gave TRT-LLM the edge on small models.
- Piecewise CUDA-graph capture (Level 4 Topic 07) closed the launch-overhead gap.
- FP8 paths in vLLM via llm-compressor are mature.

In 2026, "TRT-LLM is faster" is true on most workloads — by 10-30% — but not by the 2-3× it was in 2023. Operational cost (install pain, build times, harder debug) has to be weighed against the perf delta.

## Operational cost — the honest part

TRT-LLM is heavier than vLLM in every operational sense:

```
                     vLLM           TRT-LLM
                     ────           ───────
install              1 pip          CUDA + cuDNN + cuBLAS pinned;
                                    multi-GB wheel; sometimes builds from source
first-time setup     1 min          5-30 min (engine build per shape/precision)
debugging            Python tb      C++ stack + opaque kernel errors
adding a new model   merge a PR     wait for kernel support OR contribute
shape changes        free           rebuild
GPU change           free           rebuild
```

For a bake-off, document the friction. *Operational cost is part of engine selection.* Production teams deal with this constantly.

## When to actually pick TRT-LLM

```
Pick TRT-LLM when:                     Pick vLLM/SGLang when:
──────────────────────                ──────────────────────
Hopper or Blackwell fleet             Mixed GPU fleet
Stable model + stable shapes          Frequent model swaps
NVFP4 at full speed needed            FP8 is enough
Throughput is the SLA                 Latency / TTFT is the SLA
Static workload, predictable QPS      Dynamic workload, autoscaling
NIM contract / enterprise support     Open source ops
```

## The PyTorch flow (the modern path)

```python
from tensorrt_llm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    tensor_parallel_size=1,
    dtype="bfloat16",
    quant_config={"quant_algo": "FP8"},  # on Hopper+
)

outputs = llm.generate(
    prompts=["Explain paged KV cache in one paragraph."],
    sampling_params=SamplingParams(max_tokens=200, temperature=0.7),
)
```

`trtllm-serve` (CLI) starts an OpenAI-compatible server using the same machinery:

```bash
trtllm-serve Qwen/Qwen2.5-7B-Instruct --port 8002 \
    --tp_size 1 --kv_cache_free_gpu_memory_fraction 0.9
```

Same client code as vLLM and SGLang — that's the point of the OpenAI standard.

## The build-engine flow (read-only)

You should be able to recognize this when you see it in older repos:

```bash
trtllm-build \
    --checkpoint_dir ./qwen-converted \
    --output_dir ./engine \
    --gemm_plugin float16 \
    --max_batch_size 64 --max_input_len 4096 --max_output_len 1024 \
    --use_paged_context_fmha enable
```

The output is a *plan binary* tied to (GPU model, CUDA version, TRT version, shapes). Move it to a different GPU → rebuild. Bump TRT version → rebuild. This is the operational tax that drove the move to the PyTorch flow.

## Pitfalls

1. **Comparing default-flag vLLM to maximally-tuned TRT-LLM.** Spend equal tuning effort or document the asymmetry.
2. **Ignoring the build time.** A 30-minute engine build is part of the deployment pipeline. CI cycles get long.
3. **Pinning TRT version to the model.** A new TRT release can require a re-quantize + re-build. Document upgrade paths.
4. **Forgetting NVFP4 needs Blackwell.** On Hopper you get FP8; the 2× on top of FP8 from NVFP4 is Blackwell-only.
5. **Treating NIM as the engine.** NIM is a distribution. The engine inside is TRT-LLM (or vLLM, in some NIM containers since 2025).

## What to do this topic

1. On a Hopper GPU (rent one if needed): `pip install tensorrt-llm`. Document the install friction in your notes.
2. Run `tensorrt_llm.LLM(...)` on the same model as Topic 01.
3. Build with FP8 quantization. Time the build.
4. Hit `trtllm-serve` with the same harness from Topic 01. Compare TTFT and throughput against vLLM at the same concurrency.
5. Try long context (4K prompt). The TRT-LLM lead should be largest here.

## References

- TRT-LLM home — https://nvidia.github.io/TensorRT-LLM/
- TRT-LLM source — https://github.com/NVIDIA/TensorRT-LLM
- PyTorch flow examples — https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/llm-api
- NVFP4 deep dive (NVIDIA blog) — https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/
- NIM home — https://www.nvidia.com/en-us/ai-data-science/products/nim-microservices/
- FlashInfer (where TRT-LLM kernels increasingly live) — https://github.com/flashinfer-ai/flashinfer
