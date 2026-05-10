# 06 — MLC-LLM

## What it is

A compile-once-run-anywhere LLM engine, built on **TVM Unity** (now Apache TVM). Instead of dispatching to per-backend kernel libraries the way vLLM (FlashInfer/CUDA) and llama.cpp (per-backend kernel folders) do, MLC compiles the model graph itself down to native code per target: CUDA PTX, Metal, Vulkan, ROCm, WebGPU, even WASM.

The thesis: kernel libraries don't exist for every target you might want to ship to (browser? Android? AMD APU? Raspberry Pi?). A compiler-based approach generates kernels per target from one source.

## When you reach for it

```
You need... and...                                    Pick MLC-LLM?
─────────────────────────────────────────────         ─────────────
One model running on CUDA + Metal + Vulkan            Yes
WebGPU / browser inference                            Yes (the strongest fit)
Android / iOS deployment of an LLM                    Yes (or MLX on iOS)
Heterogeneous edge fleet (Jetson + Orin + AMD APU)    Yes
Datacenter NVIDIA serving                             No — vLLM/SGLang/TRT-LLM
Mac local serving                                     No — llama.cpp / MLX
Maximum throughput on Hopper                          No — TRT-LLM
```

For Project 2 (the bake-off), MLC isn't a primary entry. But you should be able to:
- explain what TVM Unity does in 60 seconds,
- run their pre-compiled WebGPU demo in your browser,
- name the workload regime where MLC is the right answer.

## The compile pipeline

```
PyTorch / HF model
        │
        │  weight conversion (dtype + quant: q4f16_1, q4f32_1, q3f16_1, etc.)
        ▼
TVM Unity Relax IR (model graph)
        │
        │  schedule: tile sizes, layout, fusion, target-specific ops
        ▼
target-specific lowered IR
        │
        │  codegen
        ▼
.so / .dylib / .dll / .wasm / WebGPU shader
        │
        │  packaged with the GGUF-like weight file
        ▼
runtime loads the artifact + weights, exposes a chat API
```

The output is a `mlc-chat-config.json` + a model lib + the quantized weights. One artifact per target. Production deployments precompile for all targets they ship to.

## What it costs vs the alternatives

```
                MLC-LLM             llama.cpp           vLLM
                ───────             ─────────           ─────
build time      minutes-to-hours    instant (download)  instant
artifact        per-target .so      one .gguf           Python
runtime size    small               small               medium
kernel quality  decent              good (hand-tuned)   excellent (FlashInfer)
throughput      mid                 mid (CPU/Metal)     high (CUDA/H100+)
portability     extreme             high                low (CUDA-only mostly)
ecosystem       small               huge                huge
```

For the cross-platform problem MLC is solving, "decent" kernel quality is enough. For datacenter throughput, it isn't.

## WebGPU — the differentiator

The WebGPU build is genuinely useful. A 7B Q4 runs in Chrome / Edge / Safari (Tech Preview) at usable speeds on a discrete GPU laptop. Use cases:

- **Privacy-critical demos.** "Run this model in your browser; we never see your data."
- **Trial deployments.** Ship a quantized assistant as a static page.
- **Games / interactive media.** LLMs in unity/unreal-style web exports.

The browser-tab live demo is the fastest way to internalize what MLC enables: https://chat.webllm.ai/

## What to do this topic (light)

This is a 1-2 hour topic. The point is awareness:

1. Open https://chat.webllm.ai/ in Chrome. Send a few prompts. Inspect the dev tools (network, GPU usage) — confirm everything's local.
2. Skim the MLC docs for their compile pipeline (`compile_and_run.py` reproduces the smallest possible end-to-end).
3. Write 50-100 words for your notes: when you'd reach for MLC. (Hint: cross-platform shipping, browser deployment, edge fleets.)

## Pitfalls

1. **Treating it as a vLLM competitor.** Different problem. Datacenter NVIDIA serving is not its lane.
2. **Underestimating compile time.** Custom models need full re-compilation per target. CI cycles get long.
3. **Picking it for Mac.** llama.cpp's Metal backend or MLX is usually a better fit on Mac specifically.
4. **Ignoring the ecosystem gap.** Bug surface, model availability, community velocity — MLC is much smaller than vLLM/llama.cpp.

## References

- MLC-LLM home — https://llm.mlc.ai/
- MLC-LLM source — https://github.com/mlc-ai/mlc-llm
- WebLLM browser demo — https://chat.webllm.ai/
- TVM Unity / Apache TVM — https://tvm.apache.org/
- WebGPU spec — https://www.w3.org/TR/webgpu/
- MLC on Android / iOS — https://llm.mlc.ai/docs/deploy/android.html
