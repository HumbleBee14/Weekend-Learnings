# Level 5 — Learning Path

The engine bake-off level. Take the `mini-vllm` you finished in Level 4, line it up against vLLM, SGLang, TensorRT-LLM, llama.cpp, and MLC-LLM, and figure out which engine wins on which workload — with numbers. Then go beyond engines into the orchestration layer that's eating the field in 2026: disaggregated serving, NVIDIA Dynamo, llm-d, multi-LoRA, spec decode in production, VLMs.

```
The engines themselves          (01-06)   one topic per engine, plus internals
The bake-off                    (07)      Project 2 deliverable
Beyond a single engine          (08-09)   disaggregation + orchestration
What production actually does   (10-12)   multi-LoRA, offline batch, spec decode
The non-LLM runtime path        (13)      ORT and TensorRT (the runtime)
Multimodal                       (14)      VLM serving — half the 2026 surface
```

## The path

| Folder | Time | What you walk away with |
|---|---|---|
| `01-vllm-hello-world/` | 1-2h | vLLM serving the OpenAI-compatible endpoint; TTFT/ITL/throughput harness |
| `02-vllm-internals/` | 2-3h | Walked the V1 source: AsyncLLM, EngineCore, Scheduler, BlockPool, FlashInfer |
| `03-sglang-and-radixattention/` | 1-2h | Prefix-heavy workload pointed at both engines; the RadixAttention edge measured |
| `04-tensorrt-llm/` | 2-3h | TRT-LLM PyTorch flow, FP8 build, NIM context, the operational tax |
| `05-llama-cpp-deep-dive/` | 1-2h | GGML/GGUF, K-/i-/UD-/FP4 quants, when CPU beats GPU on $/Mtok |
| `06-mlc-llm/` | 1h | TVM Unity, WebGPU demo, when cross-platform is the right answer |
| `07-engine-bake-off/` | 4-6h | Project 2 — `reports/bakeoff.md` with G6-G9 + workload-conditional recs |
| `08-disaggregated-inference/` | 2-3h | Why prefill/decode split helps; KV transport (NIXL vs LMCache); a simulator |
| `09-dynamo-and-llmd/` | 2h | Dynamo and llm-d architectures mapped to Level 7 mini-platform topics |
| `10-multi-lora-serving/` | 2-3h | Two LoRAs hot-swapped on one base; throughput and switching costs measured |
| `11-offline-batch-inference/` | 1-2h | vLLM's offline mode; throughput + $/Mtok math vs online serving |
| `12-speculative-decoding-in-prod/` | 2-3h | n-gram / EAGLE-3 / P-EAGLE / MTP; acceptance rate as the metric; the quality check |
| `13-onnx-runtime-and-tensorrt/` | 2-3h | ORT + TRT (the runtime, not TRT-LLM); the small-model tier of a real stack |
| `14-vlm-serving/` | 2-3h | Vision encoder + projector + LLM; prefix-cache integration with image hashes |

Total: ~25-40 hours focused work. Front-loaded by the bake-off.

## What's new in 2026 (deltas vs 2024-2025 content)

The research backing this level surfaced several things that have changed status. If you've read older guides:

- **vLLM V1 is the default.** Chunked prefill on by default; CUDA graphs auto; spec decode first-class. Pre-2025 flags are no-ops or removed.
- **Disaggregated prefill/decode is standard, not novel.** "Almost every production-grade LLM serving framework — Dynamo, llm-d, Ray Serve LLM, SGLang, vLLM, LMCache, MoonCake — runs on disaggregation" (Hao AI Lab retrospective).
- **NVIDIA Dynamo 1.0 (2026)** is NVIDIA's "inference operating system for AI factories." Customer-facing as part of NIM.
- **llm-d** entered CNCF Sandbox (March 2026). K8s-native, Envoy AI Gateway-fronted, LMCache-backed.
- **Mooncake joined PyTorch Ecosystem** (Feb 2026). KV disaggregation is mainstream.
- **Block-hash kv-connector** is the cross-engine standard between vLLM and LMCache.
- **FlashInfer** is the kernel layer underneath vLLM, SGLang, and TRT-LLM. When someone says "vLLM's attention kernel," they probably mean FlashInfer.
- **TRT-LLM's lead has narrowed.** Still the throughput leader on Hopper/Blackwell with FP8/FP4, but vLLM and SGLang have closed most of the gap.
- **TRT-LLM's PyTorch flow** has largely replaced the older C++ engine-build flow for new development.
- **P-EAGLE (Feb 2026)** is the biggest spec-decode delta of the year. In vLLM v0.16+.
- **MTP heads** (DeepSeek-V3, Qwen3, Llama-4-Scout) are the high-end default — spec decode trained into the base.
- **Multi-LoRA at scale** is solved (Punica / S-LoRA kernels in vLLM); thousands of adapters per base is feasible.
- **VLMs are the default**, not the exception; vLLM, SGLang, TRT-LLM all have native support.
- **llama.cpp gained FP4** (May 2026) — local users now have NVFP4/MXFP4.

## What hardware you need

- **A real NVIDIA GPU** for Topics 01-04, 07, 08, 10-12. RunPod / Vast.ai / Lambda. Budget $30-50.
- **Hopper (H100) or newer** for Topic 04 FP8 measurements.
- **Blackwell (B200)** if you want NVFP4 numbers on TRT-LLM.
- **Apple Silicon Mac** for Topic 05 (llama.cpp Metal) and the M-series side of the bake-off.
- **Any machine with Python** for Topics 02 (reading), 06 (browser demo), 09 (architecture reading), 13 (ORT-CPU works fine).

## Each topic folder

Same shape as Levels 1-4:

- `CONCEPTS.md` — theory + 2026 state + ASCII diagrams + canonical references
- One or more code files (`.py`, `.yaml`) demonstrating the topic
- `README.md` — quickstart, expected output, things to try, where this goes

## Project 2 closes here

`engine-bakeoff/` repo with `runner.py`, per-engine configs, four workloads, and `reports/bakeoff.md`. The report is one of the two heaviest deliverables in the curriculum (the other is `mini-platform` from Level 7).

The required graphs:

- **G6** — TTFT bar chart per engine, short vs long prompts
- **G7** — throughput per engine on identical workload
- **G8** — GPU memory vs context length per engine
- **G9** — $/Mtok per engine + quant combination, including CPU-only llama.cpp

Workload-conditional recommendations are the deliverable. "vLLM is fastest" is not a finding; "for chatbot-style traffic with shared system prompts on Hopper, SGLang's TTFT p99 is 2× lower than vLLM's at concurrency 32" is.

## After this level

- Level 6 — distributed training. The other half of the field. Connects via the trained checkpoint that Level 7 will serve.
- Level 7 — `mini-platform`. The orchestration layer above engines: KV-aware router, autoscaler, observability, multi-tenant fairness. The toy version of Dynamo / llm-d.
- Level 8 — local-first agents on Apple Silicon. Project 4. Parallel track to the datacenter side.
