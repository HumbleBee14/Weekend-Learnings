# Level 8 — Learning Path

The local-and-on-device parallel track to Levels 5–7. Same systems thinking, Apple Silicon substrate. 16 topics organized into four sub-arcs:

```
Substrate                  (01-05)  unified memory, MLX, the framework bake-off, Metal
Apple-native paths         (06-07)  ANE / Core ML, Foundation Models framework
Serving and runtime levers (08-11)  serving stack, MoE, KV/spec, agentic loop
Training and shipping      (12-16)  QLoRA, DPO/GRPO, CPU paths, multi-Mac, privacy
```

## The path

| Folder | Time | What you walk away with |
|---|---|---|
| `01-unified-memory-mental-model/` | 1h | UMA mental model, why it changes the bandwidth ceiling |
| `02-mlx-basics/` | 2-3h | Lazy graph, autograd, the mlx-lm/-vlm/-whisper/-embeddings ecosystem |
| `03-mlx-vs-llama-cpp-vs-mps/` | 2-3h | Same model three ways; **G18** of Project 4 |
| `04-m5-neural-accelerators/` | 1-2h | Per-GPU-core matmul units; `mx.fast.matmul` path |
| `05-metal-shaders/` | 1-2h | Custom Metal kernel; the boundary you mostly do not cross |
| `06-ane-and-coreml/` | 2h | ANE / Core ML where it still wins (iPhone, SD/Whisper, Vision integrations) |
| `07-foundation-models-framework/` | 2-3h | Swift API, `@Generable`, custom adapter training for the on-device 3B |
| `08-local-serving-stack/` | 2-3h | Ollama (MLX) / LM Studio / vLLM-MLX / llama-server, OpenAI compat contract |
| `09-moe-on-mac/` | 2h | Llama 4 Scout, Qwen3-Next, DeepSeek V3.2 active-param math |
| `10-kv-cache-quant-and-spec-decode/` | 2-3h | 4-bit KV, QuantSpec, EAGLE-3 |
| `11-agentic-ide-backend/` | 3-4h | Sub-100 ms TTFT loop, two-model architecture, tool-calling |
| `12-qlora-on-device/` | 2-3h | `mlx_lm.lora`, DoRA, catastrophic-forgetting check; **G20** |
| `13-local-dpo-and-grpo/` | 2-3h | Preference learning without RLHF infra — DPO, ORPO, GRPO |
| `14-cpu-simd-and-sme2/` | 1-2h | AMX / SME2 / AVX-512+VNNI / NEON; when CPU competes |
| `15-distributed-mac-inference/` | 1-2h | `exo`, MLX distributed, Thunderbolt 5 mesh |
| `16-privacy-and-pcc/` | 1-2h | Three-posture model, PCC attestation, threat model |

Total: ~30-40 hours of focused work.

## What's new in 2026 (deltas vs 2024-2025 content)

The local field has matured fast. Items that have changed status in case you saw older material:

- **MLX is now Apple Silicon's dominant LLM framework** — Ollama added an MLX backend (March 2026 preview), vLLM-MLX delivers 400+ tok/s with continuous batching and paged KV. ~2–2.5× faster than llama.cpp Metal at matched quantization.
- **M5 Neural Accelerators** (October 2025+) — matmul units in every GPU core. MLX targets them via `mx.fast.matmul`; llama.cpp does not yet.
- **Foundation Models framework** matured throughout 2026 — Swift-native `LanguageModelSession`, `@Generable`, third-party adapter training.
- **MoE changed the local game** — Llama 4 Scout (17B-active / 109B-total), Qwen3-Next 80B-A3B, DeepSeek V3.2 fit on consumer Macs because only active params hit memory bandwidth.
- **KV cache quantization** (4-bit) is the single feature that made 100k-context inference viable on 64 GB Macs. Supported in MLX, llama.cpp, vLLM-MLX.
- **Apple QuantSpec** — self-speculative decoding with hierarchical 4-bit KV (Apple ML research; informs MLX's spec-decode direction).
- **`exo` and MLX distributed** — pipeline parallelism across multiple Macs over Thunderbolt 5 is real for 405B-class models.
- **CPU matrix extensions are the real story** — Intel AMX, Arm SME2 (Apple M4+). Pure SIMD-only LLM is a 2023 conversation.
- **Apple Private Cloud Compute** is the bridge for harder queries the on-device 3B can't satisfy. Hardware-rooted attested inference, no persistent storage, public binary transparency.
- **llama.cpp added FP4 (May 2026)** — local users now have NVFP4/MXFP4 paths.
- **Unsloth Dynamic v2.0 GGUFs** beat both K-quants and i-quants for size/quality (relevant when serving llama.cpp).

## What hardware you need

- **Apple Silicon Mac** for almost everything. M3/M4 fine for 7B-class. M5 helps for the Neural Accelerator topics. 64 GB RAM is the sweet spot; 128 GB unlocks 70B-class fine-tuning and Llama 4 Scout.
- **macOS 26+** required for Topic 07 (Foundation Models) and recent MLX features.
- **Xcode 26+** for the Swift sample in Topic 07.
- A **second Mac** with a Thunderbolt 5 cable is enough to play with Topic 15. Optional.

For learners on non-Apple hardware: most CONCEPTS.md content is portable (the math is the same). Code in 02–05, 09, 10, 12, 13 won't run; Topics 06, 14 (CPU-only path), and 16 (theory) are still reachable.

## Each topic folder

Same shape as Levels 1–7:

- `CONCEPTS.md` — theory + 2026 state + ASCII diagrams
- One or more code files (`.py` / `.swift` / `.sh`) demonstrating the topic
- `README.md` — quickstart, expected output, things to try, where the topic goes next

## Project 4 closes here

`local-agent` ships `reports/local.md` with three required graphs:

- **G18** — TTFT and tokens/sec: MLX vs llama.cpp Metal vs PyTorch MPS, same Mac, same model, same quant. Topics 02 / 03 / 08 feed this.
- **G19** — memory pressure curve as context grows on unified memory; mark where the system starts swapping. Topics 01 / 09 / 10 feed this.
- **G20** — quality before/after on-device QLoRA — task-specific accuracy vs general-benchmark drift (catastrophic-forgetting check). Topics 12 / 13 feed this.

The privacy section (Topic 16) closes the report. State explicitly which posture every model and tool is in.

```
local-agent/
├── agent/                       # Topic 11 backbone
├── models/
│   └── adapters/                # Topic 12 / 13 outputs
├── benchmarks/
│   ├── three-way.py             # G18 — Topic 03
│   ├── moe-vs-dense.py          # Topic 09
│   └── cloud-vs-local.py        # Topic 11
├── foundation-models-demo/      # Topic 07 (optional)
└── reports/
    └── local.md                 # the deliverable
```

## After this level

Level 9 is the compiler-and-kernels tour — the layer below MLX, llama.cpp, and the rest. The local stack you've built here will be the test substrate for some of those experiments. The privacy posture you established in Topic 16 is the one you defend the next time someone asks "is it really local?"
