# Level 8 — Local & On-Device Intelligence

> Outer reference: [`systems-for-ml/README.md`](../README.md) · Project: **Project 4 — `local-agent`**

## Week goal

Ship a local-first agentic system on Apple Silicon that's competitive with cloud APIs on small/medium models. By Friday you should be able to:

- Run the same model in MLX, llama.cpp (Metal), and PyTorch MPS on the same Mac, and explain when to choose each.
- Build an agentic loop with sub-100ms TTFT and zero per-token cost.
- Fine-tune a 7B model with QLoRA on your laptop, then DPO/ORPO it on preference pairs.
- Use Apple's Foundation Models framework (Swift) for the system 3B model and ship a custom adapter.
- Deploy a multi-model local stack (Ollama with MLX backend) competitive with what Continue.dev, Tabby, Zed users actually run.
- Understand the full local inference stack from kernel to serving layer — what tools like Ollama, LM Studio, Cursor, and Continue actually run under the hood.

This week is the parallel track to Levels 5–7 — same systems thinking, different substrate.

## Where this fits

- **Comes after:** Levels 1–7 (you understand serving, optimization, profiling, distributed training, platform).
- **Parallel to:** Level 7 — the platform you build there is cloud-shaped; the local-agent here is the on-device shape.
- **Comes before:** Level 9 (compiler tour).
- **Project this feeds:** **Project 4 (`local-agent`)** — ships `reports/local.md` with G18–G20.

## 2026 reality check — the local field has matured fast

- **MLX is now Apple Silicon's dominant LLM framework.** Ollama switched to MLX backend (March 2026 preview). vLLM-MLX delivers 400+ tok/s with continuous batching and paged KV. MLX is **2–2.5× faster than llama.cpp Metal** at matched quantization, with lower peak memory.
- **M5 Neural Accelerators** (October 2025+): matmul units embedded in every GPU core. M5 Max 40-core has 40 of them. Apple's claim: ~4× peak GPU compute for AI vs M4. MLX targets these directly via `mx.fast.matmul`.
- **Foundation Models framework** (WWDC25, matured throughout 2026): Swift-native API to the on-device ~3B model that powers Apple Intelligence. Third-party access. **Custom adapters trainable** with Apple's Python toolchain.
- **MoE changed the local game.** Llama 4 Scout (17B active / 109B total), Qwen3-Next 80B-A3B, DeepSeek V3.2 — only active params hit memory bandwidth, so they fit on consumer Macs. M5 Max runs Llama 4 Scout at 50+ tok/s.
- **KV cache quantization** (4-bit/8-bit) supported by MLX, llama.cpp, vLLM-MLX. The single feature that made 100K-context inference viable on 64GB Macs.
- **Speculative decoding mainstream.** Apple's QuantSpec (self-speculative decoding with hierarchical 4-bit KV) shipped into MLX. Typical 2–3× throughput at quality parity.
- **Distributed inference across Macs** — `exo` shards models across multiple Macs over Thunderbolt 5 mesh. Experimental but real for 405B-class models.
- **CPU SIMD landscape:** AVX-512+VNNI baseline, **Intel AMX** the big x86 story (LLaMA-3 3B INT8: ~57 tok/s with AMX vs ~28 without), **Arm SME2** is the ARM matrix extension Apple started shipping on M4+. Pure SIMD-only LLM is a 2023 conversation — in 2026 you want the matrix extensions.

## Topic-by-topic deep dive

| # | Topic | What you build |
|---|-------|---------------|
| 01 | unified-memory-mental-model | UMA, zero-copy tensors, why it changes the game |
| 02 | mlx-basics | Tensor ops, lazy eval, autograd on Metal |
| 03 | mlx-vs-llama-cpp-vs-mps | Same model three ways — measure throughput, memory |
| 04 | m5-neural-accelerators | Targeting M5's per-GPU-core matmul units |
| 05 | metal-shaders | Custom Metal kernel — the Apple equivalent of CUDA C++ |
| 06 | ane-and-coreml | Neural Engine path; when Core ML still wins |
| 07 | foundation-models-framework | Swift API + custom adapter training |
| 08 | local-serving-stack | Ollama (MLX backend), LM Studio, vLLM-MLX |
| 09 | moe-on-mac | Llama 4 Scout, Qwen3-Next, DeepSeek V3.2 active-param math |
| 10 | kv-cache-quant-and-spec-decode | 4-bit KV, QuantSpec, EAGLE-3 on MLX |
| 11 | agentic-ide-backend | Sub-100ms TTFT loop; canvas-first interaction |
| 12 | qlora-on-device | MLX-LM LoRA on 7B–70B |
| 13 | local-dpo-and-grpo | Preference learning without RLHF infrastructure |
| 14 | cpu-simd-and-sme2 | AVX-512 / AMX / SME2 / NEON — when CPU beats GPU |
| 15 | distributed-mac-inference | exo / Thunderbolt 5 mesh — 405B on three Macs |
| 16 | privacy-and-pcc | Threat model: on-device vs Private Cloud Compute vs cloud |

### 01 — `unified-memory-mental-model`

**What's different on Apple Silicon.** No host↔device boundary. CPU and GPU share the same physical DRAM. A tensor allocated by MLX is *literally the same memory* the CPU sees — no `cudaMemcpy`, no PCIe round-trips, no pinned-buffer dance.

**Why it matters for LLMs.** KV cache lives in shared RAM. Long contexts that would require host offloading on a discrete GPU (where bandwidth from DRAM to GPU is PCIe-limited) just work on M-series with the full 460–614 GB/s bandwidth.

**The catch.** Bandwidth is shared between CPU and GPU. If your CPU is busy tokenizing while the GPU runs decode, both contend for the same memory bus. Production code keeps tokenization off the hot path.

### 02 — `mlx-basics`

**MLX in one paragraph.** JAX-style lazy graph + NumPy-like tensor API + autograd, native to Metal. `mx.array` is the core type. Operations build a graph; `mx.eval(...)` materializes it. Lazy eval enables fusion automatically.

**Build steps.**
1. `pip install mlx mlx-lm`. Requires Apple Silicon and macOS 14+ (macOS 26.2+ for M5 Neural Accelerators).
2. Run `mlx_lm.generate --model mlx-community/Qwen2.5-7B-Instruct-4bit --prompt "hello"`.
3. Write a small script: load the model with `mlx_lm.load`, generate with `mlx_lm.generate`. Same shape as HuggingFace's API.
4. Inspect: `mx.array([1, 2, 3])` — see the lazy graph, then `mx.eval(...)` to materialize.

**The ecosystem.**
- `mlx-lm` — text gen, fine-tuning, GGUF/safetensors loaders, spec decoding, KV cache quant.
- `mlx-vlm` — Qwen-VL, LLaVA, InternVL, Llama 4 vision.
- `mlx-whisper` — speech.
- `mlx-embeddings` — local embeddings (made local RAG genuinely viable).
- `mlx-community` HuggingFace org — thousands of pre-quantized checkpoints.

### 03 — `mlx-vs-llama-cpp-vs-mps`

**The three frameworks, side by side.**

| | MLX | llama.cpp Metal | PyTorch MPS |
|---|---|---|---|
| Format | safetensors / MLX-native | GGUF | PyTorch / safetensors |
| Speed (7B 4-bit) | ~230 tok/s | ~150 tok/s | trails both significantly |
| Memory | lowest (zero-copy) | medium | highest |
| Day-one model support | days–weeks | hours | varies |
| Cross-platform | no (Apple only) | yes | yes (subset) |
| Training | yes (LoRA, full SFT, DPO) | inference-only | yes |
| Production use | growing fast | universal fallback | rare for LLM |

**Build steps.**
1. Same 7B model, 4-bit quantized, in all three.
2. Same prompts, same hardware. Measure TTFT, tokens/sec, peak memory.
3. **G18 of Project 4** — the three-way comparison.

**Insight to carry.** PyTorch MPS has not closed the gap on transformers. `torch.compile` on MPS is partial in 2026. For LLM work on Apple Silicon, MLX is the answer. PyTorch MPS is for non-LLM workloads or for when you need PyTorch ecosystem code unchanged.

### 04 — `m5-neural-accelerators`

**What they are.** M5 (October 2025+) embeds matmul Neural Accelerators in every GPU core. M5 base: 10 cores × accelerator. M5 Max 40-core: 40. Apple claims ~4× peak GPU compute for AI vs M4.

**How to target them.** MLX exposes them via `mx.fast.matmul` (and uses them automatically for transformer layers when supported). llama.cpp does not yet — that path is MLX-exclusive in 2026.

**Practical implication.** A 7B 4-bit on M5 Max should hit 300+ tok/s in MLX (vs ~230 on M3/M4 Max). If you're on M5 hardware, prefer MLX more strongly.

### 05 — `metal-shaders`

**What it is.** Metal Shading Language — Apple's GPU compute language. C++14 dialect. The closest equivalent to CUDA C++.

**When you'd write one.** Almost never for LLM serving — MLX's compiled kernels and `mx.fast` paths cover the standard ops. You'd reach for raw Metal for: novel attention variants, custom samplers, fused agentic-loop primitives. It's deep-specialization territory, but worth knowing the boundary exists.

**Build steps (light touch).** Read the [MLX custom Metal kernel guide](https://ml-explore.github.io/mlx/build/html/dev/extensions.html). Write a tiny elementwise kernel. Optional: read MLX's source for an attention implementation.

### 06 — `ane-and-coreml`

**Apple Neural Engine (ANE).** A fixed-function neural inference accelerator separate from the GPU. Excellent for fixed-shape models with low latency requirements; difficult to target for arbitrary LLM workloads (variable seq lengths, KV cache, attention masking).

**Where Core ML still wins in 2026.**
- iPhone deployment when ANE utilization matters more than raw GPU throughput (Core ML schedules across CPU+GPU+ANE; MLX is GPU-only).
- Stable Diffusion / SDXL / FLUX image generation on iPhone (`ml-stable-diffusion`).
- Whisper for system-level dictation.
- Anything integrating with Vision/Speech/CreateML frameworks.

**Where MLX wins.** LLM serving on Macs. Research workflows. Fine-tuning. Anything where you iterate.

**Rule of thumb.** Shipping in the App Store on iPhone → Core ML or Foundation Models. Mac dev/research/server → MLX. Cross-platform → llama.cpp.

### 07 — `foundation-models-framework`

**What it is.** Swift-native API to Apple's on-device 3B model (the same model behind Apple Intelligence). Quantization is mixed 2/4-bit, quantization-aware-trained from scratch. Zero per-token cost. Runs fully offline.

**API surface.**
- `LanguageModelSession` — stateful chat.
- `@Generable` macro — guided structured Swift output.
- `Tool` protocol — tool calling.
- `ResponseStream` — streaming.

**Custom adapters.** Apple ships a Python toolchain to train LoRA-style adapters. Ship the adapter alongside your app. Base model stays on-device, adapter specializes it. This is the path for app-specific fine-tunes.

**Build steps.**
1. On macOS 26+ with Xcode 26: create a SwiftUI project, import `FoundationModels`, instantiate a `LanguageModelSession`, send a prompt. ~10 lines.
2. Optional: use Apple's adapter training Python toolchain on a small custom dataset, embed the adapter in your app bundle.
3. Document languages supported (English, French, German, Italian, Spanish, Portuguese-BR, Chinese-simplified, Japanese, Korean) and the model's actual ceiling — it's a 3B, not a GPT-4 replacement.

**Honest framing.** The 3B model is for the writing/summarization/extraction/classification slice of an app. For coding agents and hard reasoning, fall back to a server model or run a bigger local model via Ollama/MLX.

### 08 — `local-serving-stack`

**The 2026 production-ranked stack.**

1. **vLLM-MLX** (`vllm-mlx`) — continuous batching + paged KV cache + MLX backend. 400+ tok/s. Best for serving real concurrent traffic from a Mac.
2. **Ollama (MLX backend)** — March 2026 preview. CLI-first daemon, OpenAI-compatible `/v1/chat/completions` on `:11434`. Multi-model concurrent serving with LRU eviction.
3. **LM Studio** — GUI-first, OpenAI-compatible server on `:1234`. Best UX for non-engineers and model exploration.
4. **`llama-server`** (llama.cpp's HTTP server) — universal fallback, cross-platform.

**Limitations to know.** Ollama and LM Studio do not have continuous batching like vLLM. Fine for a developer's machine or an internal tool; not for serving 100 concurrent users. If you need that, vLLM-MLX.

**Build steps.**
1. `brew install ollama && ollama pull qwen2.5:7b-instruct-q4_K_M`.
2. `ollama serve` with MLX backend enabled (set `OLLAMA_BACKEND=mlx` or use the MLX-aware build).
3. Hit it with the OpenAI client.
4. Run a multi-model workload: pull two models, send requests to both; observe LRU eviction.

### 09 — `moe-on-mac`

**The MoE math that changed local.** A standard 70B dense model needs ~140GB at fp16, ~35GB at 4-bit — barely fits on 64GB Macs and is bandwidth-bound. A MoE like Llama 4 Scout (109B total, 17B active per token) at 4-bit needs ~55GB total but only routes ~9GB of memory bandwidth per token. **Result:** larger total parameter count, faster inference.

**Models that matter on Mac in 2026.**
- **Llama 4 Scout** — 17B active / 109B total. Runs at 50+ tok/s on M5 Max.
- **Qwen3-Next 80B-A3B** — 3B active / 80B total. Tiny per-token bandwidth.
- **DeepSeek V3.2** (smaller variants) — sparse expert routing.

**Build steps.** Pull a MoE 4-bit through MLX. Compare its tok/s and active-memory-bandwidth to a dense model of similar quality. The MoE wins meaningfully.

### 10 — `kv-cache-quant-and-spec-decode`

**KV cache quantization.** Same idea as weight quantization, applied to KV cache. 4-bit KV roughly halves memory used by the cache. The single feature that made 100K-context inference viable on 64GB Macs.

**Apple QuantSpec.** Self-speculative decoding using a hierarchical 4-bit KV cache as the "draft." Shipped into MLX. Typical 2–3× throughput.

**EAGLE-3 on MLX.** Available where draft weights exist for the target model. Same idea as Level 4's spec decoding.

**Build steps.**
1. Enable 4-bit KV on a 7B model in MLX (`--kv-bits 4`).
2. Run a 100K-token prompt. Measure memory + TTFT vs 16-bit KV.
3. Enable QuantSpec or EAGLE-3 if available. Measure end-to-end speedup.

### 11 — `agentic-ide-backend`

**The pitch.** Cloud-API agents pay 200–500ms RTT per call and $0.01–0.10 per turn. A canvas-first agentic IDE with sub-100ms TTFT and zero per-token cost is structurally different.

**Build steps for `local-agent`.**
1. Local model: Qwen2.5-Coder 7B for autocomplete (fast, frequent), Qwen3-Coder 32B for chat (slower, less frequent). Both via Ollama-MLX.
2. Tool calling via JSON schema (`outlines` or vLLM-MLX structured output).
3. UI: even a simple TUI / web canvas works for the report. The point is the loop.
4. Multi-tool agent: file read, file edit, shell exec. Loop until task complete or step cap reached.

**Comparison artifact.** Run the same task in cloud-API mode (Claude / GPT-4) and local mode. Capture: TTFT, total wall time, $ cost (cloud), # tool calls, success rate. **Part of G18 / G19 / G20.**

### 12 — `qlora-on-device`

**The 2026 path.** `mlx_lm.lora --model <hf-id> --train --data <jsonl>`. Auto-detects 4-bit base → runs QLoRA (frozen quantized base, fp16 adapters).

**Practical numbers.**
- M3 Max 64GB: 7B QLoRA, batch 4, seq 2048, ~1500 tok/s training throughput.
- 13B QLoRA: comfortable on 64GB.
- 70B QLoRA: needs M5 Max 128GB and ~4-bit base.
- Mistral-7B SFT on 5k examples: ~90 minutes on M2 Max 32GB.

**Build steps.**
1. Pick a small dataset (your own writing, code style, or a public dataset).
2. `mlx_lm.lora --model mlx-community/Qwen2.5-7B-Instruct-4bit --train --data ./train.jsonl --iters 500`.
3. Save the adapter, fuse it into the base for serving.
4. Quality check: run before/after on general benchmarks (MMLU subset) — catastrophic forgetting check. **G20 of Project 4.**

### 13 — `local-dpo-and-grpo`

**Preference learning on Mac.** Practical in 2026 via `mlx-lm-lora` or `mlx-tune`.

**The trick that makes it fit.** Run reference model frozen and 4-bit while policy is fp16+LoRA. `mlx-lm-lora` does this by default.

**Workflow.**
1. SFT a 7B model (Step 12).
2. Generate paired completions on prompts.
3. Collect preferences (human-rated, or LLM-judge using a stronger model via API for "chosen" labeling).
4. Run DPO with `--beta 0.1`, lr 5e-7, 1–3 epochs.

**GRPO** (DeepSeek's group-relative variant) is the 2026 hotness — doesn't need a reference model at all, just a reward function. Major memory win on Mac.

**What doesn't work locally.** PPO with a separate value head + reward model loaded simultaneously at 13B+. Most local practitioners do SFT + DPO/ORPO and skip classical RLHF.

### 14 — `cpu-simd-and-sme2`

**The 2026 CPU LLM landscape.** Pure SIMD-only is a 2023 conversation. Matrix extensions are the path.

- **Intel AMX** (Sapphire Rapids+) — Advanced Matrix Extensions. LLaMA-3 3B INT8 hits ~57 tok/s with AMX vs ~28 without. Table stakes for CPU-only LLM serving on Xeon.
- **Arm SME / SME2** — Scalable Matrix Extension. Landed in mobile (2025 Android SoCs); Apple started shipping SME on M4+. llama.cpp added SME2 kernels in late 2025.
- **AVX-512 + VNNI + BF16** — AMD Zen 4/5 EPYC and Intel Sapphire Rapids+ baseline.
- **Apple AMX** (the CPU matmul coprocessor in earlier M-series, undocumented) — largely deprecated. Replacement is M5 Neural Accelerators (GPU side) and SME (CPU side).
- **NEON** — universal ARM SIMD baseline, llama.cpp's fallback.

**Build steps.**
1. Run llama.cpp on a small model with `-t 1` (single thread) — measure tok/s. Then `-t <num_perf_cores>`. Then with SME2 build flags. Each step should improve.
2. On Apple Silicon, `llama.cpp -ngl 0` forces CPU-only. Compare to `-ngl 99` (full GPU offload). Document the workload regimes where CPU competes (single request, low-latency, batch=1) vs where GPU dominates (any concurrency, long context).

### 15 — `distributed-mac-inference`

**`exo`** — distribute a single model across multiple Macs over Thunderbolt 5 mesh. Experimental but real for 405B-class models.

**Pattern.** Two M3 Max 64GB or three M3 Pro 36GB connected by Thunderbolt 5 (80 Gb/s) can collectively serve a 70B fp16 or 405B 4-bit model. Pipeline parallelism across Macs; each holds a layer range.

**Light touch this week.** If you have access to two Macs, try it — even 2-Mac inference is impressive. Otherwise, read the architecture and write 100 words on the topology.

### 16 — `privacy-and-pcc`

**What "on-device" buys you in 2026.**
- No data ever leaves the device. No Apple, no OpenAI, no logs, no training-data leakage, no subpoena risk.
- Offline operation.
- No per-token cost.
- Latency floor controlled by you.

**What it doesn't.** Protection from a compromised device, malicious app on the same device reading your prompts, screenshots / accessibility APIs.

**Apple Private Cloud Compute (PCC).** The bridge for harder queries. Foundation Models framework auto-routes to PCC when on-device isn't enough. PCC nodes: Apple Silicon servers with attested boot, no persistent storage, public binary transparency, Apple cannot access data.

**Threat model spectrum.**
- On-device = cryptographically strong privacy.
- PCC = strong-but-trust-Apple-attestation.
- Cloud LLM = trust-the-vendor-and-their-subprocessors.

**Output.** A short threat-model section in `reports/local.md` for `local-agent`. State explicitly what privacy your local stack actually provides and what it doesn't.

## Project 4 — close out this week

```
local-agent/
├── agent/
│   ├── loop.py                 # tool-calling loop
│   ├── tools/                  # file/edit/shell tools
│   └── prompts/
├── models/
│   ├── ollama-config.yaml      # local serving
│   └── adapters/               # QLoRA / DPO outputs
├── benchmarks/
│   ├── three-way.py            # MLX vs llama.cpp vs MPS
│   ├── moe-vs-dense.py
│   └── cloud-vs-local.py
├── foundation-models-demo/     # SwiftUI app (optional)
└── reports/
    └── local.md                # ← THE deliverable
```

**Required graphs (G18–G20).**
- **G18** — TTFT and tokens/sec: MLX vs llama.cpp Metal vs PyTorch MPS, on the same Mac, same model, same quant.
- **G19** — memory pressure curve as context grows on unified memory; mark where the system starts swapping.
- **G20** — quality before/after on-device QLoRA — task-specific accuracy vs general-benchmark drift (catastrophic forgetting check).

`reports/local.md` follows the systems-paper format. It's the Project 4 deliverable for this week.

## Definition of done

- [ ] You ran the same model in MLX, llama.cpp Metal, PyTorch MPS — have G18 with numbers.
- [ ] You stood up Ollama with MLX backend or vLLM-MLX as a real local serving layer.
- [ ] You built an agentic loop with measurable sub-100ms TTFT for autocomplete-shaped tasks.
- [ ] You QLoRA fine-tuned a 7B model on personal data and verified quality with `lm-eval-harness` subset (G20).
- [ ] You ran at least one DPO or GRPO step locally.
- [ ] You can articulate Apple Foundation Models framework — even if you didn't ship a Swift app, you know the API, the 3B ceiling, and the adapter story.
- [ ] You characterized the privacy threat model for your stack honestly.
- [ ] You have a written cloud-vs-local comparison: latency, cost, privacy, capability tradeoffs.

## Resources

- **MLX framework** — [github.com/ml-explore/mlx](https://github.com/ml-explore/mlx), [mlx-framework.org](https://mlx-framework.org/).
- **mlx-lm** — [github.com/ml-explore/mlx-examples](https://github.com/ml-explore/mlx-examples).
- **Apple — exploring LLMs with MLX on M5** — [machinelearning.apple.com/research/exploring-llms-mlx-m5](https://machinelearning.apple.com/research/exploring-llms-mlx-m5).
- **QuantSpec (Apple)** — [machinelearning.apple.com/research/quantspec](https://machinelearning.apple.com/research/quantspec).
- **Ollama MLX preview** — [ollama.com/blog/mlx](https://ollama.com/blog/mlx).
- **Foundation Models docs** — [developer.apple.com/documentation/FoundationModels](https://developer.apple.com/documentation/FoundationModels).
- **Foundation Models adapter training** — [developer.apple.com/apple-intelligence/foundation-models-adapter](https://developer.apple.com/apple-intelligence/foundation-models-adapter/).
- **vLLM-MLX** — [github.com/waybarrios/vllm-mlx](https://github.com/waybarrios/vllm-mlx).
- **mlx-tune (DPO/GRPO)** — [github.com/ARahim3/mlx-tune](https://github.com/ARahim3/mlx-tune).
- **mlx-lm-lora** — [github.com/Goekdeniz-Guelmez/mlx-lm-lora](https://github.com/Goekdeniz-Guelmez/mlx-lm-lora).
- **exo (distributed Mac inference)** — [github.com/exo-explore/exo](https://github.com/exo-explore/exo).
- **Apple Private Cloud Compute** — [security.apple.com/blog/private-cloud-compute](https://security.apple.com/blog/private-cloud-compute/).
- **CPU instruction sets for LLM inference** — [Cortensor docs](https://docs.cortensor.network/technical-architecture/ai-inference/cpu-instruction-sets-for-llm-inference-avx-amx-sme-vs-gpus).

## Common pitfalls

1. **Treating MPS as the Apple equivalent of CUDA.** PyTorch MPS is not the LLM path on Apple Silicon — MLX is. Most 2024 tutorials are wrong here.
2. **Assuming llama.cpp wins because it's "the local one."** In 2026, MLX beats it 50–90% on generation. Use llama.cpp for cross-platform; use MLX for Apple Silicon.
3. **Skipping KV cache quantization on long contexts.** Without 4-bit KV, 100K-context inference will fail on 64GB. With it, it just works.
4. **Treating Foundation Models as a GPT-4 replacement.** It's a 3B. Use it for the slice it's good at; fall back to a bigger model for hard tasks.
5. **Ignoring MoE.** Llama 4 Scout's 17B-active / 109B-total profile changes what fits on consumer Macs. Run one, see the math.
6. **No catastrophic-forgetting check on QLoRA.** A model fine-tuned on your personal style that now fails MMLU is a regression. Always run a small general-benchmark subset before/after.
7. **Confusing on-device with PCC.** Different threat models. Document which one your stack actually uses.

## What you'll be able to do after this week

> Build a local-first agentic system on Apple Silicon using MLX, llama.cpp Metal, and Ollama with sub-100ms TTFT and zero per-token cost. Benchmark MLX vs llama.cpp vs PyTorch MPS on identical workloads; characterize memory-bandwidth-vs-compute regimes on M-series unified memory. Fine-tune a 7B model with QLoRA and DPO entirely on-device and verify quality with `lm-eval-harness`. Demonstrate Apple Foundation Models framework with a custom adapter. Produce a privacy threat-model document distinguishing on-device, Private Cloud Compute, and cloud LLM postures.
