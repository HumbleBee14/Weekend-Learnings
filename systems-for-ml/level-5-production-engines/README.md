# Level 5 — Production Inference Engines

> Outer reference: [`systems-for-ml/README.md`](../README.md) · Project: **Project 2 — `engine-bakeoff`**
>
> Textbook companion (academic): [Reddi Vol 1 — *Serving*](https://mlsysbook.ai/) for the canonical serving taxonomy, plus [Reddi Vol 2 — *Distributed Inference*](https://mlsysbook.ai/) for Topics 08–09.
>
> Practitioner companion: [Kiely, *Inference Engineering*](../references/Inference-Engineering-Kiely-2025.pdf) **Ch 4** (Software) is the strongest fit in the entire book for this level — §4.3 covers vLLM / SGLang / TRT-LLM as a comparative table, §4.4 covers NVIDIA Dynamo, §4.5 covers benchmarking tooling and tips. **Ch 5 §5.5** covers disaggregation (Topics 08–09). **Ch 6** covers the modality topics (VLM Topic 14; embeddings/ASR/TTS/image-gen/video-gen are Kiely-only depth — not in this repo today). Read Ch 4 + Ch 6 before the bake-off.

## How to study this level

```
  Day 0 (15m)  ──►  Read this README — the week is Project 2 (engine-bakeoff)
  Day 1 (1.5h) ──►  Kiely Ch 4 (Software)  ── the comparative table on p106
                    is the mental model you'll defend at the end of the week.
                    + Kiely Ch 5 §5.5 for Topics 08-09 (disaggregation)
                    + Kiely Ch 6 for modality topics (13-14)
  Day 1 → 5    ──►  Topics 01 → 15, in order. For each topic:
                       1. Open the topic folder's  README.md
                       2. Read its  CONCEPTS.md
                       3. Stand up the engine, hit it with your Project 1 load harness
                       4. Capture numbers — they feed the bake-off report
  Day 5-7      ──►  Project 2 itself — Topic 07 (engine-bake-off) is the deliverable.
                    Ship reports/bakeoff.md as a systems-paper-style eval doc
                    with G6-G9 + a defended recommendation.
```

**Reference order when you get stuck:**
1. The topic's own `CONCEPTS.md`
2. Kiely Ch 4 (the practitioner comparison), Ch 5 §5.5 (disagg), Ch 6 (modalities)
3. [Reddi Vol 1 *Serving*](https://mlsysbook.ai/) + [Reddi Vol 2 *Distributed Inference*](https://mlsysbook.ai/)
4. **The engine source itself** — Level 5 is where reading vLLM/SGLang/TRT-LLM code becomes essential. Each topic's README links the right files.

**Compute:** RunPod / Lambda / Vast.ai bursts. Budget ~$30–50 for the week. A single L4 or A10 (≈$0.40/hr) gets you through most of it.

## Week goal

Drop your `mini-vllm` next to the real engines and benchmark them honestly. By Friday you should be able to:

- Stand up vLLM, SGLang, TensorRT-LLM, and llama.cpp from scratch and serve the same model through each.
- Drive identical workloads through all of them with your Project 1 load harness.
- Read the trace and explain *why* each engine wins or loses on each workload — not just "vLLM was faster."
- Pick the right engine for a given workload and defend the choice with numbers.
- Speak fluently about disaggregated prefill/decode, NVIDIA Dynamo, and llm-d — the 2026 production frontier.

This is where the inference-engineering field's center of gravity sits. The names you'll meet — vLLM, SGLang, TensorRT-LLM, llama.cpp — are what production teams actually run. Walking out of this week you should hold a real, defensible opinion about each.

## Where this fits

- **Comes after:** Level 4 (you have `mini-vllm`; you understand paged attention, continuous batching, and quantization).
- **Comes before:** Level 6 (distributed training), Level 7 (the platform that wraps these engines).
- **Project this feeds:** Closes **Project 2 (`engine-bakeoff`)** — ships `reports/bakeoff.md` with G6–G9.

## 2026 reality check — what changed

Several things have shifted enough that older guides are misleading:

- **vLLM V1 is the default.** Chunked prefill is on by default. Automatic prefix caching is on by default. CUDA graphs are auto-captured. Spec decoding is built-in. **Pre-2025 tutorials describe flags that are now no-ops or removed.**
- **Disaggregated prefill/decode is now standard, not novel.** "Almost every production-grade LLM serving framework — NVIDIA Dynamo, llm-d, Ray Serve LLM, SGLang, vLLM, LMCache, MoonCake — runs on disaggregation" (Hao AI Lab retrospective).
- **NVIDIA Dynamo (1.0 production, 2026)** is NVIDIA's "inference operating system for AI factories." Customer-facing as part of NIM.
- **llm-d** entered CNCF Sandbox in March 2026. Open-source disaggregated inference framework on Kubernetes. Backed by Red Hat / IBM / Google.
- **FlashInfer** is the kernel layer underneath vLLM, SGLang, and TRT-LLM. When someone says "vLLM's attention kernel," they probably mean FlashInfer.
- **TensorRT-LLM's lead has narrowed.** Still the throughput leader on Hopper/Blackwell with FP8/FP4, but vLLM and SGLang have closed most of the gap when `torch.compile` + CUDA graphs + FA3 are enabled.
- **MLC-LLM** is the cross-platform compiler-based engine — relevant when you need one model running across CUDA / Metal / Vulkan / WebGPU.

## Topic-by-topic deep dive

| # | Topic | What you build |
|---|-------|---------------|
| 01 | vllm-hello-world | Serve a model via vLLM, hit the OpenAI-compatible endpoint |
| 02 | vllm-internals | What's actually doing the work — V1 engine, scheduler, FlashInfer |
| 03 | sglang-and-radixattention | SGLang's prefix-tree KV reuse |
| 04 | tensorrt-llm | TRT-LLM PyTorch flow + FP8/FP4 |
| 05 | llama-cpp-deep-dive | GGML internals, Metal/CUDA/CPU backends |
| 06 | mlc-llm | The compiler-based, cross-platform engine |
| 07 | engine-bake-off | Same model + same load test across all five |
| 08 | disaggregated-inference | Prefill/decode separation, KV cache transfer |
| 09 | dynamo-and-llmd | NVIDIA Dynamo and llm-d — production frontier |
| 10 | multi-lora-serving | Hot-swap LoRA adapters |
| 11 | offline-batch-inference | vLLM offline mode for million-doc scoring |
| 12 | speculative-decoding-in-prod | EAGLE-3 in vLLM, end-to-end gain |
| 13 | onnx-runtime-and-tensorrt | The non-LLM-specific runtime path: ONNX Runtime + TensorRT (the runtime, not TRT-LLM). Where ORT/TRT win, where they don't, when you'd reach for them. |
| 14 | vlm-serving | Vision-language model inference: vision encoder + cross-attention + decoder. Different KV cache shape, different batching constraints, different prefill cost. Qwen2.5-VL / Pixtral / LLaVA. |
| 15 | triton-inference-server | The framework-agnostic outer wrapper that hosts vLLM, ORT, and TRT side-by-side. Ensemble graphs, dynamic batching for non-LLM models, the foundation of NVIDIA NIM. Every JD listing vLLM also lists Triton IS — this is why. |

### 01 — `vllm-hello-world`

**Build steps.**
1. `pip install vllm` (Linux + NVIDIA). Mac users — use Docker or a remote GPU.
2. `vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000`.
3. Hit it with the OpenAI client: `client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")`.
4. Send 50 concurrent requests via your Project 1 load harness. Capture TTFT, throughput, GPU memory.

**What to notice.** It just works. No batching code, no KV cache, no quantization knob to spelunk. That ergonomic gap — vs your `mini-vllm` — is the value vLLM provides.

### 02 — `vllm-internals`

**Read.** [Inside vLLM: anatomy of a high-throughput LLM inference system](https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html) (Sept 2025) — official deep dive. The single best document for understanding how a real serving engine is structured.

**What to extract.**
- **Engine V1** — the new architecture; AsyncLLM + EngineCore split, zero-overhead prefix caching.
- **Scheduler** — token-budget-based, mixes prefill and decode in same step.
- **Block manager** — paged KV cache (you already implemented one in Level 4).
- **Sampler** — handles all sampling logic (top-k, top-p, temperature, repetition penalty, structured output via xgrammar).
- **Worker** — the actual GPU process. Multiple workers for TP.
- **Model executor** — the forward pass. Uses FlashInfer for attention.

**Map your `mini-vllm` to vLLM.** What did you skip? What's hand-wavy? List three things vLLM does that yours doesn't. (Hint: chunked prefill, CUDA-graph capture, multi-LoRA.)

### 03 — `sglang-and-radixattention`

**What it is.** Same job as vLLM, different scheduler. SGLang's killer feature is **RadixAttention**: a radix tree keyed on token sequences for prefix-aware KV reuse. Beats vLLM's hash-based prefix caching on prefix-heavy workloads (RAG, multi-turn chat, agentic loops) — up to 6.4× on those, ~29% on generic H100 benchmarks.

**Build steps.**
1. `pip install sglang`. Serve same model on port 8001.
2. Run a chatbot-style workload: 100 conversations, each with the same 4KB system prompt + varied user turns.
3. Compare to vLLM on the same workload. SGLang should win meaningfully here.
4. Run a generic completions workload (no prefix overlap). They should be close.

**2026 reality.** SGLang and vLLM have converged significantly since 2024 — both have continuous batching, paged/chunked prefill, spec decoding, CUDA graphs, TP/PP/EP/DP, disaggregated serving. Remaining differentiators:
- SGLang wins on prefix-heavy and on structured generation (compiled-FSM grammars).
- vLLM wins on generic OpenAI-compatible serving and on Python router throughput (C++ router vs SGLang's GIL-bound Python).
- Production scale: SGLang powers xAI Grok 3 and Microsoft Azure's DeepSeek R1 on AMD; runs on 400k+ GPUs.

### 04 — `tensorrt-llm`

**What it is.** NVIDIA's high-performance inference library. Throughput leader on Hopper/Blackwell, especially with FP8 and NVFP4. Heavy upfront tuning (engine builds per shape/precision); lightweight runtime.

**2026 reality.** Reached 1.0 with formal 3-month deprecation policy. The Python-first PyTorch flow has largely replaced the older C++ engine-build workflow. NVIDIA pushes **NIM** (containerized microservices wrapping TRT-LLM) as the customer-facing product. TRT-LLM kernels increasingly contributed to FlashInfer.

**Build steps.**
1. `pip install tensorrt-llm` (Linux + NVIDIA + CUDA 12+). Heavy install.
2. Use the PyTorch flow: `from tensorrt_llm import LLM`. Same OpenAI-compatible serving idea.
3. Build the engine for your model with FP8 — this takes 5–30 minutes.
4. Serve, hit with same workload.

**Honest note.** If TRT-LLM is painful to install in your environment (it often is), document the friction in your bake-off. *Operational cost* is part of engine selection — production teams deal with this constantly.

### 05 — `llama-cpp-deep-dive`

**What it is.** GGML tensor library + GGUF model format + Metal/CUDA/CPU/Vulkan backends. The dominant local-inference runtime.

**Why it's in this week.** Don't dismiss it as "the consumer one." For batch=1 latency-sensitive workloads on small/mid models, llama.cpp on CUDA is competitive with vLLM. For CPU-only deployments at low QPS, it can beat a small GPU on $/Mtok. **Project 2's break-it list explicitly includes a llama.cpp-on-CPU vs small-GPU cost-per-token comparison.**

**Build steps.**
1. `brew install llama.cpp` (Mac) or build from source.
2. Convert your model to GGUF (or download a pre-quantized one).
3. `llama-server -m model.gguf --host 0.0.0.0 --port 8003`.
4. Same workload through it.

**What to notice.**
- TTFT on small models is excellent on Apple Silicon (Metal backend mature).
- Throughput on a single A10/L4 is comparable to vLLM for batch=1.
- Loses badly at high concurrency (continuous batching support exists but is less mature).
- i-quants (IQ4_XS) genuinely competitive with INT4 AWQ in quality.

### 06 — `mlc-llm`

**What it is.** A compile-once-run-anywhere LLM engine. Uses TVM Unity to lower model graphs to native code for CUDA, Metal, Vulkan, ROCm, WebGPU. Strong on cross-platform deployment.

**When to choose it.** When you need one model running on heterogeneous hardware (some CUDA, some Metal, some Android). Or when you specifically want WebGPU / browser inference. Otherwise vLLM/SGLang are usually a better fit on datacenter GPUs.

**Build steps (light touch).** Skim [mlc.ai docs](https://llm.mlc.ai/), run their pre-compiled WebGPU demo in your browser, write 50 words on when you'd reach for it.

### 07 — `engine-bake-off`

**This is Project 2.** Same model, same prompts, same hardware, same load harness, run against all five engines. Plus your `mini-vllm` from Level 4 as a sixth entry (you'll lose, gracefully).

**Required graphs (G6–G9 from the outer plan).**
- **G6** — TTFT bar chart per engine, split by short prompt (128 tok) vs long prompt (4K tok).
- **G7** — throughput (tokens/sec) per engine on identical workload.
- **G8** — GPU memory usage vs context length, per engine.
- **G9** — cost per million tokens per engine + quantization combination. *Include CPU-only llama.cpp as one of the rows.*

**Required experimental scenarios (from outer plan's break-it list):**
- Default flags vs tuned flags per engine.
- Long-context workload (32K prompts) — exposes KV cache strategy differences.
- Prefix-heavy workload (chatbot with shared 4KB system prompt) — exposes RadixAttention's edge.
- Constrained-memory scenario (smaller GPU than the model nominally needs).
- Cross-substrate cost run: llama.cpp on CPU vs same model on a small GPU.

**Output.** `reports/bakeoff.md` written as a short systems-paper eval doc:

```
1. Problem statement
2. Methodology (model, hardware, workload, metrics)
3. Results (G6–G9, plus per-scenario tables)
4. Per-engine notes (when to choose each)
5. Recommendation: "For workload X on hardware Y, use engine Z because..."
6. Operational notes (install pain, debuggability, observability)
```

This is the document an inference team writes before adopting an engine. It is one of the two strongest artifacts in the curriculum.

### 08 — `disaggregated-inference`

**What it is.** Splitting prefill (compute-bound, processes the prompt) and decode (memory-bound, generates tokens) onto different GPU pools. Why: prefill saturates compute, decode saturates memory bandwidth. Running them on the same GPU at the same time wastes one or the other.

**KV cache transfer.** When prefill finishes, the KV cache must move to a decode worker. Two strategies:
- **NIXL / NCCL transfer** — direct GPU-to-GPU over NVLink/IB. Low latency, ties prefill-decode placement.
- **LMCache / CMX (BlueField-4)** — KV cache as cluster-level storage. Decoupled placement, slightly higher latency.

The transfer mechanics — NIXL, RDMA, GPUDirect — are covered as systems primitives in **Level 6's NCCL + RDMA topic**. This topic is about *when* to use disaggregation and *what* you transfer; that one is about *how* the transfer actually moves bytes between GPUs across nodes.

**2026 reality.** Disaggregation is the production architecture for frontier-scale serving. NVIDIA Dynamo, llm-d, Ray Serve LLM, SGLang, vLLM all support it.

**Build steps (conceptual).** You won't run a real disaggregated cluster this week — too much infra. But you should:
1. Read [llm-d's prefill/decode disaggregation guide](https://llm-d.ai/docs/guide/Installation/pd-disaggregation).
2. Read [NVIDIA's Dynamo KV-cache-aware routing docs](https://docs.nvidia.com/dynamo/latest/user-guides/kv-cache-aware-routing).
3. Write 200 words: when does disaggregation help? When does it hurt? (Hint: high-QPS prefill-heavy workloads benefit most; low-QPS workloads pay overhead for nothing.)

### 09 — `dynamo-and-llmd`

**NVIDIA Dynamo.** "Inference operating system for AI factories." Production 1.0, 2026. Not just a server — it's the orchestration layer (placement, scaling, routing, KV cache) above engines like TRT-LLM and vLLM. NIM (NVIDIA Inference Microservices) wraps Dynamo for customer deployment.

**llm-d.** CNCF Sandbox (March 2026). Open-source equivalent on Kubernetes. KV-cache-aware routing, prefill/decode disaggregation, integrates with Envoy AI Gateway. Backers: Red Hat, IBM, Google.

**Why both matter for this curriculum.** They are the 2026 platform layer. Level 7's `mini-platform` will draw from llm-d's architecture (KV-cache-aware router, autoscaler driven by queue depth). You don't need to install either this week — but you must be able to explain what they are.

**Build steps.** Read the architecture docs. Sketch the components on paper. Identify what your Level 7 platform will mimic at a smaller scale.

### 10 — `multi-lora-serving`

**What it is.** Serving many fine-tuned variants of one base model from a single instance. The base model's weights are loaded once; each LoRA adapter is a small set of low-rank matrices loaded per request.

**Why it matters.** Every company fine-tunes adapters for customers/features. Naive deployment (one model server per LoRA) is unaffordable. Multi-LoRA serving makes it feasible — one base model serves dozens of LoRAs simultaneously.

**Build steps.**
1. Train two tiny LoRAs (e.g., one for code, one for poetry) on top of your base model.
2. Serve through vLLM with `--enable-lora --lora-modules code-lora=path1 poetry-lora=path2`.
3. Send requests specifying different LoRAs. Measure: throughput delta vs single-LoRA, memory cost per additional LoRA, switching latency.

### 11 — `offline-batch-inference`

**What it is.** vLLM's offline mode — `LLM(model=...).generate(prompts)` — for batch processing millions of documents (classification, summarization, scoring). Different optimization regime: throughput at all costs, latency irrelevant.

**Build steps.** Take 10K prompts. Run through `vllm.LLM().generate()` in batches. Measure tokens/sec and $/Mtok. Compare to running them one-by-one through the server.

### 12 — `speculative-decoding-in-prod`

**Build steps.**
1. Enable spec decoding in vLLM with EAGLE-3 weights (if available for your model) or n-gram (always available).
2. Same workloads as the bake-off, with spec decode on.
3. Measure: speedup, acceptance rate, quality (`lm-eval-harness` again).
4. Workload sensitivity: chat (high acceptance) vs code (varies) vs hard reasoning (low acceptance).

### 13 — `onnx-runtime-and-tensorrt`

**What this is.** Two production runtimes that exist outside the LLM-specific stack and that you'll meet in real codebases. They're not optimized for autoregressive generation the way vLLM is — but they're often the right choice for embeddings, reranking, vision, audio, or smaller transformer models served at scale.

**ONNX Runtime (ORT).** Microsoft's cross-framework runtime. Takes an ONNX-format model (an open exchange format) and runs it via a graph executor. Backends: CPU (oneDNN, AVX-512), CUDA, ROCm, CoreML (Apple), DirectML (Windows), WebGPU (browser), TensorRT (as a sub-execution-provider). Single binary, portable across platforms.

Where ORT wins:
- **Embedding / reranking models at scale** — BGE, jina-embeddings, ColBERT, BERT-variants. ORT's graph optimizer beats raw PyTorch on these by 2–3× and ships as a static binary.
- **CPU inference** — small classification / NER models on CPU clusters (still common in production).
- **Edge deployment** — Windows, Android, browsers (via WebGPU). PyTorch can't go there cleanly.
- **Cross-framework portability** — model trained in PyTorch or TF, exported once to ONNX, runs anywhere.

Where ORT loses:
- **Autoregressive LLM serving.** No paged KV, no continuous batching. You'd write that yourself on top.
- **Hopper-/Blackwell-specific kernels** — TRT-LLM and vLLM's tensor-core paths are usually faster for big LLMs.

**TensorRT (the runtime).** NVIDIA's inference compiler+runtime, NVIDIA-only. Takes a model (ONNX or via the TRT API), produces an optimized plan binary, runs it. Aggressive: kernel auto-tuning, layer fusion, INT8/FP8 calibration, kernel selection per shape. The plan is hardware-and-version-specific — you re-build per GPU.

Where TRT (not TRT-LLM) wins:
- **Vision and ASR models** — ResNet, ViT, DETR, Whisper, YOLO. TRT plans for vision are 2–5× faster than PyTorch eager and slightly faster than ORT.
- **Sub-billion-parameter transformers** with fixed shapes — fits TRT's compiled-plan model well.
- **Embedded/edge NVIDIA** (Jetson) — TRT is the standard runtime there.

Where TRT loses:
- **Dynamic-shape autoregressive workloads** — TRT prefers static shapes; LLM decode has dynamic sequence length. This is exactly why TRT-LLM exists — to handle the LLM-specific dynamics on top of TRT.
- **Fast iteration** — re-building plans takes minutes; bad for development loops.

**The decision tree.**
```
Is your workload autoregressive LLM serving?
  → Yes: vLLM / SGLang / TRT-LLM (Topics 01–04 of this level)
  → No: Is it transformer-shaped with fixed shapes?
        → Yes (vision/ASR/embedding): TensorRT or ORT-with-TRT-EP
        → No (general DL): ONNX Runtime
```

**Build steps.**
1. Take a small embedding model (e.g., `BAAI/bge-small-en-v1.5`). Export to ONNX via `torch.onnx.export` or `optimum`.
2. Run via ORT, ORT-with-CUDA-EP, ORT-with-TensorRT-EP. Measure throughput (embeddings/sec) at batch sizes 1, 8, 32, 128.
3. Compare to PyTorch eager and `torch.compile` paths.
4. Repeat with a small ViT (e.g., `google/vit-base-patch16-224`). The TRT win should be larger here.
5. Document: at what batch size does each runtime win?

**Honest framing.** This is *not* the path for serving your fine-tuned 70B model. It *is* the path for serving the embedding model that retrieves context for your 70B's RAG, or the safety classifier that filters its inputs, or the vision encoder feeding your VLM (Topic 14). A real production stack uses several runtimes, each for what it's best at.

**Resources.**
- ONNX Runtime — https://onnxruntime.ai/
- ORT execution providers — https://onnxruntime.ai/docs/execution-providers/
- TensorRT — https://docs.nvidia.com/deeplearning/tensorrt/
- HuggingFace Optimum (PyTorch → ONNX export pipeline) — https://huggingface.co/docs/optimum/

### 14 — `vlm-serving`

**What it is.** Vision-Language Models — models that take both image and text input. As of 2026 these are the default, not the exception: Qwen2.5-VL, Pixtral, LLaVA-OneVision, Gemini, GPT-4o, Claude 3.7 are all multimodal. Most "agentic workloads" need them (browse-the-web agents, screenshot analysis, document understanding).

**The architecture in one diagram.**

```
                ┌─────────────────────────────────────────────────────┐
                │   Vision encoder (ViT-style, often SigLIP / CLIP)   │
   image  ─────→│   patches the image, emits visual tokens            │
                │   ~256-576 tokens for a 336×336 image               │
                └─────────────────────────────────────────────────────┘
                                    ↓
                ┌─────────────────────────────────────────────────────┐
                │   Projector (small MLP or cross-attention)          │
                │   maps visual tokens into the LLM's embedding space │
                └─────────────────────────────────────────────────────┘
                                    ↓ visual tokens (now in LLM token space)
                ┌─────────────────────────────────────────────────────┐
                │   LLM decoder (regular causal transformer)          │
   text   ─────→│   prepends/interleaves visual tokens with text      │
                │   does autoregressive generation as usual           │
                └─────────────────────────────────────────────────────┘
                                    ↓ output tokens
```

**What's different from text-only LLM serving.**

1. **Two-stage prefill.** Vision encoder runs first (a separate small forward pass — heavy compute, no KV cache), then the LLM prefill consumes the projected visual tokens. The encoder pass is *not* batched the same way as LLM prefill — different model, different shape, different optimal batch size.

2. **Variable visual token counts.** Different image resolutions produce different numbers of visual tokens (Qwen2-VL goes from ~64 to ~16K visual tokens depending on image size). Padding waste in the visual prefill is *worse* than text padding waste because the variance is bigger.

3. **KV cache for visual tokens looks weird.** The LLM's KV cache contains entries for visual tokens that have no real "text" meaning. Prefix caching across requests has to consider whether the *same image* was sent — a hash of the image bytes is needed in addition to the text prefix hash.

4. **Memory pressure.** A 1080p image preserved at full resolution = ~16K visual tokens. Times batch size 8 = 128K tokens of KV cache from images alone — before any text. Long-context budgets blow up fast.

5. **Encoder-decoder placement decisions.** On a heterogeneous setup, the vision encoder might run on a smaller GPU (it's compute-light relative to the LLM). This is exactly the disaggregation pattern from Topic 08 generalized to VLMs.

**Production engines and VLM support (May 2026).**
- **vLLM**: native VLM support since v0.6 (mid-2024). Qwen2-VL, Qwen2.5-VL, Pixtral, LLaVA, MiniCPM-V, Phi-3-Vision all work via the standard `LLM(model=...)` API.
- **SGLang**: VLM support added in 2025; ahead on prefix-cache deduplication for repeated images.
- **TRT-LLM**: VLM support exists, but the multimodal pipeline is more setup-heavy.
- **vLLM V1 chunked prefill**: handles large visual prefills by splitting them across multiple decode steps — important when one image generates 16K tokens.

**Build steps.**
1. Serve Qwen2.5-VL-7B-Instruct (or smaller — Qwen2-VL-2B if memory tight) via vLLM. Send a request with one image + one text question.
2. Measure: vision-encoder time, LLM prefill time, decode time. Three separate phases — note their relative cost.
3. Send a batch of 8 requests with *the same* image, different text prompts. Measure with and without prefix caching enabled. The visual-token portion of the prefix should hit cache.
4. Send 8 requests with *different* images. Confirm no false-positive cache hits (would be a quality bug).
5. Stress test: 8 requests with very different image sizes (256×256 to 1024×1024). Watch the visual token count vary; observe padding waste.

**Honest framing.** Most "agentic workloads" in 2026 are multimodal. If you can serve text-only LLMs but not VLMs, you're missing half the production landscape.

**Resources.**
- vLLM multimodal docs — https://docs.vllm.ai/en/latest/usage/multimodal_inputs.html
- Qwen2.5-VL paper — https://arxiv.org/abs/2502.13923
- LLaVA family — https://llava-vl.github.io/
- SGLang multimodal — https://docs.sglang.ai/backend/multimodal.html

## Project 2 — close out this week

```
engine-bakeoff/
├── configs/
│   ├── vllm.yaml
│   ├── sglang.yaml
│   ├── trt-llm.yaml
│   ├── llama-cpp.json
│   └── mini-vllm.yaml          # your Level 4 entry
├── workloads/
│   ├── short-prompts.jsonl
│   ├── long-prompts.jsonl
│   ├── prefix-heavy.jsonl
│   └── memory-constrained.jsonl
├── runner.py                   # drives all engines uniformly
└── reports/
    └── bakeoff.md              # ← THE deliverable
```

`reports/bakeoff.md` is the artifact. It's one of the two heaviest deliverables in the curriculum (the other is `mini-platform` from Project 3).

## Definition of done

- [ ] You served the same model through vLLM, SGLang, TensorRT-LLM, llama.cpp on a uniform load harness.
- [ ] G6–G9 are in `reports/bakeoff.md` with Setup/Observation/Insight captions.
- [ ] You produced a written recommendation for at least three workload-hardware pairs.
- [ ] You can explain disaggregated prefill/decode, name Dynamo and llm-d, and articulate the 2026 production stack.
- [ ] You measured spec-decoding lift on at least two workload types and have the numbers.
- [ ] You demonstrated multi-LoRA serving with at least two adapters.

## Resources

- **vLLM V1 guide** — [docs.vllm.ai/v1_guide](https://docs.vllm.ai/en/stable/usage/v1_guide/).
- **vLLM anatomy blog** — [blog.vllm.ai/2025/09/05/anatomy-of-vllm.html](https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html). Required reading.
- **SGLang docs** — [docs.sglang.ai](https://docs.sglang.ai/).
- **TensorRT-LLM** — [github.com/NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM). Use the PyTorch flow.
- **llama.cpp** — [github.com/ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp).
- **MLC-LLM** — [llm.mlc.ai](https://llm.mlc.ai/).
- **NVIDIA Dynamo** — [nvidia-dynamo docs](https://docs.nvidia.com/dynamo/latest/).
- **llm-d** — [llm-d.ai](https://llm-d.ai/).
- **FlashInfer** — [github.com/flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer).
- **Disaggregated inference retrospective** — [haoailab.com/blogs/distserve-retro](https://haoailab.com/blogs/distserve-retro/).
- **vLLM Production Stack** — [docs.vllm.ai/projects/production-stack](https://docs.vllm.ai/projects/production-stack/en/latest/).

## Common pitfalls

1. **Comparing default-flag vLLM to maximally-tuned TRT-LLM.** Honest bake-offs use comparable tuning effort per engine, or document the asymmetry explicitly.
2. **Not measuring memory.** Two engines at the same throughput but with 2× memory difference are not equivalent — one fits more LoRAs / longer context.
3. **Skipping the prefix-heavy workload.** It's the scenario where SGLang's strongest feature shows up. Without it, your bake-off conclusion will be incomplete.
4. **Believing a single number.** Engine A wins on workload X, engine B wins on workload Y. The recommendation is workload-conditional.
5. **Treating disaggregation as exotic.** It's table stakes in 2026 production. You should be able to explain it, sketch the KV transfer step, and name at least one system that implements it (Dynamo, llm-d, vLLM, SGLang).
6. **Ignoring operational cost.** TRT-LLM's install pain is real. Document it. Engine ergonomics matter in production as much as peak throughput.

## What you'll be able to do after this week

> Build a reproducible benchmark harness comparing vLLM, SGLang, TensorRT-LLM, and llama.cpp on identical workloads (short/long/prefix-heavy/memory-constrained), characterizing TTFT, throughput, memory, and $/Mtok across engine + quantization combinations. Produce workload-conditional engine recommendations grounded in measured data. Demonstrate multi-LoRA serving and EAGLE-3 speculative decoding integration in vLLM.
