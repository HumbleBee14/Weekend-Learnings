# Level 4 — LLM Optimization Techniques

> Outer reference: [`systems-for-ml/README.md`](../README.md) · Project: closes **Project 1 (`mini-vllm`)**

## Week goal

Take `mini-serve` from Level 1, profiled in Level 3, and turn it into `mini-vllm` — a server with a paged KV cache, continuous batching, and at least one quantization mode, all justified by measurements. By Friday you should be able to:

- Implement a paged KV cache manager from scratch (pages, block table, free list, eviction). This is *the* core data structure inside vLLM/SGLang.
- Quantize a model to FP8 or INT4 and prove (with `lm-eval-harness`) that you didn't break it.
- Explain what `torch.compile` actually does for inference, and where it helps vs hurts.
- Speak fluently about FP8 / NVFP4 / MXFP4 / GGUF i-quants — the quantization vocabulary the field actually uses in 2026.
- Implement and benchmark continuous batching against your Level 1 naive batcher.

This is the densest week in the curriculum. Take the full week.

## Where this fits

- **Comes after:** Level 1 (server), Level 2 (kernels), Level 3 (profiling — the *justification* for everything you fix here).
- **Comes before:** Level 5 (engine bake-off — your `mini-vllm` is one of the bake-off entries; it should hold its own on small workloads).
- **Project this feeds:** Closes **Project 1**. Ships `reports/project1.md` with G1–G5.

## 2026 reality check

Some terminology has shifted since the textbook era:

- **Continuous batching is no longer a special technique** — it's table stakes. Every engine has it.
- **Paged attention is the canonical KV cache layout** in 2026. The field has converged on it.
- **Chunked prefill is default-on in vLLM V1.** Don't write tutorials describing it as a flag to enable.
- **FP8 is mainstream**, not exotic. Hopper (H100) and Blackwell (B100/B200) both support it natively.
- **NVFP4 (E2M1)** and **MXFP4** are the next quantization frontier — Blackwell-class hardware feature.
- **`llm-compressor`** (vllm-project) is the canonical PTQ toolkit — replaces ad-hoc AWQ/GPTQ scripts.
- **i-quants** (IQ4_XS, IQ3_XXS, IQ2_M) beat K-quants at the same bitrate and are mainstream for sub-4-bit GGUF.
- **BitNet 1.58** is real but research-bound: ternary weights require training from scratch, no post-hoc conversion. Microsoft's BitNet b1.58 2B-4T is the only credible open base.

The curriculum below reflects this — older guides will recommend things that are now defaults or have been superseded.

## Topic-by-topic deep dive

| # | Topic | What you build |
|---|-------|---------------|
| 01 | quantization-basics | INT8 / FP16 / BF16 inference baseline |
| 02 | fp8-and-nvfp4 | FP8 (E4M3/E5M2) and NVFP4/MXFP4 — the 2026 datacenter precisions |
| 03 | weight-only-ptq | AWQ, GPTQ, SmoothQuant via `llm-compressor` |
| 04 | local-quant-formats | GGUF K-quants vs i-quants; EXL2 |
| 05 | extreme-quantization | 3-bit / 2-bit / BitNet 1.58 — frame as research, not deployment |
| 06 | quality-evaluation | `lm-eval-harness` — every quant tested, never "feels fine" |
| 07 | torch-compile | Dynamo + Inductor; gotchas; when it hurts |
| 08 | kernel-fusion | Why fusion eliminates HBM round-trips |
| 09 | kv-cache-naive | Implement a contiguous KV cache, feel fragmentation |
| 10 | kv-cache-paged | Paged KV cache: pages, block table, free list (mini-vLLM) |
| 11 | kv-cache-eviction | LRU vs sliding-window vs prefix-sharing |
| 12 | long-context-stress | 100K-token workloads — naive collapses, paged holds |
| 13 | speculative-decoding | EAGLE-3, n-gram, draft model — acceptance rate matters |
| 14 | continuous-batching | Multiple users, mixed lengths, no padding waste |
| 15 | structured-output | Outlines / xgrammar / JSON schema — constrained decoding |
| 16 | serving-concurrency | Sharded locks for the KV block manager, lock-free queues for admission, cancellation propagation, stream multiplexing — the concurrency patterns vLLM and SGLang actually use |
| 17 | spec-decode-systems | Speculative decoding at the systems level: how spec interacts with continuous batching scheduling, verification/rollback semantics, tree-spec (EAGLE-3) acceptance handling, multi-model coordination |

### 01 — `quantization-basics`

**The hierarchy in 2026.**
- **FP32** — reference precision; rarely used for inference, occasionally for sensitive ops (LayerNorm).
- **FP16 / BF16** — default mixed-precision. BF16 has wider exponent range, dominates for training.
- **FP8 (E4M3 / E5M2)** — Hopper+ hardware-native. E4M3 for forward, E5M2 for gradients.
- **NVFP4 (E2M1) / MXFP4** — Blackwell-native. 4-bit float with adaptive block scaling.
- **INT8 / INT4** — integer quantization, dominant on consumer/non-NVIDIA hardware.

Run a baseline: same model in FP32, FP16, BF16. Measure throughput and `lm-eval-harness` score. Establish the reference.

### 02 — `fp8-and-nvfp4`

**Why it matters.** FP8 is what the field actually deploys on Hopper-class hardware in 2026. Skipping it leaves you reasoning about a precision regime that's no longer the production default.

**FP8 formats.**
- **E4M3** — 4-bit exponent, 3-bit mantissa. Range ~±448. Used for forward pass, weights, activations.
- **E5M2** — 5-bit exponent, 2-bit mantissa. Range ~±57344. Used for gradients (wider dynamic range).

**NVFP4 / MXFP4.** 4-bit floats grouped into blocks of 16 or 32 elements, each block sharing a scaling factor stored as E8M0 (a power-of-two bit-shift — hardware-friendly). NVFP4 is NVIDIA's E2M1 variant with block size 16; MXFP4 is the OCP standard with block size 32. Hardware: NVFP4 on Blackwell, MXFP4 on Blackwell + AMD MI355. Why block scaling? Per-tensor FP4 loses too much accuracy because one outlier blows the global scale. A per-block scale fixes each group of 16–32 elements independently — same hardware cost, much better accuracy. The Blackwell B200 delivers ~18,000 sparse FP4 TFLOPS vs ~9,000 sparse FP8 — the 2× hardware throughput ratio is why FP4 matters.

**Two tools for MX quantization in 2026:**
- **`llm-compressor`** (`vllm-project/llm-compressor`, v0.9.0+): supports W8A8 (FP8), MXFP8, MXFP4, NVFP4, and mixed precision. The standard path for quantizing models for vLLM deployment.
- **TorchAO** (`pytorch/ao`): PyTorch-native tensor subclass abstractions for MXFP4/MXFP6/MXFP8. Better for research/experimentation; integrates cleanly with `torch.compile`.

**Build steps (FP8 minimum, NVFP4 optional if no Blackwell access).**
1. Install `llm-compressor`. Use the FP8 dynamic-quantization recipe on a 1–7B model.
2. Serve through vLLM (which supports FP8 KV cache + FP8 weights natively).
3. Compare to FP16 baseline: throughput, memory, `lm-eval-harness` MMLU score.
4. (Optional / Blackwell only) Run the NVFP4 recipe in `llm-compressor`; benchmark dequantization overhead with Nsight.

### 03 — `weight-only-ptq`

**AWQ vs GPTQ vs SmoothQuant.** All three are post-training quantization (PTQ) methods that quantize weights only (activations stay in higher precision).
- **AWQ** (Activation-aware Weight Quantization) — protects salient weight channels based on activation magnitudes. Generally best quality at INT4 for chat models.
- **GPTQ** — Hessian-based, layer-by-layer second-order method. Slightly worse than AWQ on most models, comparable on some.
- **SmoothQuant** — shifts quantization difficulty from activations to weights via a per-channel scale. Useful for W8A8 (both weights and activations 8-bit).

In 2026 these are all available through **`llm-compressor`**. Don't write your own AWQ implementation — this isn't a research week. Use the toolkit, measure the quality cost.

**Build steps.**
1. Quantize a 7B model to INT4 with AWQ via `llm-compressor`.
2. Serve through vLLM.
3. Measure: throughput uplift, memory reduction, MMLU/HumanEval delta.

### 04 — `local-quant-formats`

**GGUF.** Single-file format used by llama.cpp / Ollama / LM Studio. Bundles weights + tokenizer + metadata. The format the local-AI world standardized on.

**K-quants vs i-quants in 2026.**
- **K-quants** (Q4_K_M, Q5_K_M, etc.) — block-wise mixed-precision. Q4_K_M is the long-time default sweet spot.
- **i-quants** (IQ4_XS, IQ3_XXS, IQ2_M) — importance-matrix-trained non-linear codebooks. **Now beat K-quants at the same bitrate**. Mainstream for sub-4-bit since mid-2024.

**Recommendation for 2026:** for 4-bit, IQ4_XS is the right default. For 3-bit, IQ3_M. K-quants only when an i-quant variant doesn't exist for your model.

**EXL2** — ExLlamaV2's format. Mixed-bit-rate within a model (different layers at different precisions). Niche but excellent for consumer GPUs.

**Build steps.** Take your INT4 AWQ model from Step 03, also quantize to GGUF IQ4_XS via `llama-quantize`. Run both through `lm-eval-harness`. The numbers will be close — that's the point. The GGUF version runs on your laptop with no GPU.

### 05 — `extreme-quantization`

**Frame:** this is the research frontier, not the deployment baseline. Two distinct ideas:

1. **3-bit / 2-bit GGUF i-quants** — usable today for very memory-constrained scenarios (running 70B on a 24GB GPU). Quality drops noticeably but stays useful for many tasks.
2. **BitNet b1.58** — ternary weights {-1, 0, +1}. Trained from scratch (cannot convert post-hoc). Microsoft released BitNet b1.58 2B-4T, open-weights. `bitnet.cpp` runs it with 1.4–6× speedup and 55–80% energy reduction on CPU. **No frontier-scale models exist yet** — quality at 2B is the proof-of-concept ceiling.

**Why this matters for the curriculum:** know it exists, know why FP8/FP4 ate its mindshare for production (post-hoc convertible from FP16, hardware-native on current GPUs). 2026 datacenter inference runs on FP8/FP4. CPU inference and curiosity runs on BitNet.

### 06 — `quality-evaluation`

**This is non-negotiable.** Every quantization, every `torch.compile`, every speculative-decoding change in this curriculum ends with an `lm-eval-harness` run. Without it, "this is faster" is an unfounded claim.

**Build steps.**
1. `pip install lm-eval`. Pick three benchmarks: MMLU (knowledge), HumanEval (code), GSM8K (reasoning).
2. Run on your FP16 baseline. Record the numbers.
3. Run on each quantized version. Compute the deltas.
4. Anything > 1% absolute drop on MMLU is a real regression, document it.

**Output for Project 1.** A table in `reports/project1.md`:

| Variant | Throughput (tok/s) | Memory (GB) | MMLU | HumanEval | Quality cost |
|---------|-------------------|-------------|------|-----------|--------------|
| FP16 baseline | … | … | … | … | 0 |
| FP8 dynamic | … | … | … | … | … |
| INT4 AWQ | … | … | … | … | … |
| GGUF IQ4_XS | … | … | … | … | … |

### 07 — `torch-compile`

**What it does.** Captures your `nn.Module`'s forward pass into a graph (Dynamo), lowers it through Inductor to fused Triton kernels and CUDA graphs. Eliminates Python overhead, fuses elementwise ops, reduces kernel launches.

**2026 reality.**
- For LLM inference: ~35–40% TTFT improvement, 25–30% throughput improvement on top of eager.
- It's the kernel-fusion + CUDA-graph layer underneath modern vLLM and SGLang — not a separate manual optimization for serving.
- **Common gotchas:**
  - Graph breaks (in-place ops on views, dynamic Python control flow, unannotated custom ops) silently fall back to eager. **Always check** `TORCH_LOGS=graph_breaks`.
  - Cold-start compile time is 10s–minutes. Painful for autoscaling. Use the on-disk compile cache.
  - Do NOT compose with TensorRT-LLM (TRT runs its own compiler).
  - MPS (Apple) support is partial in 2026 — don't rely on it; use MLX on Apple.
  - Highly dynamic shapes can hurt or be neutral.

**Build steps.**
1. Wrap your model: `model = torch.compile(model)`.
2. First request will be slow (compilation). Discard it.
3. Run your `mini-serve` benchmark again. Compare.
4. Run with `TORCH_LOGS="graph_breaks,recompiles"`. If you see graph breaks, find and fix at least one.

### 08 — `kernel-fusion`

**What it is.** Combining multiple operators into a single kernel so intermediate tensors stay in registers/SRAM instead of round-tripping to HBM. Example: `softmax(QK^T) @ V` as three kernels writes the N×N attention matrix to HBM twice. Fused as one kernel (FlashAttention) it never materializes.

You don't write fused kernels by hand this week — `torch.compile` and FlashAttention do it for you. But you need to recognize the pattern in profiles. After running your compiled model in Step 07, open the trace: you should see *fewer, larger* kernels. If you see the same number of kernels as eager, your compile didn't actually fuse.

### 09 — `kv-cache-naive`

**What it is.** Allocate a single contiguous tensor of shape `(batch, max_seq_len, num_heads, head_dim)` per layer. Each request gets a fixed slice. Prefill writes the prompt's K and V; decode appends one row per step.

**Why it's a strawman.** Three failure modes:
1. **Internal fragmentation** — every request reserves `max_seq_len` even if it'll only use 100 tokens. 80%+ of KV memory is wasted.
2. **External fragmentation** — when a request finishes and frees its slice, the gap may be too small for the next request.
3. **No prefix sharing** — two requests with the same system prompt store the prefix's KV twice.

You're going to feel each of these. That pain is the motivation for paged KV cache.

**Build steps.**
1. Implement a `KVCache` class that allocates one contiguous tensor per layer.
2. `allocate(request_id, max_len) -> slice`, `append(request_id, k, v)`, `free(request_id)`.
3. Drop into `mini-serve`. Run a workload with mixed lengths (50–4000 tokens). Log GPU memory used.
4. Plot memory utilization over time. You'll see ~30% actually used, 70% reserved.

### 10 — `kv-cache-paged`

**What it is.** PagedAttention. Memory is divided into fixed-size pages (e.g., 16 tokens of KV each). A *block table* per request maps logical token positions to physical page indices. Pages are allocated on demand from a *free list*. New request gets one page, asks for more as it grows.

**Why it works.**
- No internal fragmentation — pages are small (16 tokens), waste is bounded by page size.
- Easy prefix sharing — same physical pages can be referenced by multiple requests' block tables.
- Dynamic growth — request grows by allocating one more page, not by being moved.

**Build steps.**
1. `class PagedKVCache`:
   - `pages: Tensor` of shape `(num_pages, page_size, num_heads, head_dim)` — physical storage.
   - `free_list: list[int]` — page indices available.
   - `block_tables: dict[req_id, list[int]]` — logical-to-physical map per request.
2. `allocate(req_id) -> page_idx`, `append(req_id, k, v)` (allocates new page if current is full), `free(req_id)` (returns pages to free list).
3. Implement attention against this layout. **Hint:** for a clean implementation, gather pages into a contiguous tensor before attention; for a fast implementation, use a kernel that handles paged layout (FlashInfer does this — you can read its API). For Level 4 the gather version is fine.
4. Drop into `mini-serve`. Run the same mixed-length workload. Memory utilization should be 90%+ now.

**Reference reading.** vLLM's PagedAttention paper (Kwon et al., SOSP 2023). Read Sections 3 and 4. The block-table concept is the entire idea.

### 11 — `kv-cache-eviction`

**Why eviction matters.** Memory is finite. When all pages are allocated and a new request arrives, you must either (a) preempt an existing request, (b) evict pages from a paused request, (c) refuse the request. Different strategies for different workloads.

**Strategies.**
- **LRU on requests** — evict the least-recently-active request entirely. Simple, fair-ish.
- **Sliding window** — keep only the last N tokens of KV per request. Useful for very long contexts where old tokens contribute less.
- **Prefix-sharing aware (RadixAttention-style)** — when shared-prefix pages are referenced by multiple requests, evicting the prefix only when *no* request needs it. SGLang's approach.

**Build steps.**
1. Implement LRU and sliding-window eviction in your paged cache.
2. Workload A: chatbot (high prefix overlap). Workload B: long single-turn generations.
3. Measure cache hit rate, p99 latency, throughput for each strategy on each workload.
4. **G4 and G5** of Project 1 come from this step.

### 12 — `long-context-stress`

**The stress test.** Run a 100K-token prompt through `mini-vllm`. The naive cache from Step 09 will OOM or fragment. The paged cache from Step 10 will handle it. **G3** of Project 1 is the resulting context-length-vs-TTFT curve.

### 13 — `speculative-decoding`

**What it is.** A small "draft" model proposes K tokens per step. The big "target" model verifies all K in parallel (one forward pass instead of K). For each accepted prefix, you generate K tokens in one decode step.

**2026 reality.**
- **EAGLE-3** is the SOTA draft method (NeurIPS 2025). Fuses low/mid/high-level features. 2–6× speedup on chat workloads, integrated into vLLM V1, SGLang, TensorRT-LLM.
- **EAGLE-2** with dynamic tree verification — common where EAGLE-3 weights aren't available.
- **Medusa** (multiple decoding heads) — older, simpler (~2×). Falling out of favor.
- **n-gram / prompt-lookup decoding** — zero training, useful for code and long-context tasks with high token repetition.
- **Acceptance rate matters more than speedup**. Chat: 60–80%. Code: higher. Hard reasoning: lower. Workload-dependent.

**Build steps (light touch this week).**
1. Use vLLM's spec-decode flag with EAGLE-3 weights for a model that has them, or n-gram for any model.
2. Measure speedup on chat workload vs reasoning workload.
3. Write up: "for workload X, spec decode delivered Y× speedup with Z% acceptance rate."

You can also implement a tiny n-gram speculator yourself for understanding — it's ~50 lines.

### 14 — `continuous-batching`

**What it is.** The scheduler from your batcher in Level 1, but smarter. Instead of waiting for a batch to fill or finishing all sequences before starting new ones, the scheduler runs a forward pass each iteration on whatever is currently active. New requests slot in immediately; finished requests free their slots.

**Build steps.**
1. Replace the static batcher in `mini-serve` with an iteration-based scheduler.
2. Each scheduler step: collect requests in `RUNNING` state, build the batch (their current decode tokens), forward pass, append outputs, mark finished requests.
3. New requests entering: allocate KV cache (paged), tokenize prompt, set state to `RUNNING`.
4. Compare to Level 1 naive batching: throughput, fairness (does a fast request still get fast TTFT even when long ones are running?).

### 15 — `structured-output`

**What it is.** Constrained decoding — the model's output is forced to conform to a grammar (regex, JSON schema, BNF). Implemented by masking the logits at each step to allow only valid next tokens.

**2026 tools.**
- **Outlines** — Python library, finite-state-machine approach.
- **xgrammar** — newer, optimized FSM compilation. Default in vLLM as of late 2025.
- **JSON Schema mode** — built into OpenAI-compatible APIs of vLLM/SGLang/TGI.

**Why infra cares.** Agentic workloads need reliable JSON output. Grammar masking has per-token overhead — measurable and worth knowing.

**Build steps.** Add `response_format={"type": "json_schema", "schema": …}` to your vLLM endpoint. Measure ITL with and without grammar masking on a JSON-output workload.

### 16 — `serving-concurrency`

**What it is.** Real serving stacks are not "one async batcher loop." They have several concurrent loops sharing state, and the locking strategy decides whether the system scales or melts. This topic is the concurrency patterns vLLM and SGLang actually use.

**Patterns to learn.**
- **Sharded locks for the KV block manager.** A global lock around the page free-list serializes every allocation. vLLM uses sharded locks (per-GPU shard, or per-page-pool shard). Read `vllm/core/block/block_manager_v2.py` and find the locking; understand why a single mutex would be the bottleneck.
- **Lock-free or single-writer admission queue.** The scheduler thread is the single writer to the running batch state; HTTP handlers are readers. Single-writer / multi-reader is the cleanest concurrency pattern when it fits.
- **Cancellation propagation.** Client disconnects mid-stream; the decode slot must be freed *promptly*. Naive: poll on every step. Better: an `asyncio.Event` watched by both the HTTP handler and the scheduler. Worse-case: zombie decode slot for the full max_tokens. Build the worse case, observe it, fix it.
- **Stream multiplexing.** One async stream per concurrent decode. Each yields tokens at its own rate. Backpressure: bounded per-stream queue; if client is slow, the queue fills and the scheduler stalls just *that* stream, not the batch.
- **Async vs threaded for tokenization/detokenization.** Tokenization is CPU-bound — running it in the event loop blocks everything. Push it to a thread pool (`asyncio.to_thread`) or a separate worker process.

**What to read in vLLM source:**
- `vllm/core/scheduler.py` — the top-level scheduling loop
- `vllm/core/block/block_manager_v2.py` — locking strategy
- `vllm/engine/async_llm_engine.py` — the asyncio bridge to the scheduler
- `vllm/engine/output_processor/` — how outputs get demuxed back to per-request streams

**Why this matters here.** Levels 1 and 7 use one async loop and call it batching. Production engines have 5+ concurrent loops sharing state via carefully-designed locking. Knowing the difference is the difference between "I built a batcher" and "I understand serving concurrency."

**Build steps.**
1. Add cancellation propagation to your `mini-vllm` from this level. Inject 30% of clients disconnecting mid-decode. Measure: how long do their decode slots stay zombie? Fix it; remeasure.
2. Replace your single global lock around the block manager with sharded locks (8 shards, hash by block ID). Run the same workload; measure tail latency at high concurrency. Should improve.
3. Read vLLM's `block_manager_v2.py`. Annotate every lock acquire/release. Confirm your understanding matches.

### 17 — `spec-decode-systems`

**What it is.** Topic 13 covered speculative decoding as an algorithm (acceptance rate, draft vs target). This topic covers it as a systems problem.

**The hard parts.**
- **Scheduler interaction.** Spec decoding produces a *variable* number of accepted tokens per step (sometimes 1, sometimes 4+). Continuous batching schedulers assume a fixed token-per-step. Reconciling this requires either (a) batching at a finer granularity or (b) accepting that batch slots have variable advance.
- **Tree-based spec (EAGLE-3, Medusa).** The draft model proposes a *tree* of candidate continuations, not a sequence. Verification picks the longest accepted prefix. This means: the verifier's attention mask is non-causal in the spec token region. Custom kernels.
- **Multi-model coordination.** Draft model on different hardware (smaller GPU, or CPU)? Then spec is a network round-trip per step — has to be hidden.
- **Rollback semantics.** When tokens are rejected, you must restore the KV cache to the state before they were inserted. Means the cache write is *tentative* — you commit only on accept. If your KV manager has lazy/async writes, this gets thorny.
- **Quality regression.** Spec decoding produces *exactly the same* output distribution as the target model (it's mathematically equivalent), but only if implemented correctly. Subtle bugs (off-by-one in the verification mask, wrong sampling RNG state) silently change the distribution. Test with `lm-eval-harness` before/after.

**What to read.**
- vLLM spec decoding design — `vllm/spec_decode/` directory
- EAGLE-3 paper for the tree-spec semantics
- The "verify once, sample many" trick — how to verify a tree of K continuations in one forward pass

**Build steps.**
1. Enable spec decoding in your vLLM stack from Level 5's bake-off. Use n-gram (no draft model needed) first.
2. Measure: speedup, acceptance rate, ITL distribution (should have lower mean and higher variance than non-spec).
3. Inject 1000 prompts, measure quality with `lm-eval-harness`. Confirm no regression vs non-spec.
4. Read vLLM's tree-spec verification code; write 100 words on how the attention mask is constructed for a tree of 4 candidates.

## Project 1 — close out this week

`mini-vllm` ships at end of week. Folder layout:

```
mini-serve/
├── server.py
├── batcher_continuous.py        # NEW — replaces batcher.py
├── kv_cache/
│   ├── naive.py
│   ├── paged.py                 # ← the artifact
│   └── eviction.py              # LRU + sliding-window
├── quantization/
│   └── recipes/                 # llm-compressor configs
├── loadtest/
└── reports/
    ├── week1.md                 # Level 1
    ├── profiling-mini-serve.md  # Level 3
    └── project1.md              # ← THE deliverable
```

`reports/project1.md` must contain G1–G5 with Setup/Observation/Insight captions, plus the quality-cost table from Topic 06.

## Definition of done

- [ ] You implemented a paged KV cache from scratch — pages, block table, free list, eviction.
- [ ] You quantized to FP8 or INT4 (AWQ via `llm-compressor`) and ran `lm-eval-harness` to verify.
- [ ] You ran the long-context stress test (100K tokens) and showed naive cache failing where paged held up.
- [ ] You ran `torch.compile`, checked for graph breaks, measured the actual lift.
- [ ] All five graphs (G1–G5) are in `reports/project1.md` with quantitative captions.
- [ ] You can explain in one paragraph what NVFP4 is and why Blackwell exists.

## Resources

- **PagedAttention paper** — [Kwon et al., SOSP 2023](https://arxiv.org/abs/2309.06180). Sections 3–4.
- **vLLM V1 design** — [docs.vllm.ai/v1_guide](https://docs.vllm.ai/en/stable/usage/v1_guide/).
- **vLLM prefix caching** — [docs.vllm.ai/prefix_caching](https://docs.vllm.ai/en/stable/design/prefix_caching/).
- **`llm-compressor`** — [github.com/vllm-project/llm-compressor](https://github.com/vllm-project/llm-compressor).
- **EAGLE-3** — [github.com/SafeAILab/EAGLE](https://github.com/SafeAILab/EAGLE).
- **lm-eval-harness** — [github.com/EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).
- **Microsoft BitNet** — [huggingface.co/microsoft/bitnet-b1.58-2B-4T](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T).
- **Inside vLLM blog** — [blog.vllm.ai/2025/09/05/anatomy-of-vllm.html](https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html). Read this before Level 5.
- **GGUF i-quants explainer** — search "ikawrakow imatrix quantization" or read [llama.cpp's quantize/README](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md).

## Common pitfalls

1. **Skipping `lm-eval-harness`.** "Looks fine" is not a benchmark. The 30-minute eval run is non-negotiable.
2. **Implementing AWQ from a paper instead of using `llm-compressor`.** This is a systems course, not a research course.
3. **Believing `torch.compile` worked without checking for graph breaks.** Silent fallback to eager is the most common bug.
4. **Comparing throughput across precisions without verifying quality.** "FP8 is 2× faster" with -8% MMLU is a regression, not a win.
5. **Writing a "paged KV cache" that's actually still a contiguous tensor.** The free list and block table are the test — if you don't have those, you wrote a slab allocator.
6. **Treating BitNet as a deployment option.** It is not, in 2026, for any production-scale model. Frame it as research.

## What you'll be able to do after this week

> Implement a paged KV cache manager with LRU and sliding-window eviction. Integrate it with a continuous-batching scheduler to handle 100K-token contexts on a single GPU. Quantize a 7B model to FP8 and INT4 (AWQ via `llm-compressor`) and verify quality with `lm-eval-harness`. Diagnose `torch.compile` graph breaks and measure their cost.

Each clause has real measurements behind it. The point isn't the bullet list — it's that you've actually built the data structure and the loop, so when you read vLLM's source it isn't a black box anymore.
