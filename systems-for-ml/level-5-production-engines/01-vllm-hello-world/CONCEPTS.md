# 01 — vLLM Hello World

## What vLLM is

A high-throughput LLM serving engine. Born at UC Berkeley (Sept 2023, the PagedAttention paper). Now a PyTorch Foundation project. The default open-source baseline against which every other engine is measured.

What you get out of the box in 2026:

- OpenAI-compatible HTTP server (`/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`)
- Paged KV cache with automatic prefix caching (on by default in V1)
- Continuous batching with chunked prefill (on by default in V1)
- CUDA graph capture (piecewise, auto)
- FlashInfer kernels for attention
- Tensor / pipeline / expert / data parallelism flags
- Quantization paths: FP8, AWQ, GPTQ, GGUF, BitsAndBytes, NVFP4 on Blackwell
- Multi-LoRA, speculative decoding, structured output (xgrammar)

You do nothing. The flags exist to tune; the defaults already beat hand-rolled servers by a wide margin. That ergonomic gap is the whole reason vLLM exists.

## V1 vs V0 — what you're actually running

Since vLLM 0.8 (Q1 2025), V1 is the default. By 2026 V0 is gone from the codebase.

```
V0 (2023-2024)                       V1 (2025-2026)
──────────────                        ──────────────
Single Python process                 AsyncLLM (front) + EngineCore (back)
Step rebuilds full batch state        Persistent batch + diff updates
Scheduler+exec on same GIL            Scheduler in EngineCore, exec in workers
Prefix cache opt-in                   Prefix cache always on (zero-overhead)
Manual chunked prefill                Chunked prefill on by default
Manual CUDA graph capture             Piecewise CUDA graphs auto
Spec decode bolt-on                   Spec decode first-class
```

Pre-2025 tutorials describe flags that are now no-ops or removed. Read the V1 user guide before tuning anything: https://docs.vllm.ai/en/stable/usage/v1_guide/

## The two ways to use vLLM

### Online serving (this topic)

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000
```

You get an OpenAI-compatible HTTP server. Hit it with the OpenAI Python client by setting `base_url`. This is the production deployment shape.

### Offline batch (Topic 11)

```python
from vllm import LLM
llm = LLM(model="Qwen/Qwen2.5-7B-Instruct")
outputs = llm.generate(prompts, sampling_params)
```

In-process, batch-oriented, no HTTP. For million-doc scoring jobs. Different optimization regime — throughput at all costs, latency irrelevant.

## What just happened when you ran `vllm serve`

```
vllm serve <model>
       │
       ▼
┌────────────────────────────────────────────────────┐
│ AsyncLLM (front-end, async Python)                 │
│   - HTTP handlers                                  │
│   - tokenization                                   │
│   - sampling-param parsing                         │
│   - SSE streaming back to client                   │
└────────────────────────────────────────────────────┘
       │  ZMQ over shared memory
       ▼
┌────────────────────────────────────────────────────┐
│ EngineCore (back-end process)                      │
│   - Scheduler (token-budget, mixes prefill+decode) │
│   - Block manager (paged KV)                       │
│   - Driver process for workers                     │
└────────────────────────────────────────────────────┘
       │  per-rank workers (TP, PP, DP)
       ▼
┌────────────────────────────────────────────────────┐
│ Worker (one per GPU rank)                          │
│   - Model executor (forward pass)                  │
│   - FlashInfer attention                           │
│   - Sampler                                        │
│   - KV-cache GPU tensors                           │
└────────────────────────────────────────────────────┘
```

The split matters: front-end Python latency no longer blocks the GPU step. This is why V1 throughput on small models jumped 5-10× over V0.

## The flags worth knowing on day one

```
--model                       HF repo or local path
--dtype                       auto | bfloat16 | float16 | fp8
--max-model-len               cap context length (saves KV memory)
--gpu-memory-utilization      fraction of VRAM to claim (0.9 default)
--tensor-parallel-size        TP across GPUs on one node
--pipeline-parallel-size      PP across stages
--enable-prefix-caching       on by default in V1
--enforce-eager               disable CUDA graphs (debugging only)
--quantization                fp8 | awq | gptq | bitsandbytes | gguf
--enable-lora                 + --max-loras + --lora-modules
--speculative-config          n-gram, EAGLE, MLP spec
```

`--gpu-memory-utilization` is the one you'll touch first when something OOMs. Drop it from 0.9 to 0.85 to leave headroom for activations on long context.

## Apple Silicon / no-NVIDIA reality

vLLM officially targets NVIDIA + Linux. There is a `vllm` Mac wheel that runs on CPU only — useful for testing API shape, useless for benchmarks.

Real options:

1. **Remote GPU** — RunPod / Lambda / Vast.ai. An L4 at $0.40-0.80/hr is enough for 7B-class models.
2. **Docker with NVIDIA runtime** — if you have a CUDA workstation.
3. **vLLM-MLX fork** — Apple-maintained, paged KV on Metal, throughput numbers exist but are not vLLM-on-CUDA. Treat as "Apple's local-serving option," not the engine you benchmark in Project 2.

For Project 2's bake-off, you need a real NVIDIA GPU. Budget $30-50 for the week.

## Pitfalls

1. **Tutorial drift.** A 2024 blog post says "enable chunked prefill" — V1 has it on by default. Setting the flag is harmless; expecting a speedup from setting it is not.
2. **`--max-model-len` left at the model's default.** A 32K-context model with 32K reserved per request shrinks your concurrent capacity. Cap it to what you actually serve.
3. **Forgetting `--gpu-memory-utilization`.** Two engines on one GPU need it dropped per-process or they'll fight.
4. **Treating `vllm serve` as a black box.** It's not. The next topic walks through what's inside.

## References

- vLLM home — https://docs.vllm.ai/
- V1 user guide — https://docs.vllm.ai/en/stable/usage/v1_guide/
- OpenAI-compat server — https://docs.vllm.ai/en/stable/serving/openai_compatible_server.html
- Anatomy blog (required reading for Topic 02) — https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html
- Quantization support matrix — https://docs.vllm.ai/en/stable/quantization/supported_hardware.html
