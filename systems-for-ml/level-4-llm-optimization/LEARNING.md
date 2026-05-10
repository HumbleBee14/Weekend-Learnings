# Level 4 — Learning Path

The longest level in the curriculum. 17 topics organized into four sub-arcs:

```
Quantization sub-arc       (01-06)  the data-precision lever
torch.compile + fusion     (07-08)  the kernel-launch and HBM-roundtrip levers
KV cache sub-arc           (09-12)  the heart of mini-vllm
Spec decode + batching     (13-17)  the throughput multipliers
```

## The path

| Folder | Time | What you walk away with |
|---|---|---|
| `01-quantization-basics/` | 1-2h | BF16 baseline, INT8/NF4 with bitsandbytes |
| `02-fp8-and-nvfp4/` | 2-3h | FP8 (E4M3) via llm-compressor, NVFP4 (Blackwell). Two-level scaling. |
| `03-weight-only-ptq/` | 1-2h | AWQ at 4-bit. The 2026 verdict on AWQ vs GPTQ vs HQQ. |
| `04-local-quant-formats/` | 1-2h | GGUF K-quants vs i-quants, Unsloth Dynamic v2.0. |
| `05-extreme-quantization/` | 1h | 3/2-bit deployable, BitNet 1.58 still research. |
| `06-quality-evaluation/` | 2-3h | KL divergence (the 2026 standard) + lm-eval-harness. |
| `07-torch-compile/` | 1-2h | Piecewise CUDA graphs, the canonical 2026 inference recipe. |
| `08-kernel-fusion/` | 1-2h | Liger-Kernel + FlashInfer. Why fusion is the prize. |
| `09-kv-cache-naive/` | 1-2h | A working naive KV cache. Feel its four problems. |
| `10-kv-cache-paged/` | 3-4h | Paged KV cache from scratch. The OS virtual memory analogy. |
| `11-kv-cache-eviction/` | 2h | Prefix sharing, LRU eviction, the block-hash kv-connector standard. |
| `12-long-context-stress/` | 2-3h | 32K-128K context. Chunked prefill. MLA, Mooncake, LMCache awareness. |
| `13-speculative-decoding/` | 2-3h | n-gram spec, EAGLE-3, P-EAGLE (Feb 2026). Acceptance rate. |
| `14-continuous-batching/` | 3-4h | Replace static batching with continuous. The vLLM V1 scheduler model. |
| `15-structured-output/` | 1-2h | xgrammar / XGrammar-2 (May 2026). JSON schema masking. |
| `16-serving-concurrency/` | 2-3h | Sharded locks, cancellation propagation, stream multiplexing. |
| `17-spec-decode-systems/` | 1-2h | The systems-level problems spec decode raises. |

Total: ~30-45 hours of focused work. The longest level by design.

## What's new in 2026 (deltas vs 2024-2025 content)

The research backing this level surfaced several things that have changed status. Key items in case you saw older material:

- **NVFP4's two-level scaling** is the real innovation, not just smaller blocks
- **AWQ effectively won** at 4-bit weight-only; HQQ is the data-free alternative
- **KL divergence has overtaken perplexity** as the quantization quality metric
- **vLLM V1 piecewise CUDA graph pattern** is the canonical compile recipe
- **Mooncake joined PyTorch Ecosystem** (Feb 2026); KV disaggregation is mainstream
- **Unsloth Dynamic v2.0 GGUFs** beat both K-quants and i-quants
- **FP4 just landed in llama.cpp** — local users now have NVFP4/MXFP4
- **MLA (DeepSeek)** deserves first-class treatment — 93% KV reduction with *better* perplexity
- **P-EAGLE (Feb 2026)** — biggest spec-decode delta of the year, in vLLM v0.16+
- **XGrammar-2 (May 2026)** — up to 80× compilation speedup over XGrammar
- **Block-hash kv-connector** is becoming a cross-engine standard (vLLM ↔ LMCache)

## What hardware you need

- **A real GPU** for almost everything. Free Colab T4 works for Topics 01, 06, 09 but cramped for the rest.
- **Hopper (H100/H200) ideal** for Topic 02 onward (FP8 needs Hopper+).
- **Blackwell (B200)** if you want to actually run NVFP4 at full speed (Topic 02).
- **Apple Silicon Mac** for Topic 04 (GGUF) — works on any Mac.

For learners without Hopper: Topic 02 still teaches the concepts; you just won't get the FP8 throughput numbers. Rent an H100 hour ($2/hr on RunPod) for the measurements.

## Each topic folder

Same shape as Levels 1-3:

- `CONCEPTS.md` — theory + 2026 state
- One or more code files (`.py`) demonstrating the topic
- `README.md` — quickstart, expected output, things to try

## Project 1 closes here

`mini-serve` (Level 1) → `mini-vllm` (Level 4). Drop in:
- Paged KV cache (Topic 10) + prefix sharing (Topic 11)
- Continuous batching (Topic 14)
- FP8 quantization (Topic 02) — if you have Hopper+
- torch.compile (Topic 07)
- Liger-Kernel fused ops (Topic 08)
- Cancellation propagation (Topic 16)

Run the full break-it list:
- Long context (32K) workload
- Mixed-length batch with shared system prompt
- 100 concurrent users with 30% disconnecting mid-stream
- Quality regression test with lm-eval-harness

Ship `reports/project1.md` with all required graphs and a side-by-side comparison to Level 1's static-batching baseline.

## After this level

Level 5 is the engine bake-off: your `mini-vllm` vs vLLM vs SGLang vs TRT-LLM vs llama.cpp. Same workload, same hardware. Real numbers.
