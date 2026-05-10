# 07 — Engine Bake-Off (Project 2)

This is **Project 2** — the artifact for Level 5.

The deliverable is `reports/bakeoff.md`: a short systems-paper eval doc covering the same model, same prompts, same hardware, run through vLLM, SGLang, TensorRT-LLM, llama.cpp, and your `mini-vllm` from Level 4.

## What an honest bake-off looks like

Honest bake-offs aren't "engine X is fastest." They're **workload-conditional**:

> For workload X on hardware Y at quantization Z, engine A wins on metric M by N%, because of architectural reason R.

Anything less specific is folklore. Real production teams write workload-conditional recommendations because that's what survives contact with a heterogeneous fleet.

## Methodology — the things to fix before measuring

```
Fix:                              Why:
────                              ────
model                             same weights, same chat template
hardware                          same GPU SKU, same CUDA version, same driver
quantization (per engine)         document which quant per engine — best per
prompt set                        identical, with seeded sampling for repeatability
sampling params                   temp / top-p / max_tokens identical
warmup                            10+ throwaway requests; first call captures CUDA graphs
duration                          measure for ≥60s of steady-state, not 10 requests
concurrency sweep                 1, 4, 16, 64 — at minimum
metric collection                 client-side TTFT/ITL + server-side /metrics
```

The dirty truth: a bake-off where you forgot to warm up TRT-LLM (which captures graphs on first run) will say "TRT-LLM lost to llama.cpp on TTFT." That's noise. Warmup matters.

## The four required workloads (mapped to Project 2 break-it list)

```
W1  short prompts                 128-token prompt, 128-token output
                                  Tests TTFT and small-batch behavior

W2  long prompts                  4K-token prompt, 256-token output
                                  Tests prefill cost, chunked-prefill effectiveness

W3  prefix-heavy (chatbot)        4KB shared system prompt + varied user turn
                                  Tests RadixAttention / prefix cache

W4  memory-constrained            Smaller GPU than the model nominally needs
                                  Tests KV cache management under pressure
```

Plus the **cross-substrate** scenario: llama.cpp on CPU vs the same model on a small GPU. This produces G9 (cost per million tokens) and is the row that makes the report honest about $/Mtok.

## Required graphs (G6-G9)

```
G6  TTFT bar chart per engine, split by W1 (128-tok) vs W2 (4K-tok)
    Shows: TTFT scales differently with prefill cost across engines.
    Surprises: llama.cpp at batch=1 sometimes beats vLLM on W1.

G7  Throughput (tok/s) per engine on identical workload
    Shows: hand-tuned TRT-LLM usually wins on H100; vLLM closes most of the gap;
    SGLang wins specifically on W3 (prefix-heavy).

G8  GPU memory usage vs context length, per engine
    Shows: paged-KV engines flatten while contiguous-cache or oversized
    pre-allocation grows. Reveals which engine fits the most concurrent users.

G9  Cost per million tokens per engine + quantization
    Compute as: ($/hr for the instance) / (tok/s) * 1e6 / 3600
    Include CPU-only llama.cpp as one row. Cross-substrate run is the headline.
```

Each graph needs a **Setup → Observation → Insight** caption. Numbers without an insight are just noise.

## The runner contract

`runner.py` is the harness that drives all engines uniformly. Same client code; the only thing that changes is the `--base-url`. This is why we wrote `serve_and_hit.py` (Topic 01) the way we did — it's the seed of the runner.

```
runner.py
├── EngineConfig    {name, base_url, model_id, quant, warmup_n}
├── Workload        {name, prompts.jsonl, max_tokens, concurrency}
├── for engine in engines:
│   for workload in workloads:
│       warmup(engine, workload)
│       results[engine][workload] = drive(engine, workload)
└── write to results.parquet  →  plot.py  →  reports/bakeoff.md figures
```

Everything else (engine startup, model conversion, server flags) lives in `configs/` per-engine. The runner doesn't know how engines are started; it only hits OpenAI-compatible endpoints.

## Reading the results

Sketch of a real findings paragraph (placeholder numbers):

> *On W3 (prefix-heavy chat), SGLang's TTFT p99 was 2.1× lower than vLLM's at concurrency 32, while throughput was within 3%. The TTFT gap closed on W1 (no shared prefix) where both engines were within 5% on every metric. TRT-LLM led aggregate throughput on W2 by 21%, but at the operational cost of a 14-minute build cycle and FP8-only weights. llama.cpp on a 16-core EPYC at Q4_K_M beat an L4 GPU on $/Mtok at QPS ≤ 2 but lost by 4× at QPS ≥ 8.*

Each clause is workload-conditional, quantitative, and grounded in a graph. That is the level the bake-off doc is held to.

## The recommendation section

The doc must end with at least three (workload, hardware) → engine recommendations, each with the reason. Example shape:

```
For chatbot-style traffic (high prefix overlap) on Hopper:
    SGLang. Reason: RadixAttention's tree-based prefix sharing
    drops TTFT p99 by 2× on the prefix-heavy workload while
    matching vLLM on aggregate throughput.

For batch processing (offline scoring) on Hopper:
    vLLM offline mode (Topic 11). Reason: ergonomics + close-to-peak
    throughput; TRT-LLM's lead doesn't justify the build cycle.

For batch=1 latency-sensitive on Apple Silicon:
    llama.cpp + Metal. Reason: zero-Python overhead, 5-10× lower
    cold start, mature Metal kernels.

For low-QPS CPU deployment:
    llama.cpp on CPU. Reason: $/Mtok crossover at QPS ≤ 2 against
    a small cloud GPU; eliminates GPU rental entirely.
```

## Pitfalls — the ones that wreck bake-offs

1. **Default-vs-tuned asymmetry.** Spend equal tuning effort across engines or document the asymmetry explicitly.
2. **No warmup.** First-request cost is non-representative. Discard ≥10 warmup requests.
3. **Single-shot measurements.** Run for 60+ seconds of steady state; report p50/p95/p99 not just mean.
4. **Comparing different quants without quality check.** FP8 vs Q4 isn't apples-to-apples until you've shown both pass `lm-eval-harness` within tolerance (Level 4 Topic 06).
5. **Ignoring memory.** Two engines at the same throughput but with 2× memory difference are not equivalent — the slimmer one fits more LoRAs / longer context.
6. **No prefix-heavy workload.** Without it, you'll conclude "vLLM and SGLang are basically the same."
7. **Forgetting operational cost.** Install pain, build time, debuggability are first-class metrics, not footnotes.

## What to ship

```
engine-bakeoff/
├── configs/
│   ├── vllm.yaml
│   ├── sglang.yaml
│   ├── trt-llm.yaml
│   ├── llama-cpp.json
│   └── mini-vllm.yaml
├── workloads/
│   ├── short.jsonl              # W1
│   ├── long.jsonl               # W2
│   ├── prefix-heavy.jsonl       # W3
│   └── memory-constrained.jsonl # W4
├── runner.py                    # this folder
├── plot.py                      # produces G6-G9
└── reports/
    └── bakeoff.md               # THE deliverable
```

`reports/bakeoff.md` follows the systems-paper structure from the outer plan: Problem → Methodology → Results → Per-engine notes → Recommendations → Operational notes.

## References

- Project 2 brief — `../README.md` (the topic list)
- vLLM benchmark suite — https://github.com/vllm-project/vllm/tree/main/benchmarks
- SGLang benchmark suite — https://github.com/sgl-project/sglang/tree/main/benchmark
- TensorRT-LLM benchmark scripts — https://github.com/NVIDIA/TensorRT-LLM/tree/main/benchmarks
- llama-bench — https://github.com/ggml-org/llama.cpp/tree/master/tools/llama-bench
- BetterBench (academic, 2024) — https://arxiv.org/abs/2411.12990 — methodology critiques worth reading
