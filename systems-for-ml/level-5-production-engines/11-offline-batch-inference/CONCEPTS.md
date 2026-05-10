# 11 — Offline Batch Inference

## Why this is its own thing

Online serving and offline batch are different optimization regimes:

```
                           Online                Offline batch
                           ──────                ─────────────
SLA                        TTFT, p99 latency     wall-clock time, $/Mtok
Concurrency               variable, bursty       maximal, sustained
Latency tolerance         strict                 effectively infinite
Throughput target         peak                   peak² (no latency tradeoff)
Failure handling          retry per request      checkpoint per shard
Workload visibility       discovered             known up front
```

Online tunes for the worst-case request. Offline tunes for total tokens per second per dollar. Same engine, different flags, different mental model.

## When you actually run batch inference

- **Million-doc scoring.** Classify, embed, summarize, score every document in a corpus.
- **Synthetic data generation for fine-tuning.** Generate 10M Q-A pairs from seed prompts.
- **Eval pipelines.** Run a benchmark suite (`lm-eval-harness`-style) against a checkpoint.
- **Re-tagging archives.** Apply a new model to old data.
- **Rejection sampling** for RLHF / DPO data prep.

If the input set is fully known and the output is consumed asynchronously, you're doing batch.

## vLLM offline mode

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    tensor_parallel_size=1,
    dtype="bfloat16",
    quantization="fp8",          # if your GPU supports it
    gpu_memory_utilization=0.92, # crank higher than online (no autoscale headroom)
    max_num_seqs=512,            # bigger batch — latency irrelevant
    max_num_batched_tokens=8192,
    enable_prefix_caching=True,  # huge win if prompts share prefixes
)

prompts = read_lines("docs.jsonl")
params = SamplingParams(max_tokens=256, temperature=0.0, top_p=1.0)
outputs = llm.generate(prompts, params)

for prompt, out in zip(prompts, outputs):
    write_result(prompt, out.outputs[0].text)
```

Same engine internals as online, exposed in-process. The flags that change:

```
gpu_memory_utilization      0.92-0.95   (vs 0.85-0.90 online)
max_num_seqs                512-1024    (vs 64-256 online)
max_num_batched_tokens      8192-16384  (vs 2048-4096 online)
enable_chunked_prefill      True (default in V1)
enable_prefix_caching       True (default in V1)
```

You're not optimizing TTFT; you're maximizing how many concurrent sequences fit in KV cache.

## Throughput math you should know

A 7B BF16 on an H100: peak ~30K tok/s in offline mode, ~2-3K tok/s in interactive online mode. The gap comes from:

1. Bigger batch (latency unconstrained)
2. Higher GPU memory utilization
3. No SSE / OpenAI-protocol overhead
4. Prefix-cache hits across the entire input set, not just per-request

## Cost arithmetic

```
$/Mtok (offline) = ($/hr) / (output_tok/s) * 1e6 / 3600

H100 SXM at $2.5/hr * 30K tok/s out:
    $2.5/hr * (1e6 / 30000 / 3600) ≈ $0.023 / Mtok output

L4 at $0.4/hr * 5K tok/s out:
    $0.4/hr * (1e6 / 5000 / 3600) ≈ $0.022 / Mtok output

Same $/Mtok. The L4 is slower wall-clock but cheaper per token at lower
throughput — pick based on whether you need the job done in 1h or 6h.
```

This is why offline batch is often the *cheapest* serving mode per token, and why "use the API" is sometimes more expensive than running batch on a rented GPU.

## SkyPilot / Anyscale / Ray Data — the orchestration step

For serious batch jobs you need:

- **Sharding the input** across GPUs (Ray Data, SkyPilot map-reduce, plain `multiprocessing`)
- **Checkpointing** — resume from where you left off if the GPU dies
- **Rate limiting writes** — don't overwhelm the output sink
- **Quality sampling** — periodic spot-checks during the run

Ray Data has a first-class `vllm_predictor` integration for this:

```python
import ray
from ray.data.llm import vLLMEngineProcessorConfig, build_llm_processor

processor = build_llm_processor(
    vLLMEngineProcessorConfig(model="Qwen/Qwen2.5-7B-Instruct", concurrency=4),
    preprocess=lambda row: dict(prompt=row["text"], sampling_params=...),
    postprocess=lambda row, out: dict(id=row["id"], output=out.outputs[0].text),
)
ds = ray.data.read_parquet("s3://corpus/")
processor(ds).write_parquet("s3://results/")
```

Auto-shards across the Ray cluster. This is the actual production shape.

## Pitfalls

1. **Treating offline like online.** Max batch, GPU util at 95%, accept variable latency.
2. **Forgetting prefix caching.** If your batch shares system prompts (RAG, classification with a fixed instruction), prefix caching can 2-3× throughput.
3. **Sequential reads from a slow storage tier.** GPU pegged, but the input pipeline is the bottleneck. Pre-stage to local NVMe.
4. **No checkpointing.** A 12-hour batch job is a different world from a 12-second request. Plan for failure.
5. **Quality drift mid-run.** A bad prompt template gets discovered after 8 hours. Run a quality-check sample before committing the full corpus.

## What to do this topic

1. Take 10K prompts (synthetic; the data isn't the point).
2. Run them through `vllm.LLM().generate()` (script provided).
3. Measure tok/s and $/Mtok.
4. Compare to running them one-by-one through the OpenAI server. Should be 5-15× faster in offline mode.
5. (Bonus) Wire it through Ray Data for the realistic shape.

## References

- vLLM offline inference docs — https://docs.vllm.ai/en/stable/getting_started/quickstart.html#offline-batched-inference
- Ray Data + vLLM — https://docs.ray.io/en/latest/data/working-with-llms.html
- SkyPilot batch jobs — https://docs.skypilot.co/en/latest/examples/llm-serving/vllm.html
- `LLM.generate` API — https://docs.vllm.ai/en/stable/api/offline_inference/llm.html
