# 01 — vLLM Hello World

## Files

- `CONCEPTS.md` — what vLLM is in 2026, V1 vs V0, the AsyncLLM/EngineCore split, the day-one flags
- `serve_and_hit.py` — async OpenAI-client harness measuring TTFT, ITL, throughput, p50/p95/p99

## Quickstart

On a Linux + NVIDIA box:

```bash
pip install vllm openai
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000 --max-model-len 8192
# in another shell:
python serve_and_hit.py --n 50 --concurrency 8 --max-tokens 128
```

On Mac / no-GPU: rent an L4 / A10 from RunPod or Vast.ai for the week. Budget $30-50 for all of Level 5.

## Expected output

```
--- vLLM hello world: 50 requests @ concurrency 8 ---
model               Qwen/Qwen2.5-7B-Instruct
wall time           4.10s
agg throughput      1561.0 tok/s
TTFT  p50 / p95 / p99   180 / 350 / 490  ms
ITL   p50 / p95          17.5 / 24.0  ms
per-req tok/s   median   58.2
```

Numbers vary widely with GPU and prompt length. The shape that matters: TTFT in the low hundreds of ms, ITL in the 10-30ms range, throughput scaling roughly linearly with concurrency until the batch hits the KV-cache or compute ceiling.

## Try

- **Drop concurrency to 1, then ramp up to 32.** Watch throughput climb sub-linearly while TTFT stays roughly flat — that's continuous batching at work.
- **Add `--enforce-eager` to the server.** CUDA graphs off. Re-run. Throughput drops noticeably; this is what V1's piecewise CUDA-graph capture is buying.
- **Cap `--max-model-len 2048`.** More concurrent requests fit in KV cache; throughput at high concurrency goes up.
- **Hit `/metrics`** (Prometheus). Note `vllm:num_requests_running`, `vllm:num_requests_waiting`, `vllm:gpu_cache_usage_perc`, `vllm:time_to_first_token_seconds`. Level 7's autoscaler reads these.
- **Send the same prompt 10 times.** Watch TTFT collapse on requests 2-10 — automatic prefix caching hit.

## Where this goes

- Topic 02 cracks open the engine itself — what `vllm serve` actually orchestrates
- Topic 07 makes this harness the bake-off `runner.py` (also drives SGLang, TRT-LLM, llama.cpp)
- Level 7 reuses the same Prometheus metrics for the autoscaler
