# 04 — TensorRT-LLM

## Files

- `CONCEPTS.md` — what TRT-LLM is in 2026, PyTorch flow vs engine-build flow, NIM, when to actually pick it, the operational tax
- `run_pytorch_flow.py` — minimal PyTorch-flow example using `tensorrt_llm.LLM`

## Quickstart

On Linux + NVIDIA Hopper or Blackwell:

```bash
pip install tensorrt-llm
# may build from source on some configs — budget 30 min for first install

# in-process generation
python run_pytorch_flow.py

# or as a server, OpenAI-compatible
trtllm-serve Qwen/Qwen2.5-7B-Instruct --port 8002 \
    --tp_size 1 --kv_cache_free_gpu_memory_fraction 0.9
# then point Topic 01's serve_and_hit.py at http://localhost:8002/v1
```

## Expected output

First run: 5-30 minutes building the engine. Subsequent runs reuse the cached build.

```
Building engine (first run is slow — 5-30 min for 7B FP8)...
  ready in 612.4s
4 prompts, 512 output tokens in 1.81s = 283 tok/s

--- Prompt 0 ---
Paged KV cache treats key/value memory like virtual memory: each request
sees a logical sequence of tokens, but the manager maps them to non-contiguous
physical blocks of fixed size, so mixed-length traffic doesn't waste memory.
...
```

Numbers vary widely. The shape: in-process throughput should beat vLLM by 10-30% on a tuned Hopper FP8 config.

## Try

- **Time the build.** Note it. Compare to vLLM's "0 build time." Operational cost is real.
- **Enable FP8 in `quant_config`.** Re-build. Compare throughput.
- **On Blackwell:** try NVFP4. The 2× on top of FP8 is Blackwell-only.
- **Same workload as Topic 01.** Compare apples-to-apples; document tuning effort spent on each engine.
- **Document install friction in your notes.** This is part of the bake-off finding.

## Where this goes

- Topic 07 — TRT-LLM is one of the entries in the bake-off
- Topic 08 — disaggregated serving works with TRT-LLM via the Dynamo orchestration layer
- Topic 13 — TensorRT (without the -LLM) is the runtime path for non-autoregressive workloads (vision, embeddings)
