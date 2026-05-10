# 02 — vLLM Internals

## Files

- `CONCEPTS.md` — the V1 process model, scheduler, block manager, FlashInfer, what your `mini-vllm` skipped
- `walk_lifecycle.py` — prints the request lifecycle as ASCII with file/function references into the vLLM source

## How to use this topic

This is a reading topic. The exercise is opening the vLLM source side-by-side with your `mini-vllm` and walking the lifecycle.

```bash
git clone https://github.com/vllm-project/vllm
cd vllm
python <path>/walk_lifecycle.py
# then open each file the script names, in order
```

Read the [anatomy blog](https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html) end-to-end. It's the single best document on real serving-engine internals. ~45 minutes; budget that.

## What to walk away with

- A one-page hand-drawn diagram of the request lifecycle (AsyncLLM → EngineCore → Scheduler → Worker → Sampler → back).
- A list of three things vLLM does that your `mini-vllm` doesn't.
- The vocabulary to read a vLLM PR or issue without bouncing — `SchedulerOutput`, `KVCacheManager`, `BlockPool`, `GPUModelRunner`, `KVConnector`, `extra_keys`.

## Try

- **Find where xgrammar plugs in.** Trace a structured-output request from `SamplingParams` to the bitmask applied before softmax. (Hint: `vllm/v1/sample/`.)
- **Find the disconnect path.** Client disconnects mid-stream. What cancels what? (Hint: `AsyncLLM.abort` → `EngineCore.abort_requests` → `Scheduler.finish_requests`.)
- **Find the KVConnector interface.** Read the abstract class and one implementation (LMCache or NIXL). Topic 08 builds on this.

## Where this goes

- Topic 03 — same exercise for SGLang
- Topic 08 — disaggregated serving uses the KVConnector you found here
- Level 7 — the Prometheus metrics emitted by the scheduler are what your autoscaler reads
