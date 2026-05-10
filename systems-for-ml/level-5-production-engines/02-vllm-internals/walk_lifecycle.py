"""
02 - vLLM internals: a reading walkthrough printed as ASCII.

This script doesn't run vLLM. It prints the request lifecycle annotated with
the actual file:line references in the vLLM source as of v0.11 (May 2026).

Use it as a checklist: open each referenced file in your editor while reading
the anatomy blog (https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html).

Pin to a specific commit before relying on the line numbers — they drift.
The relative locations (which function in which file) are stable; the line
numbers are not.
"""

LIFECYCLE = r"""
╔══════════════════════════════════════════════════════════════════╗
║ vLLM V1 — life of one /v1/chat/completions request               ║
╚══════════════════════════════════════════════════════════════════╝

[1] HTTP arrives
    vllm/entrypoints/openai/api_server.py
        FastAPI route: create_chat_completion()
        Validates request, builds SamplingParams, calls AsyncLLM.generate()

[2] AsyncLLM (front-end process)
    vllm/v1/engine/async_llm.py :: AsyncLLM.generate()
        - tokenize prompt (offloaded to a thread pool to free the event loop)
        - assemble EngineCoreRequest
        - send via ZMQ to EngineCore
        - return an async iterator of output tokens to the route handler
        - SSE chunks go back through the FastAPI streaming response

[3] EngineCore (back-end process)
    vllm/v1/engine/core.py :: EngineCore.run_busy_loop()
        Each tick:
          a) drain new requests from the input queue, add to scheduler
          b) scheduler.schedule() -> SchedulerOutput
          c) workers execute SchedulerOutput, return ModelRunnerOutput
          d) scheduler.update_from_output(...)
          e) push outputs back to AsyncLLM via ZMQ

[4] Scheduler (the heart)
    vllm/v1/core/sched/scheduler.py :: Scheduler.schedule()
        - decides who runs this step within max_num_batched_tokens budget
        - mixes running decodes (1 token each) with prefill chunks
        - admits new requests up to KV-cache capacity
        - allocates blocks via KVCacheManager
        - emits a diff (newly scheduled, finished, preempted) — not full state

[5] KVCacheManager
    vllm/v1/core/kv_cache_manager.py :: KVCacheManager.allocate_slots()
    vllm/v1/core/block_pool.py        :: BlockPool.get_new_blocks()
        - paged KV: free-list pop + per-request block_table append
        - prefix-cache lookup: hash chain over blocks (sha256 by default)
        - cache salting / extra_keys for multimodal and per-tenant isolation

[6] Worker.execute_model()
    vllm/v1/worker/gpu_worker.py :: GPUWorker.execute_model()
    vllm/v1/worker/gpu_model_runner.py :: GPUModelRunner.execute_model()
        - prepare_inputs: pack token_ids, positions, block_tables into tensors
        - replay piecewise CUDA graphs for the static portions of the forward
        - run attention eagerly (dynamic shape) via FlashInfer
        - sample on-GPU (no CPU sync until the result is needed)

[7] Attention kernel
    vllm/v1/attention/backends/flashinfer.py
    flashinfer/python/flashinfer/decode.py and prefill.py
        - paged-KV attention: gathers K,V from non-contiguous block ids
        - fused with rotary embeddings, masks, scaling

[8] Sampler
    vllm/v1/sample/sampler.py :: Sampler.forward()
        - applies temp / top-k / top-p / min-p / penalties
        - structured output: xgrammar bitmask masking before softmax
        - returns sampled token ids on GPU

[9] Update + emit
    Scheduler.update_from_output()  --> marks finishing requests done
    AsyncLLM consumes EngineCoreOutputs, yields one chunk per token to FastAPI
    FastAPI emits the SSE chunk to the client

  Loop until the request hits stop, max_tokens, or client disconnect.
  Disconnect propagation: AsyncLLM cancels, EngineCore aborts the request,
  Scheduler frees the KV blocks. (Level 4 Topic 16 covers this.)

╔══════════════════════════════════════════════════════════════════╗
║  Things to underline as you read                                ║
╚══════════════════════════════════════════════════════════════════╝

* The diff-based SchedulerOutput is what makes V1 fast at large batch sizes.
  V0 sent the whole batch state every step; V1 sends "added X, removed Y."
* CUDA graphs are captured per shape bucket (per batch size, basically).
  First request at a new shape pays a one-time capture cost.
* Prefix-cache hits cost zero KV memory and zero compute for matched tokens.
  The block_table just points at existing blocks.
* The KVConnector interface lets LMCache, Mooncake, NIXL plug in for
  cross-engine KV transfer. This is how disaggregated serving works in 2026.
"""


def main() -> None:
    print(LIFECYCLE)


if __name__ == "__main__":
    main()
