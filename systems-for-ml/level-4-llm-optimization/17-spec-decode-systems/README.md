# 17 — Speculative Decoding Systems

## Files

- `CONCEPTS.md` — five systems-level problems spec decode raises (variable accept rate, KV rollback, tree-spec mask, multi-model coordination, quality regression bugs); P-EAGLE integration

## What you do this topic

Reading-driven. Understand the systems integration. The implementation lives in vLLM's source.

## Quickstart

```bash
git clone https://github.com/vllm-project/vllm
cd vllm

# Read the spec decode source
ls vllm/spec_decode/
cat vllm/spec_decode/spec_decode_worker.py
cat vllm/spec_decode/multi_step_worker.py

# Trace one verification step from start to finish:
# - Where draft proposes K tokens
# - Where the verifier runs them
# - Where rejection happens
# - Where KV cache rollback happens
```

## Worth doing

Run vLLM with spec decode enabled and *measure quality*:

```bash
# Without spec
lm-eval --model vllm \
    --model_args 'pretrained=meta-llama/Llama-3.1-8B-Instruct' \
    --tasks mmlu,gsm8k,humaneval --batch_size 8 \
    --output_path results/no_spec

# With spec (n-gram or EAGLE)
lm-eval --model vllm \
    --model_args 'pretrained=meta-llama/Llama-3.1-8B-Instruct,speculative_config={"method": "ngram", "num_speculative_tokens": 5, "prompt_lookup_max": 5}' \
    --tasks mmlu,gsm8k,humaneval --batch_size 8 \
    --output_path results/spec

# Compare results — should be IDENTICAL within sampling noise. If not, there's a bug.
```

This is the test that catches the "subtle implementation bugs" from CONCEPTS.md. If MMLU dropped 1 point with spec decode on, the implementation is broken.

## What you should walk away with

- Why spec decode is non-trivial at the systems level
- Where bugs typically hide (mask construction, rollback, RNG state)
- The systematic way to verify correctness (lm-eval before/after)
- Awareness that spec decode is mature in vLLM but the underlying mechanics are still evolving (P-EAGLE was Feb 2026)

## Where this goes

Level 4 closes here. You have, end-to-end:

- Quantization (Topics 01-06)
- torch.compile + kernel fusion (07-08)
- Paged KV cache + prefix sharing + eviction + long context (09-12)
- Speculative decoding (13, 17)
- Continuous batching (14)
- Structured output (15)
- Serving concurrency primitives (16)

That's `mini-vllm`. Project 1 closes: drop the paged KV + continuous batching into your Level 1 server, run the full break-it list, ship `reports/project1.md` with all required graphs.

Level 5 is the engine bake-off — `mini-vllm` benchmarked head-to-head against vLLM, SGLang, TensorRT-LLM, llama.cpp.
