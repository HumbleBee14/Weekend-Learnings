# 10 — Multi-LoRA Serving

## Files

- `CONCEPTS.md` — why multi-LoRA exists, Punica/S-LoRA kernels, what to measure, the 2026 frontier
- `multi_lora_demo.py` — drives vLLM with two adapters and measures throughput vs single-LoRA baseline

## Quickstart

```bash
# 1. Train two tiny LoRAs (PEFT + Trainer, 5-10 min each on a small GPU)
#    Save to ./adapters/code and ./adapters/poetry in PEFT format.

# 2. Serve
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --enable-lora --max-loras 4 --max-lora-rank 64 \
    --lora-modules code=./adapters/code poetry=./adapters/poetry

# 3. Drive
pip install openai
python multi_lora_demo.py
```

## Expected output

```
Single-adapter baselines:
  [base (no LoRA)     ]  768 tok in 1.18s = 651 tok/s   TTFT mean 180 ms
  [code LoRA          ]  384 tok in 0.65s = 591 tok/s   TTFT mean 195 ms
  [poetry LoRA        ]  384 tok in 0.62s = 619 tok/s   TTFT mean 188 ms

Mixed batch (this is the multi-LoRA test):
  [interleaved        ]  768 tok in 1.30s = 590 tok/s   TTFT mean 210 ms
```

The interleaved row should be within ~10-20% of the no-LoRA baseline. That's what Punica's batched kernels buy you.

## Try

- **Add a third LoRA.** Confirm memory and throughput scale linearly.
- **Drop `--max-loras` to 1.** Watch the interleaved test's TTFT explode — adapters are now thrashing CPU↔GPU.
- **Increase the rank to 128.** More memory per adapter; throughput should drop slightly.
- **Quality check** — run `lm-eval-harness` on each adapter against the base. The whole point of LoRAs is task quality; verify it.
- **Cache salting** (`extra_keys` in the prefix-cache RFC) — confirm two tenants with the same prompt don't share a cache hit when salts differ.

## Where this goes

- Topic 12 — speculative decoding interacts with multi-LoRA (draft and target both need the same adapter, mostly)
- Level 7 — multi-tenant fairness builds on multi-LoRA: per-tenant LoRAs with per-tenant quotas
