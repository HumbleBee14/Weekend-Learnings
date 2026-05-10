# 13 — Speculative Decoding

## Files

- `CONCEPTS.md` — what spec decode is, three method families (draft model, EAGLE, n-gram), tree-spec verification, P-EAGLE (Feb 2026), acceptance rate
- `measure_spec_decode.py` — vLLM offline API comparison: baseline vs n-gram vs (optional) EAGLE-3

## Quickstart

```bash
pip install vllm
python measure_spec_decode.py
```

## Expected output

```
config                          time     tokens     throughput
─────────────────────────────────────────────────────────────────
baseline                        4.21s     1024     243.2 tok/s
n-gram (k=5)                    2.78s     1024     368.2 tok/s   ← 1.51× win on code
```

n-gram does well on code because repetitive structure makes prompt-lookup matches frequent. On free-form chat, n-gram is closer to baseline; you'd reach for EAGLE-3 or P-EAGLE there.

## Try

- **Switch the prompts to free-form chat** — n-gram's win shrinks.
- **Increase `num_speculative_tokens` to 8 or 10** — higher K helps when acceptance is high; hurts when it's low.
- **Enable EAGLE-3** if you have a model with a pre-trained head (see https://huggingface.co/yuhuili/EAGLE-LLaMA3.1-Instruct-8B).
- **Measure quality** — `lm-eval-harness` HumanEval before and after spec decode. Should be identical (modulo seed differences); if not, there's a bug in the implementation.
- **Find the prompts where spec decode loses** — hard reasoning / math problems. Acceptance rate drops, overhead exceeds the win, throughput goes *down*.

## What you should walk away with

- Working n-gram spec decode on a real workload, with measured throughput improvement
- Understanding of acceptance rate as the key metric
- Awareness of EAGLE-3 and P-EAGLE for when n-gram isn't enough
- The systems-side knowledge that spec decode interacts subtly with continuous batching and KV cache (Topic 17)

## Where this goes

- Topic 14 — continuous batching (the system that schedules these forward passes)
- Topic 17 — spec decode systems integration (scheduler, rollback, tree-spec verification)
