# 03 — Evaluation Pipeline

## Files

- `CONCEPTS.md` — eval as a platform feature, lm-eval-harness with the vLLM backend, the regression gate, what it catches and what it misses.
- `eval_runner.py` — gate evaluator. Reads candidate scores, fetches `serving` scores from the registry, applies rules, writes verdict.
- `gate.yaml` — declarative regression rules.
- `example_scores.json` — sample lm-eval-harness output shape.

## Quickstart

```bash
# 1. Real eval (you do this on a GPU host):
lm_eval --model vllm \
        --model_args pretrained=path/to/checkpoint \
        --tasks mmlu,gsm8k,humaneval \
        --output_path results/myrun.json

# 2. Run the gate (mini-platform-side):
python eval_runner.py gate \
    --candidate-id ckpt_42 \
    --model-name minigpt \
    --scores example_scores.json \
    --gate gate.yaml
```

## Expected output

```json
{
  "verdict": "approved",
  "passed": 3,
  "required": 3,
  "rules": [
    {"benchmark": "mmlu",      "metric": "acc",    "pass": true,  "note": "cand=0.6712 prev=None rule=ge_rel(prev, 0.99)"},
    {"benchmark": "gsm8k",     "metric": "acc",    "pass": true,  "note": "cand=0.5821 prev=None rule=ge_rel(prev, 0.97)"},
    {"benchmark": "humaneval", "metric": "pass@1", "pass": true,  "note": "cand=0.4390 prev=None rule=ge_rel(prev, 0.95)"}
  ]
}
```

(First run has no prev, so all rules trivially pass — that's the documented `prev is None -> True` behaviour.)

## Try

- Manually plant a `serving` row in the registry (Topic 04). Re-run with a candidate that drops MMLU 2 points. Confirm `rejected` with exit code 2.
- Add `--write-status` to push the verdict back into `models.status`. This is the actual control-plane wiring.
- Add a fourth rule using `le_rel(prev, 1.05)` to flag suspicious score *jumps* (often a contamination signature).
- Pipe this from Topic 02's scheduler: when a job hits DONE, submit an eval job; when eval finishes, submit `eval_runner.py gate`.

## Where this goes

- Topic 04: registry consumes the `approved` / `rejected` verdict; only `approved` checkpoints can be promoted to `serving`.
- The gate is **G14's** break-it list entry "model regression — same engine, new checkpoint that's 5% worse on lm-eval-harness — verify regression gate blocks deploy".
