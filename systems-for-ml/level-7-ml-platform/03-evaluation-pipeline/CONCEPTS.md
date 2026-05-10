# 03 — Evaluation Pipeline

## Eval as a first-class platform feature, not a script

The 2024 default was "the researcher runs eval on a notebook before pushing". That doesn't scale and it doesn't gate anything. The 2026 default is:

```
checkpoint DONE  ->  eval auto-runs  ->  scores written to registry
                                          │
                                          ▼
                                  regression gate decides
                                  approved  /  rejected
```

The gate is the point. Without an automatic block on regressions, the only thing protecting production is the trainer's own discipline. With the gate, deploys cannot regress more than X% on Y benchmarks.

## What runs

`lm-eval-harness` (EleutherAI) is the de-facto open evaluator. As of 2026 it covers:

- MMLU, MMLU-Pro, MMLU-Redux
- GSM8K, MATH
- HumanEval, HumanEval+, MBPP
- HellaSwag, ARC, TruthfulQA, IFEval
- AGIEval, GPQA
- multilingual: MGSM, XNLI

For LLM-as-a-judge style metrics (LMSYS Arena-Hard, MT-Bench), separate runners exist; they are slower and require a strong judge model.

References:
- lm-evaluation-harness — https://github.com/EleutherAI/lm-evaluation-harness
- HELM — https://crfm.stanford.edu/helm/
- vLLM-backed eval (huge speedup vs HF) — pass `--model vllm` to lm-eval-harness

## How it runs at scale

Two patterns:

**1. Same-host eval.** When the trainer node has GPUs idle (it usually does between epochs / after final checkpoint), launch eval in-place. Cheap, but contends with training if you start eval while training is still going.

**2. Eval pool.** A separate set of GPUs (or a queue on the same cluster) just for eval. The scheduler from Topic 02 handles it: when a training job hits DONE, it submits an eval job pinned to the eval pool. Production-shape; what every real platform does.

**vLLM as the eval backend.** `lm-eval-harness --model vllm` runs the same prompts ~5-10x faster than the HF backend, especially on multi-choice tasks. Fewer wasted GPU-minutes, eval runs more often, gates fire more reliably. Default in 2026.

## The regression gate

A rule. Encode it as data, not code-in-the-CI-script.

```
gate.yaml
  - benchmark: mmlu
    metric: acc
    rule: ge_rel(prev, 0.99)        # within 1% of previous serving
  - benchmark: gsm8k
    metric: acc
    rule: ge_rel(prev, 0.97)
  - benchmark: humaneval
    metric: pass@1
    rule: ge_rel(prev, 0.95)
  required_pass: 3                  # all three must pass to approve
```

Gate evaluator reads the candidate's scores from the registry, the previous-serving's scores from the registry, and decides `approved` or `rejected`. Result is written back to the registry as a status transition (Topic 04).

The gate is **not** the only signal; it is the last automated one. Above it sits the human for high-impact promotions. Below it sit unit tests, smoke tests, dataset-leak checks. The gate is the floor.

## Failure modes the gate catches

1. **Quiet regression** — a code change breaks tokenisation; MMLU drops 4 points; the trainer didn't notice. Gate fires.
2. **Quantisation-introduced drift** — FP8 build of a checkpoint silently lost 2 GSM8K points. Gate fires (assuming you re-eval after quantisation, which you must — Topic 02 quality-evaluation in Level 4).
3. **Dataset contamination** — sometimes appears as a *score increase*. The gate is symmetric: optional rule `le_rel(prev, 1.05)` flags suspicious gains for human review. Default off; turn on for high-stakes models.

## Failure modes the gate misses

- **Distribution-shift in deployment.** Benchmarks measure benchmark performance. Real traffic differs.
- **LLM-as-a-judge bias.** Arena-Hard and MT-Bench depend on a judge that has its own preferences. A new model can pattern-match the judge better and "win" without genuine improvement.
- **Long-context degradation.** Most public benchmarks cap at 4K-32K tokens. A 1M-context regression past that won't show up.

Mitigations: an internal eval set built from real logs, plus production canary measurement (small % of traffic, compare on tasks where you have ground truth or replay).

## Build sequence for `mini-platform`

```
1. Trainer finishes -> Topic 02 scheduler marks DONE
2. Eval scheduler (cron / event listener) sees DONE
3. Submit eval job: `lm-eval-harness --model vllm --tasks mmlu,gsm8k,humaneval --batch_size auto`
4. Eval job writes JSON scores to `eval/results/<job_id>.json`
5. Registrar reads scores, attaches to checkpoint row, transitions `staged -> eval -> approved/rejected`
6. Approved checkpoints become candidates for Topic 04's promotion to `serving`
```

## Pitfalls

1. **No baseline, no rule.** First model has no `prev`. Gate must allow first-time pass-through.
2. **Comparing across different harness versions.** lm-eval-harness scoring has changed across versions. Pin the version in your gate config and re-baseline when you bump it.
3. **Few-shot count drift.** MMLU 5-shot vs 0-shot scores differ wildly. Lock the prompt template.
4. **Sample size noise.** GSM8K has ~1.3K test items; ±0.5% is noise. Set the rule's tolerance above the noise floor (or use confidence intervals).
5. **Skipping eval for "small" changes.** Adapters and LoRA fine-tunes also need the gate. Especially adapters — they are the easiest regression source in 2026 multi-LoRA serving.

## References

- lm-evaluation-harness — https://github.com/EleutherAI/lm-evaluation-harness
- vLLM as eval backend (lm-eval-harness docs) — https://github.com/EleutherAI/lm-evaluation-harness#vllm
- HELM — https://crfm.stanford.edu/helm/
- LMSYS Arena-Hard — https://github.com/lmarena/arena-hard-auto
