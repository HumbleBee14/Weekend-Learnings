# Level 5 — Eval Depth, Data & the Full Pipeline

> Track map: [`post-training/README.md`](../README.md) · Primary read: [RLHF Book](https://rlhfbook.com) (evaluation) + [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness) docs
>
> Goal: learn to judge a model *honestly*, curate data deliberately, and reproduce the entire SFT → DPO → GRPO pipeline end-to-end into one written report.

## The WHY

Levels 0–4 used a lightweight eval so you could move fast. Level 5 is where eval grows up — because at this point the interesting failures are *eval* failures: a benchmark that leaked into training, a judge that prefers longer answers, an RL model that games the reward. **You cannot trust a post-training result you cannot measure honestly**, and honest measurement is a skill, not a script.

## Where this fits

- **Comes after:** Level 4 — you now have three checkpoints (SFT, DPO, GRPO) to compare.
- **Rejoins:** `systems-for-ml/` L7 `mini-rlxf` — this track is the *methods depth*; that topic is the *platform orchestration* of the same pipeline.

## Topics

| # | Topic | What you learn / build |
|---|-------|------------------------|
| 01 | llm-as-judge | Alpaca Eval / MT-Bench / Arena-Hard; judge biases (length, position, self-preference) and how to blunt them |
| 02 | static-benchmark-suites | Running `lm-eval-harness` / `lighteval`; IFEval, MMLU-Pro, GPQA — what each actually tests |
| 03 | contamination-and-goodhart | Detecting benchmark leakage; why chasing a metric corrupts it; time-windowed benchmarks |
| 04 | data-curation | Quality > quantity; dedup, decontamination, data mixing; why data is the real product |
| 05 | full-pipeline-reproduced | One script: base → SFT → DPO → GRPO, end-to-end, cloud + Mac |
| 06 | report-and-handoff | Write the systems-paper-style report; hand the pipeline to `systems-for-ml` L7 |

## What "done" looks like

- A single reproducible run producing all three checkpoints and one comparison table.
- A `reports/` writeup: Problem → Architecture → Experiments → Findings → Tradeoffs, with the SFT-vs-DPO-vs-GRPO deltas as numbered findings.
- A clear statement of which method was worth its complexity *for this task* — the judgment the whole track exists to build.

## Eval checkpoint

This level *is* the eval checkpoint. The deliverable is the honest, contamination-aware comparison the earlier levels were building toward.

## Teach-back

Final teach-back for the track: explain to someone with zero background what SFT, DPO, and GRPO each did to your model, and when you'd reach for each. That explanation is the point of the whole track.
