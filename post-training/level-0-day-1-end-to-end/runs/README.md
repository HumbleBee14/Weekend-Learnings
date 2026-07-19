# runs/ — curated, shareable run evidence

`data/`, `out/`, and `reports/` are **gitignored scratch** — regenerated on every run.
This folder is the opposite: **small, hand-picked artifacts from runs worth showing** — proof of what actually happened, for anyone reading the repo without rerunning it.

Convention:

- One folder per noteworthy run: `YYYY-MM-DD-<backend>-<what>/`
- Inside: the raw log (`training-log.txt`), a `RESULTS.md` (Setup → Observation → Insight), and nothing heavy — no model weights, no datasets, keep it a few KB.
- Add a run only when it demonstrates something (a first, a delta, a failure worth keeping). Not every run.

| Run | What it shows |
|---|---|
| [`2026-07-19-mlx-day0/`](2026-07-19-mlx-day0/) | First verified end-to-end Day 0 run (MLX, Mac): baseline exact-match 0.133 → 1.000 after a ~40s LoRA SFT |
