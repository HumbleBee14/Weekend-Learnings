# 07 — Engine Bake-Off (Project 2)

This is the project for Level 5. The deliverable is `reports/bakeoff.md`.

## Files

- `CONCEPTS.md` — methodology, the four required workloads, what an honest bake-off looks like
- `runner.py` — engine-agnostic OpenAI-client harness; drives multiple engines through the same workloads
- `engines.yaml` — engine list and workload list
- `plot.py` — produces G6, G7, G8 (if memory column added), G9 (with cost YAML)

You'll write next to these:

- `workloads/short.jsonl`, `long.jsonl`, `prefix-heavy.jsonl`, `memory-constrained.jsonl`
- `configs/*.yaml` — per-engine startup configs (server flags, model id, quant)
- `reports/bakeoff.md` — the systems-paper write-up

## Quickstart

Start each engine on its own port (commands per engine in Topics 01-05). Then:

```bash
pip install openai pandas matplotlib pyyaml
python runner.py --engines engines.yaml --out results/
python plot.py --results results/ --out figures/
```

## What `bakeoff.md` must contain

```
1. Problem statement       (1 paragraph)
2. Methodology             (model, hardware, workloads, metrics, warmup, duration)
3. Results                 (G6-G9 + per-scenario tables)
4. Per-engine notes        (when to choose each, install pain, debuggability)
5. Recommendations         (3+ workload-conditional recommendations)
6. Operational notes       (build times, debug, observability, ergonomics)
```

Each finding is quantitative. Each finding has a workload condition. Each finding cites a graph.

## Try

- **Run with default flags first**, then with each engine tuned. Document the delta.
- **Add a quality regression check.** Run `lm-eval-harness` (Level 4 Topic 06) at each engine's chosen quant. If quality drops outside tolerance, the throughput number is moot.
- **Capture GPU memory** during the run (`nvidia-smi --query-gpu=memory.used --format=csv -l 1` in a sidecar) and add to results JSON for G8.
- **Add `mini-vllm`** as the sixth row. You will lose; that's not the point. The point is calibrating where hand-rolled lands vs production.

## Where this goes

- `reports/bakeoff.md` is one of the two heaviest deliverables in the curriculum.
- Level 7's `mini-platform` will route to whichever engine wins for its target workload.
