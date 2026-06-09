# Prompt 07 — Distributed Training Platform for a 70B Model

You're the platform lead at a research lab. The team is about to start pre-training a 70B model from scratch. Budget: 256 H100s for ~6 weeks.

Requirements:
- Achieve **>50% MFU** (model FLOPs utilization)
- **Goodput target: >85%** (account for node failures, restarts, stragglers)
- **Checkpoint every 30 minutes** without stalling training
- **Tolerate single-node failures** without restarting from scratch
- Data pipeline must **not bottleneck the GPUs** — 256 H100s eating tokens needs serious throughput
- Researchers will iterate on the data mix; pipeline needs to be re-runnable

Design the training platform. This is a training problem, not an inference problem.
