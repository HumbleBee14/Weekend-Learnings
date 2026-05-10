# 02 — Training Job Scheduler

## Why a training scheduler exists at all

A "training job" is not a function call. It is a long-lived, GPU-greedy, fault-prone process that:

- runs for hours to weeks,
- cannot share its GPUs with other jobs (most of the time),
- must checkpoint or it loses days of work,
- needs to be retried elastically when nodes fail,
- needs its outputs (checkpoints, logs, metrics) routed somewhere downstream.

Generic CI runners and Kubernetes Deployments do none of that. The orchestration that goes around the training process is the *scheduler*.

## The 2026 production landscape

| System | Origin | Where it fits |
|---|---|---|
| **SLURM** | HPC, 1990s | National labs, big-tech-research clusters with InfiniBand |
| **Kueue** | Kubernetes SIG, 2024+ | Quota-aware batch on K8s; pairs with KubeRay or PyTorchJob |
| **KubeRay (RayJob)** | Anyscale | The default for Ray-based training and RLHF; tight integration with vLLM rollouts |
| **Volcano** | CNCF | Gang scheduling on K8s, common in China-stack ML clusters |
| **Internal tools** | every big lab | Slurm-shaped wrapper with their own quota model |

Trend: K8s-native (Kueue + KubeRay or PyTorchJob) is the dominant new-greenfield choice in 2026. SLURM remains entrenched where InfiniBand + bare-metal scheduling matters.

References:
- Kueue — https://kueue.sigs.k8s.io/docs/concepts/
- KubeRay RayJob — https://docs.ray.io/en/latest/cluster/kubernetes/getting-started/rayjob-quick-start.html

## The minimum viable scheduler

Five operations. Any training scheduler in any framework reduces to these:

```
submit(config)        -> job_id, status=PENDING
start(job_id)         -> status=RUNNING, allocate GPUs, spawn process
status(job_id)        -> status, last metrics, last checkpoint
fail(job_id, reason)  -> status=FAILED; optionally retry
finish(job_id)        -> status=DONE; trigger eval (Topic 03)
```

Storage: a table. SQLite is enough for `mini-platform`.

```
jobs
  job_id          TEXT PRIMARY KEY
  config_json     TEXT
  status          TEXT   -- PENDING|RUNNING|FAILED|DONE
  gpus            INT
  pid             INT
  started_at      TIMESTAMP
  finished_at     TIMESTAMP
  last_step       INT
  last_loss       REAL
  checkpoint_path TEXT
  retry_count     INT
```

## Failure handling — the lessons that bite

**1. The scheduler outlives the job.** If your scheduler is a Python process that dies, your jobs become orphans. Always: jobs persist in the DB; the scheduler is reconstructible from the DB. On restart: scan PIDs, re-attach to running processes, mark dead PIDs FAILED.

**2. Retries need exponential backoff and a cap.** Infinite retry on a deterministic OOM burns weeks of GPU time. Default: 3 retries, exponential backoff, cap at 24h.

**3. Checkpoints are the resume primitive.** If a job FAILED with `last_step=14000`, retry should start from the latest checkpoint, not step 0. Level 6 covered async checkpointing; the scheduler reads `checkpoint_path` from the DB and passes it as `--resume`.

**4. Metrics belong in two places.** The training process emits to its own logger (Weights & Biases, MLflow, or `train.log`). The scheduler also reads `last_step / last_loss` periodically — heartbeat-style — so a hung job (no progress in N minutes) is detectable. A "running" status with a 4-hour-old `last_step` is the canonical hung-job signature.

**5. GPU allocation is not the scheduler's job in 2026.** On K8s you delegate to the scheduler plugin (Kueue / Volcano / native scheduler with device plugins). On bare metal you delegate to SLURM. `mini-platform` cheats: it uses `CUDA_VISIBLE_DEVICES` and trusts you not to oversubscribe.

## Connection to the rest of Level 7

- **Topic 03 (eval)** subscribes to `status=DONE` transitions. When a job finishes, eval runs automatically on `checkpoint_path`.
- **Topic 04 (registry)** receives the checkpoint at `DONE`, gives it a version, attaches eval scores.
- **Topic 16 (mini-RLXF)** is the most demanding consumer: it submits *many* short rollout jobs against a vLLM serving the current trainer's weights, with weight sync via NCCL. Same scheduler primitives, much higher submit-rate.

## What "light touch" means here

You are not building SLURM. You are building the smallest thing that exposes the right vocabulary:

- A REST endpoint or CLI: `submit`, `status`, `cancel`, `list`.
- A SQLite-backed job table.
- A worker loop that polls PENDING jobs, spawns subprocesses, monitors them, transitions states.
- One real failure-injection: kill a job mid-run, confirm scheduler marks it FAILED and (optionally) retries from checkpoint.

That is enough to see the architecture. The rest is operational scaffolding that real schedulers (Kueue, KubeRay) provide for free.

## Pitfalls

1. **No persistent DB.** In-memory dict scheduler dies on restart, jobs are orphaned. Always persist.
2. **No PID re-attachment on restart.** Without it, a scheduler restart leaves running jobs untracked forever.
3. **No timeout / heartbeat.** Hung jobs sit in `RUNNING` until you notice manually.
4. **Blocking on subprocess in the API thread.** Submit must return immediately with a `job_id`; the actual subprocess runs on a worker thread or pool.
5. **Coupling the scheduler to a specific trainer.** Take a `command` string, not a `model_name`. The scheduler doesn't care what runs.

## References

- Kueue concepts — https://kueue.sigs.k8s.io/docs/concepts/
- KubeRay RayJob — https://docs.ray.io/en/latest/cluster/kubernetes/getting-started/rayjob-quick-start.html
- SLURM (still the HPC standard) — https://slurm.schedmd.com/documentation.html
- Volcano — https://volcano.sh/en/docs/
