# 02 — Training Job Scheduler

## Files

- `CONCEPTS.md` — why training schedulers exist; the 2026 landscape (Kueue, KubeRay, SLURM, Volcano); minimum viable ops.
- `scheduler.py` — SQLite-backed job table, subprocess execution, PID re-attach on restart, retry with backoff, cancel.

## Quickstart

```bash
python scheduler.py submit "sleep 5 && echo ok"
python scheduler.py list
python scheduler.py loop &      # background reconciler
python scheduler.py submit "sleep 1 && exit 1"   # will FAIL and retry 3x
```

## Expected output

```
$ python scheduler.py list
a1b2c3d4e5f6  RUNNING    pid=44231  retries=0  cmd=sleep 5 && echo ok
b7c8d9e0f1a2  FAILED     pid=44291  retries=3  cmd=sleep 1 && exit 1
```

The retry-3 then FAILED transition is the lesson: deterministic failures hit the cap, get marked dead, stop burning compute.

## Try

- Kill the scheduler mid-job (`pkill -f "scheduler.py loop"`). Restart. Confirm RUNNING jobs whose PIDs are still alive stay RUNNING; jobs whose PIDs died transition correctly.
- Submit a job that creates `/tmp/mini_platform_done_<job_id>` on success. Confirm reconcile marks it DONE rather than FAILED.
- Add a `last_step` column update from inside the trainer (write to a sidecar file the scheduler polls). Now `status` shows progress, and you can detect hangs (no `last_step` change in N minutes).
- Pipe `submit` from your Level 6 trainer config. Confirm a real training run works end-to-end.

## Where this goes

- Topic 03: subscribe to `DONE` transitions, fire `lm-eval-harness` on the checkpoint.
- Topic 04: register the checkpoint with eval scores.
- Topic 16: this same primitive submits RLXF rollout sub-jobs at much higher rate.
