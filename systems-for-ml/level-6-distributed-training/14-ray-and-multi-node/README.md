# 14 — Ray and Multi-Node

## Files

- `CONCEPTS.md` — what Ray Train does, when Ray vs torchrun, KubeRay, failure recovery
- `train_ray.py` — 2-worker DDP via Ray Train (single-node Ray cluster)

## Quickstart

```bash
pip install "ray[default,train]" torch
ray start --head
python train_ray.py
```

## Expected output

```
(TorchTrainer pid=12345)  using worker group: 2 workers, 1 GPU each
(RayTrainWorker pid=12380, ip=127.0.0.1) step  0  loss 1.024
(RayTrainWorker pid=12380, ip=127.0.0.1) step 10  loss 0.832
...
finished: Result(...)
```

## Try

- Open the Ray dashboard: `ray dashboard` (default localhost:8265). Watch the actors during training.
- `max_failures=2` is set — kill one of the workers (`kill -9 <pid>` from the actor list). Ray Train tears down, restarts, and resumes.
- Add `ray.train.report({"loss": loss.item()})` and `ray.train.get_checkpoint()` for Ray Train's built-in checkpointing API.

## Where this goes

- Level 7's `mini-platform` uses KubeRay to schedule both training jobs and inference replicas
- Topic 15 — RL post-training composes Ray Train (training) + Ray Serve (rollout vLLM)
