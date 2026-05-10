# 14 — Ray and Multi-Node

Job orchestration when you can't (or don't want to) use SLURM. Ray Train wraps PyTorch DDP/FSDP for cluster-scheduled training jobs. KubeRay is the cloud-native pattern. The bits here are the orchestration story — not new ML, just the layer above `torchrun`.

## What Ray Train does

`torchrun` solves "launch N processes on this host with the right env vars." It does not solve:
- Which nodes does the job run on?
- What if a node is unhealthy?
- How does training share a cluster with inference, RL rollouts, data prep?
- How do I queue 50 hyperparameter trials?

Ray's job model handles all of this. Ray Train (specifically) wraps the per-process PyTorch logic and Ray's scheduler handles placement.

```python
from ray.train.torch import TorchTrainer, TorchConfig
from ray.train import ScalingConfig

def train_func(config: dict) -> None:
    # This function runs on each worker. Ray sets RANK / WORLD_SIZE / LOCAL_RANK
    # in the env before calling. dist.init_process_group works as usual.
    import torch.distributed as dist
    dist.init_process_group(backend="nccl")
    # ... your normal training code ...

trainer = TorchTrainer(
    train_func,
    train_loop_config={"lr": 3e-4, "batch_size": 8},
    scaling_config=ScalingConfig(num_workers=4, use_gpu=True),
    torch_config=TorchConfig(backend="nccl"),
)
result = trainer.fit()
```

The `ScalingConfig(num_workers=4, use_gpu=True)` says "give me 4 workers, each with one GPU." Ray's scheduler picks where they run. Process spawning, env-var injection, failure recovery, and result collection happen in Ray's runtime.

## When Ray vs `torchrun`

- **Single host, ad-hoc**: `torchrun`. Don't reach for Ray for a 2-GPU laptop run.
- **Multi-host, static**: `torchrun` with `--rdzv` works fine if you control all the hosts. SLURM also works.
- **Multi-host, dynamic, mixed workloads**: Ray. KubeRay if you're on K8s.
- **Hyperparameter tuning, RLHF rollouts, multi-tenant**: Ray. The actor model handles these naturally.

## Ray Cluster vs single-machine Ray

```bash
# single-machine Ray cluster (everything on localhost)
ray start --head
# now Ray can schedule jobs on your local resources

# multi-node Ray cluster
# on head node:
ray start --head --port=6379
# on worker nodes:
ray start --address='head:6379'
```

`ray status` shows the cluster's resources. `ray submit` runs a job. `ray dashboard` (default port 8265) is the web UI.

## KubeRay

[KubeRay](https://github.com/ray-project/kuberay) is the operator for running Ray on Kubernetes. The cluster-native pattern in 2026:

```yaml
apiVersion: ray.io/v1
kind: RayCluster
metadata:
  name: training-cluster
spec:
  headGroupSpec: { ... }
  workerGroupSpecs:
  - replicas: 8
    template:
      spec:
        containers:
        - resources:
            limits:
              nvidia.com/gpu: 8
```

Then `RayJob` resources submit training jobs to the cluster. The K8s scheduler handles preemption, node failures, autoscaling. Level 7's `mini-platform` uses this pattern.

## Failure recovery in Ray Train

`TorchTrainer` has a `RunConfig.failure_config` for retries. When a worker dies:
1. Ray detects the actor failure.
2. Trainer cancels remaining workers, releases their resources.
3. Up to `max_failures` times, it relaunches the whole worker set.
4. Each new worker re-rendezvous, loads from the latest Ray-Train checkpoint, resumes.

Compared to `torchrun --rdzv` (Topic 12), Ray adds:
- Cluster-aware restart (re-pick nodes if any are now unhealthy).
- Ray Train's checkpointing API (built on torch DCP).
- A scheduler that can preempt other Ray jobs to free resources for the restart.

## Build steps (light)

```python
import ray
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig
import torch.nn as nn
import torch.distributed as dist


def train_func(config):
    dist.init_process_group(backend="nccl")
    # ... normal DDP/FSDP loop ...
    dist.destroy_process_group()


if __name__ == "__main__":
    ray.init()
    trainer = TorchTrainer(
        train_func,
        train_loop_config={"steps": 100},
        scaling_config=ScalingConfig(num_workers=2, use_gpu=True),
    )
    trainer.fit()
```

Run with `python train_ray.py`. Behind the scenes Ray spawns 2 worker actors, sets up env vars, calls your function. Same training code as `torchrun`, different launcher.

## Reference

- Ray Train: [docs.ray.io/en/latest/train/train.html](https://docs.ray.io/en/latest/train/train.html)
- KubeRay: [github.com/ray-project/kuberay](https://github.com/ray-project/kuberay)
- Ray Train fault tolerance: [docs.ray.io/en/latest/train/user-guides/fault-tolerance.html](https://docs.ray.io/en/latest/train/user-guides/fault-tolerance.html)
- Anyscale Llama-3 training blog: [anyscale.com/blog/training-llama-3-with-ray-and-pytorch](https://www.anyscale.com/blog/)
