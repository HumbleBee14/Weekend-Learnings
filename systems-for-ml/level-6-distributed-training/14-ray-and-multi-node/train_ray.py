"""
Tiny Ray Train job: 2 workers, 1 GPU each, DDP loop. Demonstrates the
launcher swap from torchrun → Ray.

Run:
    pip install "ray[default,train]" torch
    ray start --head
    python train_ray.py
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def train_func(config: dict) -> None:
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(1024, 4096), nn.GELU(), nn.Linear(4096, 1024)).to(device)
    model = DDP(model, device_ids=[local_rank])
    optim = torch.optim.AdamW(model.parameters(), lr=config["lr"])

    for step in range(config["steps"]):
        x = torch.randn(32, 1024, device=device)
        loss = model(x).pow(2).mean()
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        if rank == 0 and step % 10 == 0:
            print(f"step {step}  loss {loss.item():.4f}")

    dist.destroy_process_group()


def main() -> None:
    import ray
    from ray.train.torch import TorchTrainer, TorchConfig
    from ray.train import ScalingConfig, RunConfig, FailureConfig

    ray.init()
    trainer = TorchTrainer(
        train_func,
        train_loop_config={"lr": 3e-4, "steps": 100},
        scaling_config=ScalingConfig(num_workers=2, use_gpu=True),
        torch_config=TorchConfig(backend="nccl"),
        run_config=RunConfig(
            name="mini-train",
            failure_config=FailureConfig(max_failures=2),
        ),
    )
    result = trainer.fit()
    print(f"finished: {result}")


if __name__ == "__main__":
    main()
