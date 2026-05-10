"""
Deliberately hang NCCL by mismatching shapes across ranks.

Run with watchdog timeout so you see the FlightRecorder dump:

    NCCL_DEBUG=INFO \
    TORCH_NCCL_TRACE_BUFFER_SIZE=2000 \
    TORCH_NCCL_DUMP_ON_TIMEOUT=1 \
    TORCH_NCCL_DEBUG_INFO_TEMP_FILE=/tmp/nccl_trace \
    torchrun --nproc_per_node=2 hang_demo.py

The collective never returns. Read /tmp/nccl_trace_<rank> after timeout to
see which rank queued which collective.
"""

import torch
import torch.distributed as dist


def main() -> None:
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # Mismatched shapes: rank 0 has (8,), rank 1 has (16,). NCCL hangs.
    if rank == 0:
        t = torch.ones(8, device=device)
    else:
        t = torch.ones(16, device=device)

    print(f"rank{rank} entering all_reduce with shape {tuple(t.shape)}")
    dist.all_reduce(t)
    print(f"rank{rank} returned (this line will not print)")


if __name__ == "__main__":
    main()
