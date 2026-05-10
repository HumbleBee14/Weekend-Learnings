# 11 — Tail Latency and Stragglers

In a synchronous training step, the slowest rank determines step time. Stragglers are the GPUs that periodically run slower than their peers. At scale they dominate the loss curve.

## The math

If each rank has independent probability `p` of being a straggler in a given step, the probability that *some* rank is a straggler is `1 - (1-p)^N`.

```
N = 8     p = 0.001    P(some straggler) = 0.8%
N = 64    p = 0.001    P(some straggler) = 6.2%
N = 1024  p = 0.001    P(some straggler) = 64%
N = 1024  p = 0.01     P(some straggler) = 99.996%
```

At 1024 GPUs with 1% per-step straggler probability per GPU, virtually every step has at least one. The tail dominates.

## Sources of stragglers

- **Thermal throttling**: a GPU under sustained load drops clocks for a few hundred ms. Then recovers. Random timing.
- **ECC retries**: HBM ECC retry on a single read is microseconds; the cumulative effect on memory-bound steps is measurable.
- **Power capping** at the rack level: PSU draws too much, BMC caps power, GPU clocks drop.
- **Network jitter**: rail switch latency variance, congestion from another tenant.
- **Imbalanced workload**: variable-length sequences in a packed batch hit this if not handled (Topic 03's sequence packing).
- **Slow dataloader on one node**: page cache miss, EBS volume hiccup, S3 throttle.
- **Process noise**: GC pause on Python, OS scheduler hiccup, kubelet-induced cgroup throttling.

## Detection

```
per_rank_step_time = [t0, t1, ..., t_{N-1}]
slow_rank = argmax(per_rank_step_time)
slack    = max - mean    # how much waste this step
```

Track `p99 step time / p50 step time` over a window. Steady-state ratio of 1.05 is healthy. 1.3+ is suffering.

The right metric for the platform-level dashboard is **goodput** (Topic 09 + 12). Tail step time degrades goodput linearly.

## Mitigations

### Stay sync, ignore mild

Most stragglers are transient. Don't react to a single slow step. Track a windowed average; act only on persistent slowdowns (≥10 consecutive bad steps from the same rank).

### Drop the rank — NCCL Communicator Shrink

For persistent stragglers (a node that won't recover), drop it. NCCL 2.27's `ncclCommShrink` lets you tear out one rank's communicator membership and continue at smaller world size. Topic 12 covers this end-to-end.

### Rebalance work

If the slowdown is structural (one rank consistently has more work), rebalance:
- DP: give the slow rank a smaller fraction of the global batch
- PP: give the slow stage fewer microbatches by splitting it differently

Frontier-scale systems (DeepMind's Pathways, Google's MaxText) implement this dynamically.

### Asynchronous training (rare in 2026 LLM)

Drop sync. Each rank trains on its own gradient. ASGD-style. Used in some RL-rollout setups (Topic 15) but not in LLM pretraining — the noise hurts convergence too much.

### Backup workers

Train with `N + k` workers; pass when the first `N` finish their step. Wastes hardware but bounds tail. Used at frontier scale where the spare capacity is cheap relative to step-time variance. Reference: [proceedings.mlsys.org/paper/2018/file/8edd72158ccd2a879f79cb2538568fdc-Paper.pdf](https://proceedings.mlsys.org/paper/2018/file/8edd72158ccd2a879f79cb2538568fdc-Paper.pdf) (TF backup-worker paper).

## G11 of Project 3

Plot p50, p95, p99 step time vs straggler severity. Inject stragglers manually:

```python
# rank 0 always slow
if rank == 0:
    time.sleep(slowdown_factor * normal_step_time)
```

Sweep `slowdown_factor` from 0 to 0.5. Plot.

The shape of the curve teaches you: a 10% straggler costs ~10% step time at world=2 (single bad rank dominates), but the same 10% straggler costs less at world=64 (the others overlap their idle waiting). The cost is bounded by `slowdown × p_slow` where `p_slow` is the fraction of time the slow rank is actually slow.

## Build steps

1. Take `ddp_train.py` from Topic 02.
2. Inject `time.sleep(0.05)` after `loss.backward()` on rank 0 only.
3. Measure step time at world_size=2 and 4. Sweep the sleep from 0 to 0.2 sec.
4. Plot p99 step time vs sleep. **G11**.

## Reference

- "Low-priority ICs": [arxiv.org/abs/2305.14456](https://arxiv.org/abs/2305.14456)
- TF backup workers (MLSys 2018): [proceedings.mlsys.org/paper/2018/file/8edd72158ccd2a879f79cb2538568fdc-Paper.pdf](https://proceedings.mlsys.org/paper/2018/file/8edd72158ccd2a879f79cb2538568fdc-Paper.pdf)
- Pathways straggler mitigation: [arxiv.org/abs/2203.12533](https://arxiv.org/abs/2203.12533)
- NCCL Communicator Shrink: [developer.nvidia.com/blog/enabling-fast-inference-and-resilient-training-with-nccl-2-27](https://developer.nvidia.com/blog/enabling-fast-inference-and-resilient-training-with-nccl-2-27/)
