# 11 — Cold Start and Warmup

## The cold-start budget you cannot escape (and can shrink)

A cold-start replica goes through this sequence. Numbers below are typical for a 70B model on H100 with default tooling:

| Phase | Time | What's happening |
|---|---|---|
| Pod schedule + image pull (cached) | 5-15s | K8s places the pod, image already on node |
| Pod schedule + image pull (cold) | 60-180s | First-time image pull from a registry |
| Container start, Python import | 5-10s | torch + vllm import, CUDA driver attach |
| CUDA context init | 2-5s | per device |
| Weights load (`.safetensors`, naive) | 60-180s | single-threaded HF loader, host->device copy |
| CUDA graph capture / kernel autotune | 10-30s | first warmup pass, cuBLAS autotune |
| Warmup requests (TTFT stabilises) | 1-5s | engine actually emits first token |

Total naive: 2-7 minutes. With every mitigation stacked: under 30s. Sub-second is achievable for *restore-from-snapshot* paths.

The full cold-start budget is the wall between "scale-to-zero" (a fantasy without mitigation) and "always warm pool of N replicas" (the reality).

## The 2026 stack of mitigations (combine, don't pick one)

### Pre-warmed pools (`minReplicas >= 1`)

The cheapest and most common. KEDA never lets the pool empty. New traffic always lands on a warm replica. Cost: `minReplicas * GPU_$/hr` of idle. Worth it for any production-shape workload.

### NVIDIA Run:ai Model Streamer

Concurrent streaming of `safetensors` directly to GPU memory, bypassing the single-threaded HF loader. Reported: **40-60% faster cold starts** on 70B-class models. Native vLLM integration since 2025.

```python
# vLLM args
--load-format runai_streamer
--model-loader-extra-config '{"concurrency": 16, "memory_limit": 80_000_000_000}'
```

The trick: the loader fans out file-range reads in parallel and pipes directly into a CUDA stream. The naive loader allocates host memory, copies, then transfers — Run:ai skips the host detour for most of the bytes.

Reference: https://developer.nvidia.com/blog/reducing-cold-start-latency-for-llm-inference-with-nvidia-runai-model-streamer/

### Tensorizer (CoreWeave)

Same idea, different format. `.tensors` files designed for streaming. Used most often when you want the file format itself to be streaming-friendly. vLLM supports loading from tensorizer with `--load-format tensorizer`.

Reference: https://github.com/coreweave/tensorizer

### ServerlessLLM-style tiered preloading

Stage weights through underutilised GPU memory + local NVMe. Reported: **6-8x speedup** over naive load. The pattern: warm replicas hold partial weights of *neighbouring* models in spare HBM; cold-starting a neighbour pulls those parts from peer HBM (NIXL / NVLink) rather than from disk.

Production examples: this is essentially what hyperscale serverless LLM platforms do internally. For mini-platform it is overkill.

Reference: https://www.usenix.org/conference/osdi24/presentation/fu

### CRIU / CUDA checkpoint

Snapshot a fully-warm process — CUDA context, weights resident in HBM, compiled graphs in RAM — and restore. **Sub-second restores** are achievable. Active vLLM forum experiments through 2026.

The key 2026 progress: NVIDIA's `cuda-checkpoint` and CRIU's CUDA support cooperate to dump GPU state alongside CPU state. Constraints: same GPU model, same CUDA version, same NCCL topology.

Reference: https://github.com/NVIDIA/cuda-checkpoint

### Warmup requests after weights load

After weights are resident, the first real request still pays for CUDA graph capture, cuBLAS autotuning, and KV pool initialization. Fix: synthetic warmup at startup. vLLM does this automatically; do it manually for any custom engine.

```python
# Synthetic warmup loop
for input_len, output_len in [(64, 8), (1024, 8), (8192, 8)]:
    engine.generate(make_synthetic_prompt(input_len), max_tokens=output_len)
```

This triggers graph capture at multiple input shapes, which prevents shape-change recompilation later.

### Image pre-pull (DaemonSet)

A DaemonSet on every node that `docker pulls` the inference image at boot. New pods on that node get image cache hits (5-15s instead of 60-180s).

```yaml
# Sketch
apiVersion: apps/v1
kind: DaemonSet
spec:
  template:
    spec:
      initContainers:
        - name: prepull
          image: vllm/vllm-openai:v0.11.0
          command: ["echo", "image cached"]
```

Trivial, hugely effective.

### Weight cache on local NVMe

Mount node-local NVMe to a path under `/var/lib/llm-cache`. Image's `/models` dir is a hostPath into that. New pods read weights from local NVMe (3-5 GB/s) instead of S3 / object storage (200 MB-1 GB/s). Pair with a CronJob that pre-fetches expected models.

## GPU fractional sharing (MIG, MPS, time-slicing) — when each is right

Cold start is faster on smaller GPUs that can host smaller models. Sometimes the right answer is fractional GPUs:

| Mechanism | What it is | When it's right |
|---|---|---|
| **MIG (Multi-Instance GPU)** | Hardware-partitioned GPU into up to 7 isolated slices | Strong isolation, predictable performance, multi-tenant; A100/H100 only |
| **MPS (Multi-Process Service)** | Multiple processes share one GPU context | Lower isolation, higher utilisation, noisy neighbours possible |
| **CUDA time-slicing** | K8s device plugin time-shares a GPU across pods | Dev/CI, never production; no isolation |

For LLM serving, MIG is the production default for multi-tenant cold-start-friendly deployments. A 70B model needs a full A100/H100. A 7B model fits in a 1g.10gb MIG slice and starts much faster.

References:
- NVIDIA MIG — https://docs.nvidia.com/datacenter/tesla/mig-user-guide/
- MPS — https://docs.nvidia.com/deploy/mps/

## RDMA / GPUDirect for model load

In multi-node setups, the cold-start path can pull weights from peer-GPU memory or remote GPU-DirectStorage targets via RDMA. NIXL (Topic 12) generalises this to KV transfer, but the same primitives load weights at >100 GB/s when configured. This is how frontier labs do sub-30s 70B starts.

## Cold-start during peak load — the worst-case test

The hardest scenario: a load spike at peak hour triggers KEDA, but every mitigation depends on something that's also under stress (image registry, NVMe IO, Prometheus scraping). The "cold start during peak" test is **G15**:

1. Pin one replica to a host. Run sustained peak load.
2. Force a scale-up by cordoning one node, breaking another, or just driving past `threshold`.
3. Capture wall-clock from "KEDA fires" to "new replica's first served request" to "SLO restored".
4. Stack the mitigations in order; re-measure. The deltas are the lessons.

## Build steps

1. Measure naive cold start (no mitigations): vanilla vLLM + naive HF loader.
2. Add `--load-format runai_streamer`. Re-measure.
3. Add image pre-pull DaemonSet. Re-measure.
4. Add a warm pool of 1 replica. Re-measure scale-up reaction time during peak.
5. Plot: cold-vs-warm request latency with annotated scale-up window. **G15.**

## Pitfalls

1. **Naive HF loader as the floor.** Single-threaded; CPU-bound; bottlenecked by host->device copy. Always switch loader.
2. **Scale-to-zero on greedy autoscalers.** First request after idle pays minutes. Always `minReplicas >= 1`.
3. **Forgetting CUDA graph capture.** First few requests are 5-30% slower than steady-state until graphs are captured.
4. **Image not pre-pulled.** A 5GB image pulls in ~30s on cold cluster, and that 30s is *added* to the cold-start budget.
5. **MIG vs MPS confusion.** MIG is hardware-partitioned (real isolation). MPS is shared-context (cooperative). Use MIG when SLOs are tier-bound; MPS only for trusted neighbour processes.
6. **Skipping warmup requests.** Engine is "ready" but first user request pays autotune.

## References

- NVIDIA Run:ai Model Streamer — https://developer.nvidia.com/blog/reducing-cold-start-latency-for-llm-inference-with-nvidia-runai-model-streamer/
- Tensorizer — https://github.com/coreweave/tensorizer
- ServerlessLLM (OSDI '24) — https://www.usenix.org/conference/osdi24/presentation/fu
- NVIDIA cuda-checkpoint — https://github.com/NVIDIA/cuda-checkpoint
- NVIDIA MIG — https://docs.nvidia.com/datacenter/tesla/mig-user-guide/
- NVIDIA MPS — https://docs.nvidia.com/deploy/mps/
