# 10 — Autoscaling (KEDA)

## Why HPA-on-CPU loses for LLM serving

CPU-bound web services scale on CPU because CPU is the saturation signal. LLM inference is GPU-bound; CPU stays around 30-50% even at full GPU saturation. By the time a CPU-based HPA fires (cooldown + threshold + smoothing), the GPU queue is already deep and TTFT has long since broken SLO.

The 2026 default is **KEDA** scaling on `vllm:num_requests_waiting`. KEDA (Kubernetes Event-Driven Autoscaling) is the CNCF graduated project that lets HPA consume custom metrics from any source — Prometheus, Kafka, RabbitMQ, etc. For LLM serving, the source is Prometheus and the metric is queue depth.

References:
- KEDA — https://keda.sh/
- KEDA + vLLM autoscaling tutorial — https://docs.vllm.ai/projects/production-stack/en/latest/use_cases/autoscaling-keda.html

## The canonical ScaledObject

```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: vllm-scaler
spec:
  scaleTargetRef:
    name: vllm-deployment
  minReplicaCount: 1            # never scale to zero unless cold-start is solved
  maxReplicaCount: 8
  pollingInterval: 5            # seconds; KEDA polls Prometheus
  cooldownPeriod: 60            # seconds before scale-down after metric drops
  triggers:
  - type: prometheus
    metadata:
      serverAddress: http://prometheus:9090
      query: avg(vllm:num_requests_waiting)
      threshold: "3"            # average waiting per replica
      activationThreshold: "1"  # below this, scale to minReplicas
```

`threshold: 3` means: if the average queue depth across replicas exceeds 3, KEDA asks HPA to add replicas. Tuning is empirical — too low and you over-provision; too high and TTFT breaks before the new replica is ready.

## What "scale-up reaction time" actually means

```
arrival surge starts          T0
queue crosses threshold       T0 + ε
KEDA polls Prometheus         T0 + ε + (up to pollingInterval)
HPA reconciles                + ~15s (HPA period)
new pod scheduled             + scheduler latency (~few s)
image pulled                  + image-pull time (10s-2min)
container starts              + a few s
weights load                  + 30-120s for 70B (Topic 11)
CUDA context init             + a few s
warmup + first-token          + 1-5s
```

Total reaction time is dominated by the bottom three lines: weights load + CUDA init + warmup. This is why **scale-up alone is not enough** — by the time the new replica is ready, the SLO breach is already several minutes old. Topic 11 covers the mitigations (pre-warmed pools, model streamer, CUDA-checkpoint restores).

## Multi-signal autoscaling — the 2026 frontier

A single signal misses cases:

- **Queue depth + KV-pressure.** A pod can have a low queue but be holding 95% of its KV blocks; one more long-context request OOMs it. Scale up before that happens.
- **Queue depth + arrival-rate prediction.** Shock detection: a sudden derivative spike triggers scale-up before queue actually fills.
- **Per-tenant queues.** Enterprise tier has its own queue; a separate ScaledObject scales the dedicated worker pool.

llm-d's **Variant Autoscaler** combines these. NVIDIA Dynamo's **SLO Planner** does the same with deeper SLO/cost optimisation. For `mini-platform`, single-signal queue-depth KEDA is the right starting point.

## Scale-down is harder than scale-up

Scaling down a pod with in-flight requests is destructive. The right pattern:

1. KEDA / HPA decides to remove a replica.
2. The replica is marked **unschedulable** (drain/cordon) — router stops sending new traffic.
3. In-flight requests finish or are migrated.
4. Pod terminates.

K8s `terminationGracePeriodSeconds` covers (3) — set it to your p99 e2e latency × 2, plus a margin. vLLM's graceful shutdown handles in-flight requests if you use `SIGTERM` correctly.

For prefix-locality-heavy workloads, scaling a pod down loses its KV cache. The next request needing that prefix re-prefills somewhere else. LMCache's persistent tier (Topic 12) softens this — the prefix lives in DRAM/NVMe/Redis after the pod dies.

## The scale-up-vs-SLA-breach race

Even with KEDA tuned aggressively, scale-up takes seconds-to-minutes. So:

- **Always run with `minReplicas` >= 1.** Scale-to-zero is a fantasy without aggressive cold-start mitigation.
- **Provision for steady-state with headroom.** If you size for exactly the mean QPS, every burst breaches. A 30-50% headroom is normal.
- **Pre-warm above expected diurnal peak.** Predicted scale (cron-driven `minReplicas`) for known traffic patterns; reactive scale for surprises.

KEDA supports cron triggers natively for predicted scale; combine with a Prometheus trigger for reactive scale.

## Variant Autoscaler — what it is

llm-d ships a custom autoscaler that picks not just *how many replicas*, but *what kind of replica*. Variants:
- prefill-heavy worker (more compute, lighter KV).
- decode-heavy worker (more KV, less compute).
- different quantisation tiers.

Decision input: workload composition (prefill-bound vs decode-bound seen in metrics). Decision output: a mix of replica types. This is the production-grade answer for disaggregated serving (prefill/decode split).

Reference: https://llm-d.ai/docs/architecture

## Build steps

1. Install KEDA in `kind` or `minikube`.
   ```bash
   helm repo add kedacore https://kedacore.github.io/charts
   helm install keda kedacore/keda --namespace keda --create-namespace
   ```
2. Stand up Prometheus + a vLLM Deployment scraping `/metrics`.
3. Apply the ScaledObject manifest in this folder.
4. Drive load past saturation. Watch replicas scale up.
5. Drop load. Watch scale-down (after `cooldownPeriod`).
6. Time the **scale-up reaction window** (load surge -> SLO breach -> new replica ready). This is **G15**.

## Pitfalls

1. **Scaling on `num_requests_running` instead of `num_requests_waiting`.** Running is always near max once batches are full; it doesn't tell you queue depth. Always `waiting`.
2. **Polling interval too long.** 30s polling means up to 30s of detection lag on top of HPA's own delays. 5s is a good default.
3. **Cooldown too short.** Flapping. 60-120s is reasonable.
4. **No `minReplicas`.** Scale-to-zero burns the first request after idle.
5. **Forgetting graceful drain on scale-down.** In-flight requests die. Set `terminationGracePeriodSeconds` and use a `preStop` hook.
6. **Same KEDA policy on free vs enterprise pools.** Different tiers have different SLOs and different scale-up costs. Separate ScaledObjects.

## References

- KEDA — https://keda.sh/docs/
- KEDA Prometheus scaler — https://keda.sh/docs/latest/scalers/prometheus/
- KEDA + vLLM tutorial — https://docs.vllm.ai/projects/production-stack/en/latest/use_cases/autoscaling-keda.html
- llm-d Variant Autoscaler — https://llm-d.ai/docs/architecture
- NVIDIA Dynamo SLO Planner — https://developer.nvidia.com/blog/nvidia-dynamo-1-production-ready/
