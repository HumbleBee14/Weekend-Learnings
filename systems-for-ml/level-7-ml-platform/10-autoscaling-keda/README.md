# 10 — Autoscaling (KEDA)

## Files

- `CONCEPTS.md` — why HPA-on-CPU fails, the canonical ScaledObject, scale-up reaction-time anatomy, multi-signal autoscaling (queue + KV), Variant Autoscaler.
- `keda-scaledobject.yaml` — production-shaped manifest with three triggers (queue depth, KV pressure, cron pre-scale) and tuned scale-up/down behavior.
- `vllm-deployment.yaml` — vLLM Deployment with the metrics endpoint, graceful drain hook, and reasonable resource requests.

## Quickstart

```bash
# 1. Local cluster.
kind create cluster --name ml-platform

# 2. KEDA.
helm repo add kedacore https://kedacore.github.io/charts
helm install keda kedacore/keda -n keda --create-namespace

# 3. Prometheus (the Topic 05 stack also works; on K8s use kube-prometheus-stack).
helm install prom prometheus-community/kube-prometheus-stack \
    -n observability --create-namespace

# 4. Deploy.
kubectl create ns ml-platform
kubectl apply -f vllm-deployment.yaml
kubectl apply -f keda-scaledobject.yaml

# 5. Drive load and watch:
kubectl get hpa -n ml-platform -w
```

## Expected output

```
NAME                     REFERENCE                    TARGETS         MINPODS   MAXPODS   REPLICAS
keda-hpa-vllm-scaler     Deployment/vllm-deployment   2/3 (avg)       1         8         1
keda-hpa-vllm-scaler     Deployment/vllm-deployment   7/3 (avg)       1         8         3   <- scale up fired
keda-hpa-vllm-scaler     Deployment/vllm-deployment   1/3 (avg)       1         8         3
keda-hpa-vllm-scaler     Deployment/vllm-deployment   0/3 (avg)       1         8         1   <- after cooldown
```

## Try

- **Tune the threshold.** Drop to `"1"`. Replicas grow eagerly; over-provisioning waste shows up in your Topic 13 cost dashboard. Raise to `"10"`. SLO breaches widen.
- **Single-signal vs multi-signal.** Comment out the `vllm:gpu_cache_usage_perc` trigger. Drive a long-context workload until KV pressure pegs. Notice how queue depth alone undercounts — pods are about to OOM but waiting count is still small.
- **Scale-down race.** Drop traffic to zero. Time termination of an extra pod that has in-flight decodes. Verify `preStop` sleep + `terminationGracePeriodSeconds` actually let the request complete.
- **G15.** Drive a sudden load surge. Stamp wall-clock for: surge start, queue threshold cross, new pod scheduled, new pod ready, SLO breach starts, SLO restored. Plot all five timestamps over the TTFT timeline.

## Where this goes

- Topic 11: scale-up reaction time is dominated by cold start. Mitigations live there.
- Topic 13: KEDA's threshold choice is a $/Mtok lever — over-provisioning is dollars.
- Topic 15: reasoning-aware serving uses `time_in_queue` (not `num_requests_waiting`) as a more honest signal under long-output traffic.

## References

- KEDA — https://keda.sh/docs/
- KEDA + vLLM autoscaling tutorial — https://docs.vllm.ai/projects/production-stack/en/latest/use_cases/autoscaling-keda.html
- llm-d Variant Autoscaler — https://llm-d.ai/docs/architecture
- NVIDIA Dynamo SLO Planner — https://developer.nvidia.com/blog/nvidia-dynamo-1-production-ready/
