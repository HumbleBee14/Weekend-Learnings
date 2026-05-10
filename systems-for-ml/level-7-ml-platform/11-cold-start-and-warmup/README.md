# 11 — Cold Start and Warmup

## Files

- `CONCEPTS.md` — the seven cold-start phases with typical times, the 2026 mitigation stack (Run:ai Model Streamer, Tensorizer, ServerlessLLM, CRIU/cuda-checkpoint, warm pools, image pre-pull, NVMe weight cache), MIG vs MPS vs time-slicing.
- `measure_cold_start.py` — runs `vllm serve`, parses log markers, emits a CSV row of phase timings.
- `warmup_requests.py` — fires synthetic prompts at multiple input lengths to trigger CUDA graph capture before live traffic.
- `image-prepull-daemonset.yaml` — DaemonSet that caches the inference image on every GPU node.

## Quickstart

```bash
# 1. Naive cold start.
python measure_cold_start.py --model meta-llama/Llama-3.2-1B-Instruct --label naive

# 2. With Run:ai Model Streamer.
pip install runai-model-streamer
python measure_cold_start.py --model meta-llama/Llama-3.2-1B-Instruct \
    --extra-args "--load-format runai_streamer" \
    --label runai_streamer

# 3. Warmup synthetic prompts after server ready.
python warmup_requests.py --base http://localhost:8000 --model meta-llama/Llama-3.2-1B-Instruct
```

## Expected output

```
=== cold-start phases (seconds since process_started) ===
  process_started          0.00
  torch_imported           1.32
  model_load_start         3.10
  model_load_done         63.10   <- naive HF loader
  graph_capture_start     63.50
  graph_capture_done      72.10
  server_ready            72.30
  health_ok               73.00

CSV (naive): naive,0.00,1.32,3.10,63.10,63.50,72.10,72.30,73.00
```

After `--load-format runai_streamer` the `model_load_done` row should drop ~40-60%.

## Try

- **Stack mitigations.** Re-measure with image pre-pulled (DaemonSet running) + Run:ai streamer + warmup. Total cold-start should be roughly halved vs naive.
- **G15.** Drive sustained peak load on one replica. Force a scale-up. Capture: KEDA fire time, new pod scheduled, weights load done, server ready, first served request, SLO restored. Plot.
- **Scale-to-zero attempt.** Set `minReplicas: 0` in Topic 10's manifest. Stop traffic for 10 minutes. Send a single request. Time it. Compare to `minReplicas: 1`. Conclude (without surprise) that scale-to-zero is unworkable for 70B serving.
- **MIG slice for a 7B.** On an A100/H100, partition into MIG slices. Boot a 7B model on a 1g.10gb slice and re-measure cold start vs full GPU. Smaller GPU = faster cold start = the case for tier-by-MIG.

## Where this goes

- Topic 10: KEDA's reaction time is dominated by these phases. Together they fix scale-up's worst case.
- Topic 12: LMCache's persistent KV tier means a cold-started replica can re-acquire prefix KV without re-prefilling — a different kind of warmup.
- Topic 13: cold pool of `minReplicas` is a $/Mtok line item; quantify it.

## References

- NVIDIA Run:ai Model Streamer — https://developer.nvidia.com/blog/reducing-cold-start-latency-for-llm-inference-with-nvidia-runai-model-streamer/
- Tensorizer — https://github.com/coreweave/tensorizer
- ServerlessLLM (OSDI '24) — https://www.usenix.org/conference/osdi24/presentation/fu
- NVIDIA cuda-checkpoint — https://github.com/NVIDIA/cuda-checkpoint
- NVIDIA MIG — https://docs.nvidia.com/datacenter/tesla/mig-user-guide/
