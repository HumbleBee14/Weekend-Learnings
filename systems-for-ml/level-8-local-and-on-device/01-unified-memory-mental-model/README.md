# 01 — Unified Memory Mental Model

## Files

- `CONCEPTS.md` — UMA vs discrete GPU; why KV cache, mmap loads, and tokenizer placement all change on Apple Silicon.
- `measure_uma.py` — probe CPU-alone, GPU-alone, and contended bandwidth on your Mac.

## Quickstart

```bash
pip install mlx numpy
python measure_uma.py
```

## Expected output

On an M3 Max 64GB:

```
== UMA bandwidth probe ==

CPU alone:                  ~80 GB/s
GPU alone (matmul-implied): ~310 GB/s
CPU under contention:       ~55 GB/s   (69% of alone)
GPU under contention:       ~210 GB/s  (68% of alone)
Sum during contention:      ~265 GB/s
```

Numbers vary by chip. The pattern is the same: when CPU and GPU run at full tilt at the same time, neither gets its full quota — they share one memory controller.

## Try

- Run a `mlx_lm.generate` in another terminal while `measure_uma.py` runs. Watch GPU bandwidth drop further; that is the cost of any other GPU consumer.
- Open Activity Monitor's GPU history pane while the script runs. Verify the GPU power draw spike lines up with the matmul phase.
- Set `OLLAMA_KEEP_ALIVE=0` and `mlx_lm` will release weights between calls. Compare the cold call (mmap fault) vs warm call.

## Where this goes

This mental model is load-bearing for every later topic. Topic 02 (MLX basics) builds on the lazy-eval / shared-buffer story; Topic 09 (MoE) and Topic 10 (KV quant) both reduce per-token bandwidth, which is the real lever this chapter exposes.
