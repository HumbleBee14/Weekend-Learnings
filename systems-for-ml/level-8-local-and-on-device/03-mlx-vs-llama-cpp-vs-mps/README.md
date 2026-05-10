# 03 — MLX vs llama.cpp Metal vs PyTorch MPS

## Files

- `CONCEPTS.md` — substrate-level differences, throughput numbers (M3 / M5), why the gap, when to pick which, benchmarking pitfalls.
- `benchmark.py` — three-way bench on the same 7B-class model. Greedy decoding, fixed prompt, prints TTFT and tok/s. Produces G18.

## Quickstart

```bash
pip install mlx mlx-lm llama-cpp-python torch transformers
huggingface-cli download mlx-community/Qwen2.5-7B-Instruct-4bit
huggingface-cli download Qwen/Qwen2.5-7B-Instruct-GGUF qwen2.5-7b-instruct-q4_k_m.gguf
GGUF_PATH=$(huggingface-cli scan-cache | grep qwen2.5-7b-instruct-q4_k_m | awk '{print $NF}') \
  python benchmark.py
```

## Expected output (M3 Max 64GB, fans on, AC power)

```
[mlx] loading...
  -> mlx-lm       TTFT=  120 ms  decode= 230.4 tok/s  (256 tokens)
[llama.cpp] loading (n_gpu_layers=-1 for full Metal offload)...
  -> llama.cpp    TTFT=  140 ms  decode= 150.2 tok/s  (256 tokens)
[mps] loading Qwen/Qwen2.5-7B-Instruct in fp16 ...
  -> torch-mps    TTFT=  410 ms  decode=  55.1 tok/s  (256 tokens)

=== summary ===
backend       TTFT (ms)    tok/s
mlx-lm              120    230.4
llama.cpp           140    150.2
torch-mps           410     55.1
```

## Try

- Repeat with a 13B model. The MLX-llama.cpp gap usually narrows slightly; the MPS gap stays bad.
- Repeat after running `mx.fast.matmul`-aware build on M5 hardware. MLX should jump to ~310 tok/s.
- Drop max_tokens to 32 to see TTFT-dominated behavior; raise to 1024 to see decode-dominated behavior.
- Compare peak resident memory with `/usr/bin/time -l ./run.sh`. PyTorch MPS will be ~50% higher than MLX on the same model.

## Pitfalls

- The torch-mps run loads the **fp16** model because MPS does not have a true 4-bit kernel matching MLX/llama.cpp. The comparison is honest because it shows the path a PyTorch MPS user would actually run.
- Quality drift: 4-bit MLX vs 4-bit GGUF is *not* identical. If you want a quality-matched comparison, run lm-eval-harness on both first.

## Where this goes

This is G18 for `reports/local.md`. Topic 04 explains why MLX widens further on M5 (Neural Accelerators). Topic 08 turns this into a serving comparison (Ollama-MLX vs `llama-server`).
