#!/usr/bin/env bash
# Drive llama-bench across CPU and Metal backends to measure the regime
# where each wins. Requires llama.cpp's `llama-bench` on PATH and a GGUF.
#
# Usage:
#   MODEL=/path/to/model.gguf bash bench_cpu_vs_gpu.sh
#
# Reports prompt-processing (pp512) and token-generation (tg128) tok/s.

set -euo pipefail

MODEL="${MODEL:-}"
if [[ -z "$MODEL" ]]; then
  echo "set MODEL=/path/to/model.gguf" >&2
  exit 1
fi
if [[ ! -f "$MODEL" ]]; then
  echo "model not found: $MODEL" >&2
  exit 1
fi
if ! command -v llama-bench >/dev/null 2>&1; then
  echo "llama-bench not on PATH. brew install llama.cpp or build from source." >&2
  exit 1
fi

P_CORES="$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || echo 8)"
TOTAL_CPUS="$(sysctl -n hw.ncpu 2>/dev/null || echo 8)"

echo
echo "model: $MODEL"
echo "P-cores: $P_CORES   total CPUs: $TOTAL_CPUS"
echo

echo "=== Metal (full GPU offload) ==="
llama-bench -m "$MODEL" -ngl 99 -p 512 -n 128 -r 3 || true
echo

echo "=== CPU only, threads = P-cores ($P_CORES) ==="
llama-bench -m "$MODEL" -ngl 0 -t "$P_CORES" -p 512 -n 128 -r 3 || true
echo

echo "=== CPU only, threads = total CPUs ($TOTAL_CPUS, includes E-cores) ==="
llama-bench -m "$MODEL" -ngl 0 -t "$TOTAL_CPUS" -p 512 -n 128 -r 3 || true
echo

HALF=$(( P_CORES / 2 ))
if (( HALF >= 2 )); then
  echo "=== CPU only, threads = P-cores/2 ($HALF) ==="
  llama-bench -m "$MODEL" -ngl 0 -t "$HALF" -p 512 -n 128 -r 3 || true
  echo
fi

echo "Read pp512 as compute-bound prefill, tg128 as bandwidth-bound decode."
echo "Adding E-cores typically regresses throughput on Apple Silicon."
