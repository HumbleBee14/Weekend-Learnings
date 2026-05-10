"""
Measure cold-start phases on your local vLLM start.

Pattern: read /metrics and the engine's stdout to capture phase boundaries.
This is a thin runner that wraps `vllm serve`, prints timestamps, and emits
a CSV row at the end you can paste into your G15 plot.

Usage:
    python measure_cold_start.py --model meta-llama/Llama-3.2-1B-Instruct
    python measure_cold_start.py --model meta-llama/Llama-3.2-1B-Instruct \\
        --extra-args "--load-format runai_streamer"
"""

import argparse
import shlex
import subprocess
import time
import sys
import urllib.request
import urllib.error


PHASES = [
    ("process_started", "process_started"),
    ("torch_imported", "import torch"),
    ("model_load_start", "Loading model"),
    ("model_load_done", "Model loaded"),
    ("graph_capture_start", "Capturing CUDA graphs"),
    ("graph_capture_done", "Capturing finished"),
    ("server_ready", "Uvicorn running on"),
]


def health_ok(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://localhost:{port}/health", timeout=1) as r:
            return r.status == 200
    except (urllib.error.URLError, ConnectionError, TimeoutError):
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--extra-args", default="")
    p.add_argument("--label", default="naive", help="row label for the CSV output")
    args = p.parse_args()

    cmd = (
        ["vllm", "serve", args.model, "--port", str(args.port)]
        + shlex.split(args.extra_args)
    )
    print("> " + " ".join(shlex.quote(c) for c in cmd))

    t0 = time.perf_counter()
    times: dict[str, float] = {"process_started": 0.0}
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )

    try:
        for line in proc.stdout:
            sys.stdout.write(line)
            elapsed = time.perf_counter() - t0
            for key, marker in PHASES:
                if key not in times and marker in line:
                    times[key] = elapsed

        # When stdout closes (or after 'server_ready') do a /health probe to time first 200.
    except KeyboardInterrupt:
        proc.terminate()
    finally:
        # Run a 5-minute health probe race.
        t_ready = None
        for _ in range(300):
            if health_ok(args.port):
                t_ready = time.perf_counter() - t0
                break
            time.sleep(1)
        if t_ready is not None:
            times["health_ok"] = t_ready

    print("\n=== cold-start phases (seconds since process_started) ===")
    for k, _ in PHASES + [("health_ok", "health_ok")]:
        if k in times:
            print(f"  {k:<22} {times[k]:7.2f}")
    csv = ",".join(f"{times.get(k, ''):.2f}" if k in times else ""
                   for k, _ in PHASES + [("health_ok", "health_ok")])
    print(f"\nCSV ({args.label}): {args.label},{csv}")


if __name__ == "__main__":
    main()
